#!/usr/bin/env bash

# This script is meant to run on the Jetson when the repo is cloned. The end goal is to have all 
# dependencies installed and the services moved to systemd so that it can manage the scripts.

set -e

APP_ROOT=/opt/p2bp/camera

# Verify required files exist
if [ ! -f "requirements.txt" ]; then
  echo "Error: requirements.txt not found"
  exit 1
fi

if [ ! -d "services" ]; then
  echo "Error: services directory not found"
  exit 1
fi

if [ ! -d "scripts" ]; then
  echo "Error: scripts directory not found"
  exit 1
fi

# ---------------------------------------------------------------------------
# Install profile resolution
#
# A Jetson runs one of two workloads:
#   * camera (default): camera-scanner / tracker / homography stack
#   * lidar           : additionally runs scan-orchestrator and drives a
#                       Raspberry Pi companion over SSH.
#
# Resolution priority (first match wins):
#   1. --profile=VAL or --profile VAL on the command line
#   2. P2BP_PROFILE env var
#   3. /etc/p2bp/profile (persisted by a previous install run)
#   4. default 'camera'; if the PCI iface comes up cleanly and arp-scan
#      spots a Raspberry Pi OUI on it, upgrade to 'lidar' automatically.
#
# The two camera Jetsons therefore keep today's behavior exactly: run
# `sudo ./install.sh` with no args and no env, get the camera path.
# ---------------------------------------------------------------------------

PROFILE=""
PROFILE_EXPLICIT=false

# Accept --profile=VAL or --profile VAL; ignore any other positional args.
_remaining_args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --profile=*)
      _val="${1#--profile=}"
      case "$_val" in
        lidar|camera) PROFILE="$_val"; PROFILE_EXPLICIT=true ;;
        *) echo "Warning: ignoring unrecognized --profile=$_val" ;;
      esac
      shift
      ;;
    --profile)
      _val="${2:-}"
      case "$_val" in
        lidar|camera) PROFILE="$_val"; PROFILE_EXPLICIT=true ;;
        *) echo "Warning: ignoring unrecognized --profile '$_val'" ;;
      esac
      shift 2 || shift
      ;;
    *)
      _remaining_args+=("$1")
      shift
      ;;
  esac
done
if [ ${#_remaining_args[@]} -gt 0 ]; then
  set -- "${_remaining_args[@]}"
else
  set --
fi
unset _remaining_args _val

if [ -z "$PROFILE" ] && [ -n "${P2BP_PROFILE:-}" ]; then
  case "$P2BP_PROFILE" in
    lidar|camera) PROFILE="$P2BP_PROFILE"; PROFILE_EXPLICIT=true ;;
    *) echo "Warning: ignoring unrecognized P2BP_PROFILE='$P2BP_PROFILE'" ;;
  esac
fi

if [ -z "$PROFILE" ] && [ -r /etc/p2bp/profile ]; then
  _persisted=$(tr -d '[:space:]' < /etc/p2bp/profile 2>/dev/null || true)
  case "$_persisted" in
    lidar|camera)
      PROFILE="$_persisted"
      PROFILE_EXPLICIT=true
      echo "Loaded profile '$PROFILE' from /etc/p2bp/profile"
      ;;
  esac
  unset _persisted
fi

if [ -z "$PROFILE" ]; then
  PROFILE=camera
  PROFILE_EXPLICIT=false
fi

if [ "$PROFILE_EXPLICIT" = true ]; then
  echo "Install profile: $PROFILE (explicit)"
else
  echo "Install profile: $PROFILE (default; auto-detect may upgrade to 'lidar' after iface config)"
fi

echo "Installing P2BP Camera stack..."

sudo apt install python3-pip -y
sudo apt install -y python3-opencv
sudo apt install -y arp-scan
sudo apt install -y ffmpeg
sudo apt install -y libavif13
sudo apt install -y logrotate

# Create canonical directory and structure
echo "Creating application directories..."
sudo mkdir -p $APP_ROOT/scripts
sudo mkdir -p $APP_ROOT/config
sudo mkdir -p $APP_ROOT/services
sudo mkdir -p $APP_ROOT/models

# Configure environment variables
echo "Configuring API credentials..."

ENV_FILE="$APP_ROOT/config/agent.env"
EXISTING_API_KEY=""
EXISTING_ENDPOINT=""

# Check if env file already exists and load values

if [ -f "$ENV_FILE" ]; then
  echo "Existing API credentials found."
  EXISTING_API_KEY_LINE=$(grep -m1 "^API_KEY=" "$ENV_FILE" || true)
  EXISTING_ENDPOINT_LINE=$(grep -m1 "^ENDPOINT=" "$ENV_FILE" || true)
  EXISTING_API_KEY="${EXISTING_API_KEY_LINE#API_KEY=}"
  EXISTING_ENDPOINT="${EXISTING_ENDPOINT_LINE#ENDPOINT=}"

  # Prompt to update API Key
  read -p "Do you want to update the API Key? (y/n): " UPDATE_API_KEY
  if [[ "$UPDATE_API_KEY" =~ ^[Yy]$ ]]; then
    read -s -p "Enter new API Key: " API_KEY
    echo
  else
    echo "Keeping existing API Key"
    API_KEY="$EXISTING_API_KEY"
  fi

  # Prompt to update API Endpoint
  read -p "Do you want to update the API Endpoint? (y/n): " UPDATE_API_ENDPOINT
  if [[ "$UPDATE_API_ENDPOINT" =~ ^[Yy]$ ]]; then
    read -p "Enter new API Endpoint URL: " API_ENDPOINT
  else
    echo "Keeping existing API Endpoint"
    API_ENDPOINT="$EXISTING_ENDPOINT"
  fi
else
  read -s -p "Enter API Key: " API_KEY
  echo
  read -p "Enter API Endpoint URL: " API_ENDPOINT
fi

if [ -z "$API_KEY" ] || [ -z "$API_ENDPOINT" ]; then
  echo "Error: API Key and Endpoint must not be empty"
  exit 1
fi

# Normalize endpoint so scripts can safely append routes.
# Users sometimes paste URLs like https://host/api or https://host/api/.
API_ENDPOINT="${API_ENDPOINT%/}"
if [[ "${API_ENDPOINT,,}" == */api ]]; then
  API_ENDPOINT="${API_ENDPOINT%/api}"
  API_ENDPOINT="${API_ENDPOINT%/}"
fi

# Write environment file

# Write env atomically to avoid partial/corrupted files if the script fails later.
TMP_ENV_FILE=$(sudo mktemp "$APP_ROOT/config/agent.env.tmp.XXXXXX")
sudo tee "$TMP_ENV_FILE" > /dev/null <<EOF
API_KEY=$API_KEY
ENDPOINT=$API_ENDPOINT
EOF
sudo mv -f "$TMP_ENV_FILE" "$ENV_FILE"

# Secure permissions
sudo chown root:root "$ENV_FILE"
sudo chmod 600 "$ENV_FILE"

echo "API credentials saved to $ENV_FILE"

# Python packages
echo "Installing Python dependencies..."
sudo pip3 install --upgrade pip
sudo pip3 install python-dotenv requests psutil

# Jetson images often ship python3-sympy via apt (distutils-installed). Pip cannot uninstall
# those cleanly, so install our pinned SymPy over it explicitly.
sudo pip3 install --ignore-installed sympy==1.12.1 mpmath==1.3.0

sudo pip3 install -r requirements.txt --no-deps

# Install Playwright browsers
echo "Installing Playwright browsers..."
sudo playwright install

# Install scripts
echo "Installing scripts..."
sudo rsync -a --delete scripts/ $APP_ROOT/scripts/

# Install models (OSNet + YOLO weights)
if [ -d "models" ]; then
  echo "Installing models..."
  sudo rsync -a --delete models/ $APP_ROOT/models/
elif [ -d "osnet" ]; then
  # Back-compat: older repo layout had ./osnet at the root.
  echo "Installing models (legacy osnet layout detected)..."
  sudo mkdir -p $APP_ROOT/models/osnet
  sudo rsync -a --delete osnet/ $APP_ROOT/models/osnet/
else
  echo "Warning: models directory not found; skipping models install."
fi

# Set app root directory permissions
sudo chown -R root:root /opt/p2bp
sudo chmod -R 755 /opt/p2bp

# Install systemd units
echo "Installing systemd units..."

# Keep a copy of all unit files in the app directory (for debugging/versioning)
sudo rsync -a --delete services/ $APP_ROOT/services/

# Copy unit files into systemd (supports both .service and .path units)
unit_files=()
while IFS= read -r -d '' f; do
  unit_files+=("$f")
done < <(find services -maxdepth 1 -type f \( -name '*.service' -o -name '*.path' \) -print0)

if [ ${#unit_files[@]} -eq 0 ]; then
  echo "Warning: No systemd unit files found in services/"
else
  sudo cp "${unit_files[@]}" /etc/systemd/system/
fi

sudo systemctl daemon-reload

# Best-effort: ensure logrotate runs (Ubuntu may use a systemd timer).
sudo systemctl enable --now logrotate.timer >/dev/null 2>&1 || true

# Deploy tightened rsyslog logrotate config (daily + 100 MB cap, 3 rotations).
# Replaces the Ubuntu default which rotates weekly and can accumulate 10+ GB.
if [ -f "services/logrotate-rsyslog" ]; then
  sudo cp services/logrotate-rsyslog /etc/logrotate.d/rsyslog
  echo "Installed logrotate config for rsyslog."
else
  echo "Warning: services/logrotate-rsyslog not found; skipping rsyslog logrotate config."
fi

# Enable all services (Scripts each have their own conditions for starting)
echo "Enabling services..."
if [ -z "$(find services -name '*.service' 2>/dev/null)" ]; then
  echo "Warning: No .service files found in services/"
else
  for service in services/*.service; do
    service_name=$(basename "$service")

    # tracker.service is managed by tracker.path (flag-controlled). tracker-toggle.service
    # is a helper that should not be enabled. scan-orchestrator.service is profile-gated
    # and handled explicitly at the end of this script.
    case "$service_name" in
      tracker.service|tracker-toggle.service|scan-orchestrator.service)
        continue
        ;;
    esac
    sudo systemctl enable "$service_name"
    echo "  Enabled: $service_name"
  done
fi

echo "Enabling path units..."
if [ -z "$(find services -name '*.path' 2>/dev/null)" ]; then
  echo "Warning: No .path files found in services/"
else
  for path_unit in services/*.path; do
    path_name=$(basename "$path_unit")
    sudo systemctl enable --now "$path_name"
    echo "  Enabled: $path_name"
  done
fi

# Ensure tracker.service is not enabled (tracker.path controls it)
sudo systemctl disable tracker.service >/dev/null 2>&1 || true
sudo systemctl disable tracker-toggle.service >/dev/null 2>&1 || true

# ---------------------------------------------------------------------------
# Configure deterministic link-local IP for the PCI Ethernet interface.
#
# Moved ahead of the Pi companion block so $IFACE is available for ARP-based
# Pi discovery below.  Runs on every profile because both camera scanning and
# lidar Pi discovery need the PCI iface selected and up.
# ---------------------------------------------------------------------------

echo "Configuring deterministic link-local IP for the PCI Ethernet interface..."

# Some Jetson images name the onboard/PCI NIC differently (e.g. eno1, enP8p1s0, enp*).
# We prefer known names when present, otherwise auto-detect a PCI NIC. We intentionally
# avoid USB Ethernet adapters, which often appear as enx<MAC>.

select_pci_eth_iface() {
  # Prefer known/expected names first
  for candidate in "eno1" "enP8p1s0"; do
    if [ -e "/sys/class/net/$candidate/address" ]; then
      echo "$candidate"
      return 0
    fi
  done

  # Auto-detect a PCI NIC
  for iface_path in /sys/class/net/*; do
    iface=$(basename "$iface_path")

    # Skip loopback and common non-physical interfaces
    case "$iface" in
      lo|docker*|br-*|veth*|virbr*|vmnet*|zt*|tailscale*|wg*|tun*|tap*|wlan*|wl*)
        continue
        ;;
    esac

    # Skip USB ethernet (typically enx<MAC>)
    case "$iface" in
      enx*)
        continue
        ;;
    esac

    # Must have a MAC address
    [ -e "/sys/class/net/$iface/address" ] || continue

    # Prefer interfaces whose device subsystem is PCI
    if [ -e "/sys/class/net/$iface/device/subsystem" ]; then
      subsystem=$(basename "$(readlink -f "/sys/class/net/$iface/device/subsystem")")
      if [ "$subsystem" = "pci" ]; then
        echo "$iface"
        return 0
      fi
    fi
  done

  return 1
}

IFACE=""
IFACE=$(select_pci_eth_iface 2>/dev/null || true)
if [ -n "$IFACE" ] && [ -e "/sys/class/net/$IFACE/address" ]; then
  echo "Selected interface: $IFACE"

  MAC=$(tr -d ':' < "/sys/class/net/$IFACE/address")
  BYTE1=$((0x${MAC:8:2}))
  BYTE2=$((0x${MAC:10:2}))

  [ "$BYTE1" -eq 0 ] && BYTE1=1
  [ "$BYTE2" -eq 0 ] && BYTE2=1

  IP_ADDR="169.254.$BYTE1.$BYTE2"

  if command -v nmcli >/dev/null 2>&1 && systemctl is-active --quiet NetworkManager; then
    # Keep connection name stable even if IFACE changes.
    CON_NAME="p2bp-linklocal"
    if ! nmcli -t -f NAME connection show | grep -Fxq "$CON_NAME"; then
      sudo nmcli connection add type ethernet ifname "$IFACE" con-name "$CON_NAME" \
        ipv4.method manual ipv4.addresses "$IP_ADDR/16" ipv4.never-default yes \
        ipv6.method ignore connection.autoconnect yes >/dev/null
    else
      sudo nmcli connection modify "$CON_NAME" \
        connection.interface-name "$IFACE" \
        ipv4.method manual ipv4.addresses "$IP_ADDR/16" ipv4.never-default yes \
        ipv6.method ignore connection.autoconnect yes >/dev/null
    fi

    # Bring up only the selected PCI ethernet connection (should not impact USB Ethernet)
    sudo nmcli connection up "$CON_NAME" >/dev/null || true
    echo "Assigned $IP_ADDR/16 to $IFACE via NetworkManager ($CON_NAME)"
  else
    # Fallback: apply non-persistent address without restarting networking services
    sudo ip link set "$IFACE" up || true
    sudo ip addr add "$IP_ADDR/16" dev "$IFACE" 2>/dev/null || true
    echo "Assigned $IP_ADDR/16 to $IFACE (non-persistent fallback)"
  fi
else
  echo "Warning: No suitable PCI ethernet interface found; skipping link-local configuration."
fi

# ---------------------------------------------------------------------------
# Profile auto-detect (only when no explicit profile was provided)
#
# Default profile is 'camera'.  If nothing explicitly asked for 'lidar' and
# we have a PCI iface up, run a quick arp-scan for a Raspberry Pi OUI.  A
# positive match is the only thing that can flip 'camera' to 'lidar'; a
# negative match or a failed scan leaves the default alone.  This keeps
# camera Jetsons fully deterministic.
# ---------------------------------------------------------------------------

PI_MAC_OUI_REGEX='^(dc:a6:32|e4:5f:01):'

if [ "$PROFILE_EXPLICIT" = false ] && [ -n "$IFACE" ]; then
  echo ""
  echo "=== Profile Auto-Detection ==="
  echo "No explicit profile supplied; probing $IFACE for a Raspberry Pi (OUI match)..."
  if sudo arp-scan --interface="$IFACE" --localnet --quiet --ignoredups 2>/dev/null \
       | awk -v re="$PI_MAC_OUI_REGEX" 'tolower($2) ~ re { found=1 } END { exit !found }'; then
    echo "Raspberry Pi detected on $IFACE; upgrading profile from 'camera' to 'lidar'."
    PROFILE=lidar
  else
    echo "No Raspberry Pi detected; staying on 'camera' profile."
  fi
fi

# Persist the final profile so subsequent reruns are deterministic.
sudo install -d -m 0755 /etc/p2bp
echo "$PROFILE" | sudo tee /etc/p2bp/profile >/dev/null
echo "Profile: $PROFILE (persisted to /etc/p2bp/profile)"

# ---------------------------------------------------------------------------
# Raspberry Pi Companion Setup (lidar profile only)
#
# Discovers the Pi via arp-scan (preferring a persisted exact MAC, then Pi
# OUIs dc:a6:32 / e4:5f:01), falling back to the hardcoded 192.168.28.2 if
# discovery yields nothing.  Then deploys LidarScanV1.py to the Pi and
# installs its Python dependencies over SSH.  If the Pi is unreachable the
# rest of the Jetson install still succeeds.
# ---------------------------------------------------------------------------

if [ "$PROFILE" = "lidar" ]; then
  PI_HARDCODED_IP="192.168.28.2"
  PI_USER="pi"
  PI_MAC_HINT_FILE="/opt/p2bp/lidar/pi_mac.hint"
  PI_REMOTE_SCRIPT_DIR="/opt/p2bp/lidar/scripts"
  PI_SETUP_OK=false

  # scan-orchestrator.service has no User= directive, so systemd runs it as
  # root with HOME=/root.  `sudo ./install.sh` on Ubuntu preserves HOME from
  # the invoking user by default, so without this line ssh-keygen /
  # ssh-copy-id / ssh would read and write /home/<user>/.ssh/ and the
  # root-owned service would never see the resulting key.  Pin HOME to /root
  # for the duration of the lidar block so every $HOME/.ssh/ reference below
  # lands in the same directory the service uses at scan time.
  export HOME=/root
  mkdir -p "$HOME/.ssh"
  chmod 700 "$HOME/.ssh"

  echo ""
  echo "=== Raspberry Pi Companion Setup (Lidar) ==="

  # Step 0 — Pick the iface that actually routes to the Pi (if the kernel
  # already has a route), otherwise default to the PCI iface we just brought
  # up.  arp-scan needs a specific iface to probe on.
  PI_IFACE="${IFACE:-}"
  if _route_out=$(ip -o route get "$PI_HARDCODED_IP" 2>/dev/null); then
    _routed_iface=$(awk '{ for(i=1;i<=NF;i++) if($i=="dev") { print $(i+1); exit } }' <<<"$_route_out")
    [ -n "$_routed_iface" ] && PI_IFACE="$_routed_iface"
    unset _routed_iface
  fi
  unset _route_out

  # Step 0b — Discover Pi IP via ARP (exact MAC if we have a hint, else OUI
  # match).  If discovery fails, fall back to the hardcoded IP so this rig
  # always at least attempts what the old script did.
  PI_IP="$PI_HARDCODED_IP"
  PI_MAC=""

  if [ -n "$PI_IFACE" ]; then
    EXACT_MAC=""
    if [ -r "$PI_MAC_HINT_FILE" ]; then
      EXACT_MAC=$(tr '[:upper:]' '[:lower:]' < "$PI_MAC_HINT_FILE" | tr -d '[:space:]')
    fi

    echo "Scanning $PI_IFACE for Raspberry Pi..."
    arp_out=$(sudo arp-scan --interface="$PI_IFACE" --localnet --quiet --ignoredups 2>/dev/null || true)

    # If --localnet missed, retry on the Pi's expected /24 (works even when
    # the Jetson has no IP in 192.168.28.0/24 because arp-scan is L2).
    if [ -z "$arp_out" ] || ! echo "$arp_out" | awk -v re="$PI_MAC_OUI_REGEX" '
          tolower($2) ~ re { exit 0 } END { exit 1 }'; then
      arp_out=$(sudo arp-scan --interface="$PI_IFACE" --network=192.168.28.0/24 \
                  --quiet --ignoredups 2>/dev/null || true)
    fi

    match=""
    if [ -n "$arp_out" ] && [ -n "$EXACT_MAC" ]; then
      match=$(echo "$arp_out" | awk -v m="$EXACT_MAC" 'tolower($2) == m { print $1, $2; exit }')
    fi
    if [ -z "$match" ] && [ -n "$arp_out" ]; then
      match=$(echo "$arp_out" | awk -v re="$PI_MAC_OUI_REGEX" '
                tolower($2) ~ re { print $1, $2; exit }')
    fi

    if [ -n "$match" ]; then
      PI_IP="${match%% *}"
      PI_MAC="${match##* }"
      echo "  Discovered Raspberry Pi: $PI_IP ($PI_MAC) on $PI_IFACE"
    else
      echo "  No Raspberry Pi OUI match on $PI_IFACE; falling back to $PI_HARDCODED_IP"
    fi
    unset arp_out match EXACT_MAC
  else
    echo "  No PCI ethernet iface resolved; falling back to $PI_HARDCODED_IP for Pi SSH."
  fi

  PI_SSH="$PI_USER@$PI_IP"

  # Step 1 — Ensure an SSH key exists on this host
  if [ ! -f "$HOME/.ssh/id_ed25519" ] && [ ! -f "$HOME/.ssh/id_rsa" ]; then
    echo "No SSH key found; generating ed25519 key pair..."
    mkdir -p "$HOME/.ssh"
    chmod 700 "$HOME/.ssh"
    ssh-keygen -t ed25519 -N "" -f "$HOME/.ssh/id_ed25519"
  else
    echo "SSH key found."
  fi

  # Step 1b — If the sudo-invoking user already has passwordless SSH to the
  # Pi, use their existing trust to append root's public key to the Pi's
  # authorized_keys.  Turns the common "I've already ssh-copy-id'd as my
  # regular user" case into zero password prompts.  Any failure here falls
  # through silently to the Step 2 ssh-copy-id path.
  if [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != root ] \
     && [ -r "$HOME/.ssh/id_ed25519.pub" ] \
     && sudo -u "$SUDO_USER" ssh -o BatchMode=yes -o ConnectTimeout=5 \
          "$PI_SSH" true 2>/dev/null; then
    echo "Detected existing Pi SSH trust for $SUDO_USER; installing root's key via that trust..."
    if cat "$HOME/.ssh/id_ed25519.pub" \
         | sudo -u "$SUDO_USER" ssh -o BatchMode=yes "$PI_SSH" '
             set -e
             mkdir -p ~/.ssh
             chmod 700 ~/.ssh
             touch ~/.ssh/authorized_keys
             chmod 600 ~/.ssh/authorized_keys
             k=$(cat)
             grep -qxF "$k" ~/.ssh/authorized_keys || echo "$k" >> ~/.ssh/authorized_keys
           '; then
      echo "  Root's public key installed on Pi (no password prompt needed)."
      # Mirror known_hosts so root's first ssh doesn't hit hostkey verification.
      _sudo_home=$(getent passwd "$SUDO_USER" | awk -F: '{print $6}')
      if [ -n "$_sudo_home" ] && [ -r "$_sudo_home/.ssh/known_hosts" ]; then
        cat "$_sudo_home/.ssh/known_hosts" >> "$HOME/.ssh/known_hosts" 2>/dev/null || true
        sort -u "$HOME/.ssh/known_hosts" -o "$HOME/.ssh/known_hosts" 2>/dev/null || true
        chmod 600 "$HOME/.ssh/known_hosts" 2>/dev/null || true
      fi
      unset _sudo_home
    else
      echo "  Could not append root's key via $SUDO_USER's trust; will fall back to ssh-copy-id."
    fi
  fi

  # Step 2 — Test (and optionally establish) passwordless SSH to the Pi
  if ssh -o BatchMode=yes -o ConnectTimeout=10 "$PI_SSH" true 2>/dev/null; then
    echo "Passwordless SSH to $PI_SSH is working."
    PI_SETUP_OK=true
  else
    echo "Passwordless SSH to $PI_SSH failed; attempting ssh-copy-id..."
    echo "(You may be prompted for the Pi password once.)"
    if ssh-copy-id -o ConnectTimeout=10 "$PI_SSH"; then
      if ssh -o BatchMode=yes -o ConnectTimeout=10 "$PI_SSH" true 2>/dev/null; then
        echo "SSH key copied successfully."
        PI_SETUP_OK=true
      fi
    fi
  fi

  if [ "$PI_SETUP_OK" = false ]; then
    echo "Warning: Could not establish SSH to $PI_SSH. Skipping Pi lidar setup."
    echo "         Jetson-side install is complete; configure the Pi manually or re-run later."
  else
    # Run Pi deployment in a subshell so failures here never abort the rest of
    # the Jetson install (set -e is active in the outer script).
    (
      set -e

      # Step 3 — Deploy LidarScanV1.py to the Pi
      echo "Deploying LidarScanV1.py to $PI_SSH:$PI_REMOTE_SCRIPT_DIR/ ..."
      ssh "$PI_SSH" "sudo mkdir -p '$PI_REMOTE_SCRIPT_DIR'"

      PI_TMP=$(ssh "$PI_SSH" "mktemp /tmp/LidarScanV1.py.XXXXXX")
      scp -o ConnectTimeout=10 scripts/LidarScanV1.py "$PI_SSH:$PI_TMP"
      ssh "$PI_SSH" "sudo mv -f '$PI_TMP' '$PI_REMOTE_SCRIPT_DIR/LidarScanV1.py' && sudo chmod 755 '$PI_REMOTE_SCRIPT_DIR/LidarScanV1.py'"
      echo "  Deployed: $PI_REMOTE_SCRIPT_DIR/LidarScanV1.py"

      # Step 4 — Check and install Python dependencies on the Pi
      echo "Checking Python dependencies on the Pi..."
      ssh "$PI_SSH" '
        python3 -c "import serial" 2>/dev/null  || { echo "  Installing pyserial..."; sudo pip3 install pyserial; }
        python3 -c "import RPi.GPIO" 2>/dev/null || { echo "  Installing RPi.GPIO..."; sudo pip3 install RPi.GPIO; }
      '

      # Step 5 — Smoke test
      if ssh "$PI_SSH" 'python3 -c "import serial; import RPi.GPIO; print(\"OK\")"' 2>/dev/null; then
        echo "Pi lidar dependencies verified."
      else
        echo "Warning: Pi dependency smoke test failed. Check pyserial / RPi.GPIO on the Pi."
      fi

      # Step 6 — Persist the Pi MAC for exact-match discovery on subsequent
      # install runs.  Non-fatal; discovery falls back to OUI + hardcoded IP.
      if actual_mac=$(ssh -o BatchMode=yes "$PI_SSH" "cat /sys/class/net/eth0/address" 2>/dev/null); then
        actual_mac=$(echo "$actual_mac" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')
        if [ -n "$actual_mac" ]; then
          sudo install -d -m 0755 "$(dirname "$PI_MAC_HINT_FILE")"
          echo "$actual_mac" | sudo tee "$PI_MAC_HINT_FILE" >/dev/null
          echo "Pinned Pi MAC hint: $actual_mac -> $PI_MAC_HINT_FILE"
        fi
      fi

      echo "=== Pi Companion Setup Complete ==="
    ) || echo "Warning: Pi companion setup encountered errors. Jetson install continues."
  fi
else
  echo ""
  echo "Profile '$PROFILE' — skipping Raspberry Pi companion setup."
fi

# ---------------------------------------------------------------------------
# scan-orchestrator.service gating
#
# The generic service-enable loop earlier skipped this unit.  We enable it on
# the lidar profile and explicitly disable it on every other profile so a
# Jetson re-provisioned from lidar back to camera doesn't keep polling
# /api/scan/device/next-pending.
# ---------------------------------------------------------------------------

if [ "$PROFILE" = "lidar" ]; then
  sudo systemctl enable scan-orchestrator.service >/dev/null 2>&1 || true
  echo "  Enabled: scan-orchestrator.service (lidar profile)"
else
  sudo systemctl disable --now scan-orchestrator.service >/dev/null 2>&1 || true
  echo "  Disabled: scan-orchestrator.service (profile=$PROFILE)"
fi

echo "Installation complete."