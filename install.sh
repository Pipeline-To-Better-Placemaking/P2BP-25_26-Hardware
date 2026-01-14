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

echo "Installing P2BP Camera stack..."

sudo apt install python3-pip -y
sudo apt install -y python3-opencv

# Create canonical directory and structure
echo "Creating application directories..."
sudo mkdir -p $APP_ROOT/scripts
sudo mkdir -p $APP_ROOT/config
sudo mkdir -p $APP_ROOT/services

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

# Set app root directory permissions
sudo chown -R root:root /opt/p2bp
sudo chmod -R 755 /opt/p2bp

# Install systemd units
echo "Installing systemd services..."
sudo cp services/*.service /etc/systemd/system/
sudo cp services/*.service /opt/p2bp/camera/services/

sudo systemctl daemon-reload

# Enable all services (Scripts each have their own conditions for starting)
echo "Enabling services..."
if [ -z "$(find services -name '*.service' 2>/dev/null)" ]; then
  echo "Warning: No .service files found in services/"
else
  for service in services/*.service; do
    service_name=$(basename "$service")
    sudo systemctl enable "$service_name"
    echo "  Enabled: $service_name"
  done
fi

# Configure deterministic link-local IP for eno1 (camera network)
echo "Configuring deterministic link-local IP for eno1..."

IFACE="eno1"
if [ -e "/sys/class/net/$IFACE/address" ]; then
  MAC=$(cat "/sys/class/net/$IFACE/address" | tr -d ':')
  BYTE1=$((0x${MAC:8:2}))
  BYTE2=$((0x${MAC:10:2}))

  [ "$BYTE1" -eq 0 ] && BYTE1=1
  [ "$BYTE2" -eq 0 ] && BYTE2=1

  IP_ADDR="169.254.$BYTE1.$BYTE2"

  if command -v nmcli >/dev/null 2>&1 && systemctl is-active --quiet NetworkManager; then
    CON_NAME="eno1-linklocal"
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

    # Bring up only eno1 connection (should not impact USB Ethernet)
    sudo nmcli connection up "$CON_NAME" >/dev/null || true
    echo "Assigned $IP_ADDR/16 to $IFACE via NetworkManager ($CON_NAME)"
  else
    # Fallback: apply non-persistent address without restarting networking services
    sudo ip link set "$IFACE" up || true
    sudo ip addr add "$IP_ADDR/16" dev "$IFACE" 2>/dev/null || true
    echo "Assigned $IP_ADDR/16 to $IFACE (non-persistent fallback)"
  fi
else
  echo "Warning: $IFACE not found; skipping eno1 link-local configuration."
fi

echo "Installation complete."