# installer/mv3dt_installer/assets/scripts/lib/common.sh
#
# Shared helpers for bundled bash fragments staged out of the mv3dt-installer
# binary (doc 00 §4.2). Sourced by every script under
# installer/mv3dt_installer/assets/scripts/. Never executed on its own.
#
# This is a purpose-written successor to the removed
# laptop/scripts/lib/common.sh, not a copy of it: the logging block below is
# carried over verbatim, but the two functions that assumed a repo checkout
# on disk (repo_root, load_env) are replaced with staged-asset equivalents
# (asset_root, load_installer_conf).
# Everything laptop/scripts/lib/common.sh had for bootstrap-phase state,
# .deb discovery, version comparison, and config-file announcement is owned
# by state.py / report.py in the Python layer now and is not ported here.
#
# Isolation: this file only reads the staged asset tree
# (`$MV3DT_ASSET_ROOT`) and the installer.conf path handed to it
# (`$MV3DT_INSTALLER_CONF`). It never reaches into a repo checkout.

# Guard: cowardly refuse if the caller ran `bash common.sh` directly.
if [[ -z "${BASH_SOURCE[0]:-}" || "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "installer/mv3dt_installer/assets/scripts/lib/common.sh must be sourced, not executed." >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

if [[ -t 2 ]]; then
  _C_RED=$'\033[31m'
  _C_YEL=$'\033[33m'
  _C_GRN=$'\033[32m'
  _C_DIM=$'\033[2m'
  _C_RST=$'\033[0m'
else
  _C_RED=""; _C_YEL=""; _C_GRN=""; _C_DIM=""; _C_RST=""
fi

_log_script_tag() {
  local src="${BASH_SOURCE[2]:-${BASH_SOURCE[1]:-$0}}"
  printf '%s' "$(basename "$src")"
}

log_info()  { printf '%s[info ]%s %s: %s\n'  "$_C_GRN" "$_C_RST" "$(_log_script_tag)" "$*" >&2; }
log_warn()  { printf '%s[warn ]%s %s: %s\n'  "$_C_YEL" "$_C_RST" "$(_log_script_tag)" "$*" >&2; }
log_error() { printf '%s[error]%s %s: %s\n'  "$_C_RED" "$_C_RST" "$(_log_script_tag)" "$*" >&2; }
log_debug() {
  [[ "${LAPTOP_DEBUG:-0}" -eq 1 ]] || return 0
  printf '%s[debug]%s %s: %s\n' "$_C_DIM" "$_C_RST" "$(_log_script_tag)" "$*" >&2
}

die() {
  log_error "$*"
  exit 1
}

# ---------------------------------------------------------------------------
# Preflight helpers
# ---------------------------------------------------------------------------

require_root() {
  if [[ "$(id -u)" -ne 0 ]]; then
    die "This script must be run as root (use sudo)."
  fi
}

require_tool() {
  local tool="$1"
  if ! command -v "$tool" >/dev/null 2>&1; then
    die "Required tool '$tool' not found on PATH."
  fi
}

# ---------------------------------------------------------------------------
# Staged asset tree
# ---------------------------------------------------------------------------
#
# asset_root: absolute path to the root of the staged assets tree that this
# fragment was run from (doc 00 §4.2, mv3dt_installer/shellout.py). Unlike
# laptop/scripts/lib/common.sh's repo_root(), this is not derived from this
# file's own location -- it is always exported by shellout.run_bundled_script
# as MV3DT_ASSET_ROOT and means the same thing regardless of which subtree
# was staged (`tree=("scripts",)` or `tree=()`), so a fragment addresses a
# sibling subtree as `$(asset_root)/<subtree>/...`.
asset_root() {
  : "${MV3DT_ASSET_ROOT:?MV3DT_ASSET_ROOT is not set; this fragment must be run via shellout.run_bundled_script with a staged tree.}"
  printf '%s\n' "$MV3DT_ASSET_ROOT"
}

# ---------------------------------------------------------------------------
# installer.conf loader
# ---------------------------------------------------------------------------
#
# Loads <install_dir>/installer.conf (doc 00 §11.2), the persisted KEY=VALUE
# file mv3dt_installer/config.py owns. The path is not derived here -- it is
# handed down by the caller as MV3DT_INSTALLER_CONF, since only the Python
# layer knows the resolved install_dir. Plain KEY=VALUE, no quoting/
# expansion (same shape as laptop/config/laptop.env.example), so `set -a`
# exports every variable it defines.
load_installer_conf() {
  : "${MV3DT_INSTALLER_CONF:?MV3DT_INSTALLER_CONF is not set; the caller must pass the resolved installer.conf path.}"
  if [[ ! -f "$MV3DT_INSTALLER_CONF" ]]; then
    die "installer.conf not found at $MV3DT_INSTALLER_CONF."
  fi
  log_debug "Loading $MV3DT_INSTALLER_CONF"
  # shellcheck disable=SC1090
  set -a; . "$MV3DT_INSTALLER_CONF"; set +a
}
