#!/usr/bin/env bash
set -euo pipefail

RULES_URL="https://raw.githubusercontent.com/realsenseai/librealsense/master/config/99-realsense-libusb.rules"
RULES_PATH="/etc/udev/rules.d/99-realsense-libusb.rules"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl command not found. Please install curl first." >&2
  exit 1
fi

run_as_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
    return
  fi

  if command -v sudo >/dev/null 2>&1; then
    sudo "$@"
    return
  fi

  echo "This script needs root privileges to write $RULES_PATH." >&2
  echo "Run it as root or install sudo." >&2
  exit 1
}

tmpfile="$(mktemp)"
trap 'rm -f "$tmpfile"' EXIT

echo "Downloading RealSense udev rules from $RULES_URL"
curl -fsSL "$RULES_URL" -o "$tmpfile"

echo "Installing udev rules to $RULES_PATH"
run_as_root install -m 0644 "$tmpfile" "$RULES_PATH"

if command -v udevadm >/dev/null 2>&1; then
  echo "Reloading udev rules"
  run_as_root udevadm control --reload-rules
  run_as_root udevadm trigger
fi

echo "Done."
