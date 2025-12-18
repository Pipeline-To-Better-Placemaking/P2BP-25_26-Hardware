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

# Python packages
echo "Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install --break-system-packages -r requirements.txt
pip3 install --break-system-packages .

# Create canonical directory and structure
echo "Creating application directories..."
sudo mkdir -p $APP_ROOT/scripts
sudo mkdir -p $APP_ROOT/config

# Install scripts
echo "Installing scripts..."
sudo rsync -a --delete scripts/ $APP_ROOT/scripts/

# Set app root directory permissions
sudo chown -R root:root /opt/p2bp
sudo chmod -R 755 /opt/p2bp

# Install systemd units
echo "Installing systemd services..."
sudo cp services/*.service /etc/systemd/system/

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

echo "Installation complete."