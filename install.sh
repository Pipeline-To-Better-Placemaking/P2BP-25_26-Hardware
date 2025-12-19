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
  EXISTING_API_KEY=$(grep "^API_KEY=" "$ENV_FILE" | cut -d'=' -f2)
  EXISTING_ENDPOINT=$(grep "^ENDPOINT=" "$ENV_FILE" | cut -d'=' -f2)
  
  read -p "Do you want to update the API credentials? (y/n): " UPDATE_CREDS
  
  if [[ ! "$UPDATE_CREDS" =~ ^[Yy]$ ]]; then
    echo "Keeping existing API credentials"
    API_KEY="$EXISTING_API_KEY"
    API_ENDPOINT="$EXISTING_ENDPOINT"
  else
    read -s -p "Enter new API Key: " API_KEY
    echo
    read -p "Enter new API Endpoint URL: " API_ENDPOINT
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
sudo tee "$ENV_FILE" > /dev/null <<EOF
API_KEY=$API_KEY
ENDPOINT=$API_ENDPOINT
EOF

# Secure permissions
sudo chown root:root "$ENV_FILE"
sudo chmod 600 "$ENV_FILE"

echo "API credentials saved to $ENV_FILE"

# Python packages
echo "Installing Python dependencies..."
sudo pip3 install --upgrade pip
sudo pip3 install -r requirements.txt

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

echo "Installation complete."