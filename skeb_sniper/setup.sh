#!/bin/bash

# Function to clean up temporary directory and container
cleanup() {
  if [ -d "$TEMP_DIR" ]; then
    echo "Cleaning up temporary directory $TEMP_DIR..."
    rm -rf "$TEMP_DIR"
  fi
  if docker ps -a -q -f name=temp > /dev/null 2>&1; then
    echo "Removing temporary container 'temp'..."
    docker stop temp > /dev/null 2>&1
    docker rm temp > /dev/null 2>&1
  fi
}

# Set trap to run cleanup on script exit (success or failure)
trap cleanup EXIT

# Load tar file if it exists, but continue if not found
TAR_FILE="skeb-sniper-linux.tar"
if [ -f "$TAR_FILE" ]; then
  echo "Loading $TAR_FILE..."
  docker load -i "$TAR_FILE"
else
  echo "Warning: $TAR_FILE not found. Proceeding with existing image if available."
fi

# Change the image tag to skeb-sniper for convenience
docker tag luckywah/skeb-sniper skeb-sniper 2>/dev/null || echo "Tag not updated; proceeding anyway."

# Pre-create the target directory with correct ownership
echo "Preparing target directory..."
mkdir -p "$HOME/skeb_sniper_data/app"
sudo chown "$(whoami):$(whoami)" "$HOME/skeb_sniper_data/app" 2>/dev/null || echo "Warning: Could not change ownership; may need sudo."

# Run a minimal temporary container named 'temp' to copy files
echo "Starting temporary container 'temp' to copy files..."
docker run -d --name temp skeb-sniper

# Copy files to a temporary location first, then move to final destination
echo "Copying files from temp container to a temporary location..."
TEMP_DIR="/tmp/skeb_sniper_temp_$$"
mkdir -p "$TEMP_DIR"
docker cp temp:/app "$TEMP_DIR"
if [ $? -eq 0 ]; then
  echo "Moving files from temporary location to $HOME/skeb_sniper_data/app..."
  cp -r "$TEMP_DIR/app"/* "$HOME/skeb_sniper_data/app/" 2>/dev/null || echo "Note: /app may be empty; continuing."
  echo "Files copied to $HOME/skeb_sniper_data/app (directory may be empty if /app in container is empty)."
else
  echo "Error: Failed to copy files from container. Check container and permissions."
  exit 1
fi

# Stop and remove any existing container
echo "Stopping and removing existing container if any..."
docker stop skeb-sniper-container 2>/dev/null
docker rm skeb-sniper-container 2>/dev/null

# Run the container
echo "Starting skeb-sniper container..."
sudo docker run -d --restart=always --name skeb-sniper-container \
  --network host \
  -e DISPLAY="$DISPLAY" \
  -v "$HOME/.Xauthority:/root/.Xauthority:rw" \
  -v "$HOME/skeb_sniper_data/app:/app" \
  -v "$HOME/skeb_sniper_firefox:/root/.mozilla/firefox" \
  -v "$HOME/skeb_sniper_config:/root/.config/skebsniper" \
  -v "/etc/machine-id:/etc/machine-id:ro" \
  skeb-sniper

if [ $? -eq 0 ]; then
  echo "Container started. Setup completed."
else
  echo "Error: Failed to start the container. Check Docker logs with 'docker logs skeb-sniper-container'."
  exit 1
fi
