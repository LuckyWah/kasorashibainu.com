#!/bin/bash

# Check if container is running
if ! docker ps -q -f name=skeb-sniper-container | grep -q .; then
    echo "Container not running. Please run './setup.sh' first."
    exit 1
fi

# Get the Xauth key from the host for the current DISPLAY
XAUTH_KEY=$(xauth list "$DISPLAY" | awk '{print $3}')
if [ -z "$XAUTH_KEY" ]; then
    echo "Error: No Xauth key found for $DISPLAY on the host. Check X11 forwarding setup."
    exit 1
fi

# Run commands in the container
docker exec -it -e DISPLAY="localhost:10.0" -e XAUTHORITY="/tmp/.Xauthority" skeb-sniper-container /bin/bash -c "
  # Remove any existing temporary Xauthority file (if possible)
  rm -f /tmp/.Xauthority 2>/dev/null || true
  # Create a new Xauthority file and add the host's key
  xauth -f /tmp/.Xauthority add localhost:10.0 . \"$XAUTH_KEY\"
  # Add the container's resolved display name as a fallback
  xauth -f /tmp/.Xauthority add \$(hostname)/unix:10 . \"$XAUTH_KEY\"
  # Verify the Xauth data
  echo \"Container Xauth list after add:\"
  xauth -f /tmp/.Xauthority list
  echo \"DISPLAY in container: \$DISPLAY\"
  echo \"XAUTHORITY in container: \$XAUTHORITY\"
  # Launch the GUI
  /app/Skeb_Sniper gui
"
