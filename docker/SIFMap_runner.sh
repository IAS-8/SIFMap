#!/bin/bash

# Define variables
REPO_DIR=""  # Update this with your repository path
IMAGE_NAME="sifmap-image"
DATA_DIR=""
USER_DIR=""
DOCKERFILE="docker/Dockerfile"
SHM="100g"
DOCKER_CMD='podman' # choose docker or podman

INTERACTIVE=false
UPDATE=false
GITUPDATE=false
ARGS=()

# Function to display usage
usage() {
    echo "Usage: $0 [-u] [-g] [-i] [args]"
    echo "  -u    Rebuild the image before running"
    echo "  -g    Pull the newest git commit."
    echo "  -i    Run the container interactively"
    echo "  [args] Arguments to pass to the podman run command"
    exit 1
}

# Parse command-line options
while getopts "ugip:" opt; do
    case "$opt" in
        u)
            UPDATE=true
            ;;
	g)
            GITUPDATE=true
            ;;
        i)
            INTERACTIVE=true
            ;;
        p)
            PORT="$OPTARG"  # Capture the port argument
            ;;
        *)
            usage
            ;;
    esac
done

# Shift to process remaining arguments
shift $((OPTIND - 1))
ARGS=("$@")

# Update the repository if needed
if [ "$GITUPDATE" = true ]; then
    echo "Updating repository..."
    git -C "$REPO_DIR" pull || { echo "Git pull failed!"; exit 1; }
fi

if [ "$UPDATE" = true ]; then
    # Rebuild the Podman image
    echo "Rebuilding the Podman image..."
    $DOCKER_CMD build -t "$IMAGE_NAME" -f "$REPO_DIR/$DOCKERFILE" "$REPO_DIR" || { echo "Image build failed!"; exit 1; }

fi

# Base podman run command
CMD="$DOCKER_CMD run --rm"

# If a port was specified, add it to the podman run command
if [ -n "$PORT" ]; then
    CMD="$CMD -p $PORT"  # Add the port mapping
fi

CMD="$CMD -it --shm-size=$SHM -v "$REPO_DIR:/app" -v "$DATA_DIR:/data" -v "$USER_DIR:/user" "$IMAGE_NAME" "

# Run the Podman container
if [ "$INTERACTIVE" = true ]; then
    echo "Running container interactively..."
    $CMD interactive

else
    echo "Running container with arguments: ${ARGS[@]}"
    $CMD "${ARGS[@]}"
fi

