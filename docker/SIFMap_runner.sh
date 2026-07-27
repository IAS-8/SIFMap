#!/bin/bash
###############################################################################
# Podman Container Launcher
#
# This script builds (optionally), updates (optionally), and launches the SIFMap
# processing container using Podman. It provides a convenient wrapper around
# `podman run` and forwards all remaining command-line arguments to the
# container entrypoint.
#
# The container entrypoint supports the following commands:
#
#   default
#       Run the example processing pipeline using the bundled example dataset.
#
#   test
#       Same as default. Run the example processing pipeline using the bundled example dataset.
#
#   preprocess
#       Run the UAV preprocessing pipeline.
#
#   preprocess_radiance
#       Run the radiance preprocessing pipeline.
#
#   process
#       Run the UAV processing pipeline with the supplied configuration file.
#
#   jupyter-notebook
#       Start a Jupyter Notebook server inside the container.
#
# Any unrecognized first argument is interpreted by the container as the
# processing configuration file and is passed directly to
# run_UAV_processing.py.
#
# ---------------------------------------------------------------------------
# Usage
#
#   ./run_container.sh -c <config.sh> [options] <entrypoint-command> [args...]
#
# Examples
#
#   # Run a processing configuration
#   SIFMap_runner.sh -c env.sh process /data/project/config.yaml
#
#   # Equivalent shorthand (configuration file interpreted by the entrypoint)
#   SIFMap_runner.sh -c env.sh /data/project/config.yaml
#
#   # Run preprocessing
#   SIFMap_runner.sh -c env.sh preprocess /data/project/config.yaml
#
#   # Start a Jupyter notebook
#   SIFMap_runner.sh -c env.sh -p 8888:8888 jupyter-notebook
#
#
# ---------------------------------------------------------------------------
# Options
#
#   -c <file>
#       Shell configuration file containing environment-specific settings.
#       This option is required.
#
#   -u
#       Rebuild the Podman image before launching the container.
#
#   -g
#       Update the local Git repository by performing a `git pull` before
#       building or running.
#
#   -p <host_port:container_port>
#       Publish a container port (e.g. 8888:8888 for Jupyter).
#
# ---------------------------------------------------------------------------
# Configuration file
#
# The configuration file specified with `-c` is sourced by this script and must
# define at least the following variables:
#
#   REPO_DIR     Path to the repository containing the Dockerfile.
#   IMAGE_NAME   Name of the Podman image.
#   DOCKERFILE   Relative path to the Dockerfile within REPO_DIR.
#   DOCKEREXEC   A docker executable, i.e., 'docker' or 'podman'
#   DATA_DIR     Directory to mount as /data.
#   USER_DIR     Directory to mount as /user.
#   SHM          Shared memory size passed via --shm-size.
#
# See config/env_config_template.sh for an example.
# ---------------------------------------------------------------------------
# Mounted directories
#
#   Host repository  -> /app
#   DATA_DIR         -> /data
#   USER_DIR         -> /user
#
# The repository is mounted into the container, allowing local source changes
# to be used immediately without rebuilding the image (unless dependencies or
# the image itself change).
###############################################################################

UPDATE=false
GITUPDATE=false
ARGS=()

# Function to display usage
usage() {
    echo "Usage: $0 [-u] [-g] [-i] [args]"
    echo "  -u    Rebuild the image before running"
    echo "  -g    Pull the newest git commit."
    echo "  -c    environment configuration"
    echo "  [args] Arguments to pass to the podman run command"
    exit 1
}

# Parse command-line options
while getopts "ugip:c:" opt; do
    case "$opt" in
        u)
            UPDATE=true
        ;;

	      g)
            GITUPDATE=true
        ;;

        p)
            PORT="$OPTARG"  # Capture the port argument
        ;;

        c) CONFIG_FILE="$OPTARG"
        ;;
        
    	*)
        usage
        ;;
    esac
done

source "$CONFIG_FILE" || {
    echo "Cannot load config: $CONFIG_FILE"
    exit 1
}


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
    podman build -t "$IMAGE_NAME" -f "$REPO_DIR/$DOCKERFILE" "$REPO_DIR" || { echo "Image build failed!"; exit 1; }

fi

# Base podman run command
PODMAN_CMD="$DOCKEREXEC run --rm"

# If a port was specified, add it to the podman run command
if [ -n "$PORT" ]; then
   PODMAN_CMD="$PODMAN_CMD -p $PORT"  # Add the port mapping
fi

PODMAN_CMD="$PODMAN_CMD -it --shm-size=$SHM -v "$REPO_DIR:/app" -v "$DATA_DIR:/data" -v "$USER_DIR:/user" "$IMAGE_NAME" "

# Run the Podman container
echo "Running container with arguments: ${ARGS[@]}"
echo "Running $PODMAN_CMD"
$PODMAN_CMD "${ARGS[@]}"

