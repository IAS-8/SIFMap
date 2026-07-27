#!/bin/sh

###############################################################################
# Container Entrypoint Script
#
# This script serves as the main entrypoint for the UAV processing container.
# It dispatches execution to different applications based on the first command-
# line argument, allowing the same container image to support multiple workflows.
#
# Usage:
#   docker run <image> <command> [arguments...]
#
# Supported commands:
#   default               Run the example UAV processing pipeline using the
#                         bundled example configuration.
#
#   test                  Same as default. Run the example UAV processing pipeline using the
#                         bundled example configuration.
#
#   preprocess            Run the UAV preprocessing pipeline with the provided
#                         arguments.
#
#   preprocess_radiance   Run the radiance preprocessing pipeline with the
#                         provided arguments.
#
#   process               Run the UAV processing pipeline with the provided
#                         configuration file and arguments.
#
#   jupyter-notebook      Start a Jupyter Notebook server accessible on all
#                         network interfaces.
#
#   interactive           Start an interactive Bash shell inside the container.
#
# Any other command is interpreted as the configuration file for
# run_UAV_processing.py, allowing the following shorthand:
#
#   podman run <image> config.yaml
#
# which is equivalent to:
#
#   podman run <image> process config.yaml
#
# All additional command-line arguments are forwarded unchanged to the selected
# application.
#
###############################################################################

COMMAND="$1"

shift
ARGS=("$@")


# Check if CMD is set to 'default' and run the default command
if [ "$COMMAND" = "default" ]; then
  exec python /app/run_UAV_processing.py /app/example_data/01_mixed_crops_13062021/config.yaml

elif [ "$COMMAND" = "test" ]; then
  exec python /app/run_UAV_processing.py /app/example_data/01_mixed_crops_13062021/config.yaml

elif [ "$COMMAND" = "preprocess" ]; then
  exec python /app/run_UAV_preprocessing.py "${ARGS[@]}"

elif [ "$COMMAND" = "preprocess_radiance" ]; then
  exec python /app/run_UAV_radiance_preprocessing.py "${ARGS[@]}"

elif [ "$COMMAND" = "interactive" ]; then
  exec /bin/bash

elif [ "$COMMAND" = "jupyter-notebook" ]; then
  exec jupyter notebook --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.notebook_dir='/' "${ARGS[@]}"

elif [ "$COMMAND" = "process" ]; then
  exec python /app/run_UAV_processing.py "${ARGS[@]}"

else
  exec python /app/run_UAV_processing.py "$COMMAND" "${ARGS[@]}"

fi

