#!/bin/sh

# This script handles different entrypoint commands based on CMD or ENTRYPOINT_CMD

# Default command is provided by CMD
COMMAND="$1"

shift
ARGS=("$@")


# Check if CMD is set to 'default' and run the default command
if [ "$COMMAND" = "default" ]; then
  exec python /app/run_UAV_processing.py /app/example_data/mixed_crops_13062021/config.yaml
 

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

