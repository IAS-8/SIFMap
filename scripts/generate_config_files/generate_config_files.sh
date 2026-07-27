#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# generate_config_files.sh
#
# Description:
#   Generates configuration files for dataset processing.
#
#   The script combines:
#     - dataset structure configuration (dataset_config.sh)
#     - default configuration directory (-c)
#     - output configuration directory (-o)
#     - dataset-defined output processing directory (dataset_config.sh)
#     - auto-discovered config files (*config.yaml, *sensor_calib.yaml)
#     - optional Docker runtime environment (-d)
#
# -----------------------------------------------------------------------------
# Usage:
#
#   Local mode:
#       bash generate_config_files.sh \
#           -s <dataset_config.sh> \
#           -c <config_dir> \
#           -o <output_config_dir> \
#           -p <search_pattern>
#
#   Docker mode:
#       bash generate_config_files.sh \
#           -s <dataset_config.sh> \
#           -c <config_dir> \
#           -o <output_config_dir> \
#           -p <search_pattern> \
#           -e <env_config.sh> \
#           -d
#
# -----------------------------------------------------------------------------
# Options:
#
#   -s   Dataset config file (required)
#   -c   Directory containing default configuration files (required)
#   -o   Output directory (required)
#   -p   Search pattern (regex)
#   -e   Environment config file (required only with -d)
#   -d   Enable Docker mode
#
# -----------------------------------------------------------------------------
# Dataset Structure (always used)
#
#     DATA_DIR/
#     └── .../
#         ├── OUTPUT_CONFIG_DIR_REL/
#         │   ├── default_config.yaml
#         │   ├── default_sensor_calib.yaml
#         │   └── <generated configuration files>
#         │
#         ├── CALIB_FILES_DIR_REL/
#         │   └── <campaign>/
#         │       └── <flight>/
#         │           ├── MW_CalibRef_760.mat
#         │           ├── MW_CalibRef_757.mat
#         │           ├── FFmap760.mat
#         │           ├── FFmap757.mat
#         │           ├── Radiometric_Coefficient*.mat
#         │           └── dark*/
#         │               └── ...
#         │
#         ├── RAW_DATA_DIR_REL/
#         │   └── <campaign>/
#         │       └── <flight>/
#         │           └── <raw measurement files>
#         │
#         └── PROCESSED_DATA_DIR_REL/
#             └── <generated processing results>
#
# The dataset layout is defined in dataset_config.sh via relative paths.
#
###############################################################################

usage() {
    cat <<EOF
Usage:
    $(basename "$0") -s <dataset_config.sh> -c <config_dir> -o <output_config_dir> -p pattern [-e <env_config.sh>] [-d]

Examples:
    # Local mode
    $(basename "$0") -s dataset_config.sh -c config/ -o output/config -p pattern

    # Docker mode
    $(basename "$0") -s dataset_config.sh -c config/ -o output/config -p pattern -e env_config.sh -d
EOF
}

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------

DATASET_CONFIG=""
CONFIG_DIR=""
OUTPUT_CONFIG_DIR=""
ENV_CONFIG=""
USE_DOCKER=false

while getopts ":s:c:o:p:e:dh" opt; do
    case $opt in
        s) DATASET_CONFIG="$OPTARG" ;;
        c) CONFIG_DIR="$OPTARG" ;;
        o) OUTPUT_CONFIG_DIR="$OPTARG" ;;
	p) SEARCH_PATTERN=$OPTARG ;;
        e) ENV_CONFIG="$OPTARG" ;;
        d) USE_DOCKER=true ;;
        h)
            usage
            exit 0
            ;;
        \?)
            echo "Error: invalid option -$OPTARG"
            usage
            exit 1
            ;;
        :)
            echo "Error: option -$OPTARG requires an argument"
            usage
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

if [[ -z "$DATASET_CONFIG" || -z "$CONFIG_DIR" || -z "$OUTPUT_CONFIG_DIR" ]]; then
    echo "Error: -s, -c, and -o are required."
    usage
    exit 1
fi

if [[ ! -f "$DATASET_CONFIG" ]]; then
    echo "Error: dataset config not found: $DATASET_CONFIG"
    exit 1
fi

if [[ ! -d "$CONFIG_DIR" ]]; then
    echo "Error: config directory not found: $CONFIG_DIR"
    exit 1
fi

# -----------------------------------------------------------------------------
# Load dataset config (defines relative dataset structure)
# -----------------------------------------------------------------------------

source "$DATASET_CONFIG"

# -----------------------------------------------------------------------------
# Docker mode handling
# -----------------------------------------------------------------------------

if [[ "$USE_DOCKER" == true ]]; then

    if [[ -z "$ENV_CONFIG" ]]; then
        echo "Error: -e is required when using -d (Docker mode)."
        usage
        exit 1
    fi

    if [[ ! -f "$ENV_CONFIG" ]]; then
        echo "Error: env config not found: $ENV_CONFIG"
        exit 1
    fi

    source "$ENV_CONFIG"

    : "${DATA_DIR:?DATA_DIR missing in env config}"
    : "${BASE_DIR:?BASE_DIR must be defined in dataset_config.sh for local mode}"
    : "${USER_DIR:?USER_DIR missing in env config}"
    : "${REPO_DIR:?REPO_DIR missing in env config}"

    PY_DATA_DIR="$DATA_DIR"
    PY_USER_DIR="$USER_DIR"
    PY_REPO_DIR="$REPO_DIR"
    PY_ENV_CONFIG="$ENV_CONFIG"

    BASE="$BASE_DIR"

else
    : "${BASE_DIR:?BASE_DIR must be defined in dataset_config.sh for local mode}"
    : "${OUTPUT_PROCESSING_DIR_REL:?OUTPUT_PROCESSING_DIR_REL must be defined in dataset_config.sh}"

    BASE="$BASE_DIR"

    PY_DATA_DIR="None"
    PY_USER_DIR="None"
    PY_REPO_DIR="None"
    PY_ENV_CONFIG="None"
fi

# -----------------------------------------------------------------------------
# Auto-discover configuration files
# -----------------------------------------------------------------------------

CONFIG_TEMPLATE=$(find "$CONFIG_DIR" -maxdepth 1 -type f -name "*config.yaml" | head -n 1)
SENSOR_CALIB_TEMPLATE=$(find "$CONFIG_DIR" -maxdepth 1 -type f -name "*sensor_calib.yaml" | head -n 1)

if [[ -z "$CONFIG_TEMPLATE" ]]; then
    echo "Error: no *config.yaml found in $CONFIG_DIR"
    exit 1
fi

if [[ -z "$SENSOR_CALIB_TEMPLATE" ]]; then
    echo "Error: no *sensor_calib.yaml found in $CONFIG_DIR"
    exit 1
fi

# -----------------------------------------------------------------------------
# Dataset paths
# -----------------------------------------------------------------------------

CALIB_FILES_DIR="$BASE/$CALIB_FILES_DIR_REL"
RAW_DATA_DIR="$BASE/$RAW_DATA_DIR_REL"
OUTPUT_PROCESSING_DIR="$BASE/$OUTPUT_PROCESSING_DIR_REL/$OUTPUT_CONFIG_DIR"

# -----------------------------------------------------------------------------
# Debug output
# -----------------------------------------------------------------------------

echo "------------------------------------------------------------"
echo "Mode                  : $( [[ "$USE_DOCKER" == true ]] && echo Docker || echo Local )"
echo "Dataset config        : $DATASET_CONFIG"
echo "Config dir            : $CONFIG_DIR"
echo "Output config dir     : $OUTPUT_CONFIG_DIR"
echo "Output processing dir : $OUTPUT_PROCESSING_DIR"
echo "Dataset root          : $BASE"
echo "Pattern		    : $SEARCH_PATTERN" 
echo "------------------------------------------------------------"

# -----------------------------------------------------------------------------
# Run Python (matches process_data.py signature exactly)
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/generate_config_files.py" \
    "$CONFIG_TEMPLATE" \
    "$SENSOR_CALIB_TEMPLATE" \
    "$CALIB_FILES_DIR" \
    "$RAW_DATA_DIR" \
    "$OUTPUT_PROCESSING_DIR" \
    "$OUTPUT_PROCESSING_DIR" \
    "$PY_DATA_DIR" \
    "$PY_USER_DIR" \
    "$PY_REPO_DIR" \
    "$PY_ENV_CONFIG" \
    "$SEARCH_PATTERN"
