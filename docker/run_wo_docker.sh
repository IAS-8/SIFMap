#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# run_wo_docker.sh
#
# Runs the SIFMap preprocessing or processing pipeline directly on the host
# (without Docker) while remaining compatible with Docker-based configuration
# files.
#
# Cross-platform:
#   - Linux
#   - macOS
#   - Windows (Git Bash / MSYS)
#
# The script loads a host environment configuration, translates Docker mount
# paths (/app, /data, /user) to their corresponding host paths, and creates
# temporary copies of the pipeline configuration and sensor calibration YAML
# files. All path substitutions are performed on these temporary copies, so the
# original configuration files remain unchanged.
#
# On Windows:
#
#   * Bash utilities (cp, sed, mkdir, etc.) operate on POSIX paths
#     (/c/... or /e/...).
#
#   * Any paths written into YAML files or passed to Windows Python are
#     converted to Windows format (C:/... or E:/...), since native Windows
#     executables do not understand Git Bash POSIX paths.
#
# Usage:
#   bash run_wo_docker.sh -c <env_config.sh> {preprocess|process} <config.yaml>
#
# Example:
#   bash run_wo_docker.sh \
#       -c docker/env_config.sh \
#       preprocess \
#       /app/configs/config.yaml
#
# Required environment variables (defined in env_config.sh):
#
#   REPO_DIR
#       Absolute path to the repository on the host.
#
#   DATA_DIR
#       Absolute path corresponding to the Docker /data mount.
#
#   USER_DIR
#       Absolute path corresponding to the Docker /user mount.
#
#   VENV_DIR
#       Absolute path to the SIFMap virtual environment.
#
# Workflow:
#
#   1. Detect operating system.
#   2. Load host environment configuration.
#   3. Normalize paths for Bash.
#   4. Select the correct Python interpreter.
#   5. Translate Docker paths to host paths.
#   6. Locate sensor calibration configuration.
#   7. Copy YAML files to a temporary directory.
#   8. Replace Docker mount paths in temporary YAML files.
#   9. Convert all YAML paths to Windows format when running on Windows.
#  10. Update the temporary config to reference the temporary calibration file.
#  11. Execute the requested pipeline.
#
# Temporary files are automatically deleted when the script exits.
###############################################################################


###############################################################################
# Detect operating system
###############################################################################

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        IS_WINDOWS=1
        ;;
    *)
        IS_WINDOWS=0
        ;;
esac


###############################################################################
# Helper functions
###############################################################################

# Convert Windows paths to Git Bash paths for use by Bash utilities.
to_host_path() {
    local p="$1"

    if (( IS_WINDOWS )); then
        if [[ "$p" =~ ^[A-Za-z]:[\\/].* ]]; then
            cygpath -u "$p"
        else
            printf "%s" "$p"
        fi
    else
        printf "%s" "$p"
    fi
}

# Convert Git Bash paths to Windows paths for native Windows executables.
# Uses forward slashes (C:/...) because they work with Python and avoid
# escaping backslashes inside sed commands.
to_windows_path() {
    local p="$1"

    if (( IS_WINDOWS )); then
        cygpath -m "$p"
    else
        printf "%s" "$p"
    fi
}


###############################################################################
# Parse arguments
###############################################################################

ENV_CONFIG=""

while getopts "c:" opt; do
    case "$opt" in
        c)
            ENV_CONFIG="$OPTARG"
            ;;
        *)
            echo "Usage: $0 -c <env_config.sh> {preprocess|process} <config.yaml>"
            exit 1
            ;;
    esac
done

shift $((OPTIND - 1))


###############################################################################
# Load environment configuration
###############################################################################

[ -n "$ENV_CONFIG" ] || {
    echo "Missing -c <env_config.sh>"
    exit 1
}

source "$ENV_CONFIG"


###############################################################################
# Normalize paths for Bash
###############################################################################

REPO_DIR=$(to_host_path "$REPO_DIR")
DATA_DIR=$(to_host_path "$DATA_DIR")
USER_DIR=$(to_host_path "$USER_DIR")
VENV_DIR=$(to_host_path "$VENV_DIR")


###############################################################################
# Select Python interpreter
###############################################################################

if (( IS_WINDOWS )); then
    PYTHON="$VENV_DIR/Scripts/python.exe"
else
    PYTHON="$VENV_DIR/bin/python"
fi

if [[ ! -x "$PYTHON" ]]; then
    echo "Python interpreter not found:"
    echo "  $PYTHON"
    exit 1
fi


###############################################################################
# Remaining arguments
###############################################################################

COMMAND="$1"
CONFIG="$2"


###############################################################################
# Translate Docker paths to host paths
###############################################################################

CONFIG="${CONFIG/#\/app/$REPO_DIR}"
CONFIG="${CONFIG/#\/data/$DATA_DIR}"
CONFIG="${CONFIG/#\/user/$USER_DIR}"

CONFIG_DIR=$(dirname "$CONFIG")


###############################################################################
# Locate sensor calibration configuration
###############################################################################

SENSOR_CALIB=$(
    sed -nE \
        "s/^sensor_calibration_config:[[:space:]]*['\"]?([^'\"]+)['\"]?.*/\1/p" \
        "$CONFIG"
)

if [[ "$SENSOR_CALIB" != /* ]]; then
    SENSOR_CALIB="$CONFIG_DIR/$SENSOR_CALIB"
fi

SENSOR_CALIB="${SENSOR_CALIB/#\/app/$REPO_DIR}"
SENSOR_CALIB="${SENSOR_CALIB/#\/data/$DATA_DIR}"
SENSOR_CALIB="${SENSOR_CALIB/#\/user/$USER_DIR}"


###############################################################################
# Create temporary files (cross-platform)
###############################################################################

if (( ! IS_WINDOWS )); then

    TMP_DIR=$(mktemp -d)

else

    TMP_BASE="/e/tmp"
    TMP_DIR="$TMP_BASE/sifmap_$$"

    mkdir -p "$TMP_DIR"

fi

trap 'rm -rf "$TMP_DIR"' EXIT

TMP_CONFIG="$TMP_DIR/$(basename "$CONFIG")"
TMP_SENSOR="$TMP_DIR/$(basename "$SENSOR_CALIB")"


cp "$CONFIG" "$TMP_CONFIG"
cp "$SENSOR_CALIB" "$TMP_SENSOR"


###############################################################################
# Determine paths to be written into YAML
###############################################################################

if (( IS_WINDOWS )); then

    REPO_DIR_SUB=$(to_windows_path "$REPO_DIR")
    DATA_DIR_SUB=$(to_windows_path "$DATA_DIR")
    USER_DIR_SUB=$(to_windows_path "$USER_DIR")

    TMP_SENSOR_OUT=$(to_windows_path "$TMP_SENSOR")
    TMP_CONFIG_OUT=$(to_windows_path "$TMP_CONFIG")

else

    REPO_DIR_SUB="$REPO_DIR"
    DATA_DIR_SUB="$DATA_DIR"
    USER_DIR_SUB="$USER_DIR"

    TMP_SENSOR_OUT="$TMP_SENSOR"
    TMP_CONFIG_OUT="$TMP_CONFIG"

fi


###############################################################################
# Replace Docker mount paths in YAML files
###############################################################################

for f in "$TMP_CONFIG" "$TMP_SENSOR"; do

    sed -E -i \
        -e "s#(^|[^[:alnum:]_])(/app)(/|$)#\1$REPO_DIR_SUB\3#g" \
        -e "s#(^|[^[:alnum:]_])(/data)(/|$)#\1$DATA_DIR_SUB\3#g" \
        -e "s#(^|[^[:alnum:]_])(/user)(/|$)#\1$USER_DIR_SUB\3#g" \
        "$f"

done


###############################################################################
# Update temporary configuration
###############################################################################

sed -i \
    "s|^sensor_calibration_config:.*|sensor_calibration_config: $TMP_SENSOR_OUT|" \
    "$TMP_CONFIG"

###############################################################################
# Execute pipeline
###############################################################################

case "$COMMAND" in

    preprocess)

        exec "$REPO_DIR/run_UAV_preprocessing.sh" "$TMP_CONFIG_OUT"

        ;;

    process)

        exec "$PYTHON" \
            "$REPO_DIR/run_UAV_processing.py" \
            "$TMP_CONFIG_OUT"

        ;;

    *)

        echo "Unknown command: $COMMAND"
        exit 1

        ;;

esac
