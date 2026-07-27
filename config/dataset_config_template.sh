###############################################################################
# dataset_config.sh (TEMPLATE)
#
# Purpose:
#   Defines the dataset structure used by generate_config_files.sh.
#
#   This file does NOT contain absolute paths (except BASE_DIR for local mode).
#   Instead, it defines relative paths that are appended to BASE_DIR.
#
# -----------------------------------------------------------------------------
# Example Dataset Structure
#
#     BASE_DIR/
#     └── 01_complete_datasets/
#         ├── 00_batch_generated_configs/
#         │
#         ├── 01_calib_files/
#         │   └── <campaign>/
#         │       ├── MW_CalibRef_760.mat
#         │       ├── MW_CalibRef_757.mat
#         │       ├── FFmap760.mat
#         │       ├── FFmap757.mat
#         │       ├── Radiometric_Coefficient*.mat
#         │       └── dark*/
#         │
#         ├── 02_raw_data/
#         │   └── <campaign>/
#         │       └── <flight>/
#         │           └── <raw measurement files>
#         │
#         └── 03_processed_batch_trial2/
#             └── <generated processing results>
#
# -----------------------------------------------------------------------------
# Notes:
#   - CALIB_FILES_DIR_REL, RAW_DATA_DIR_REL define input dataset locations
#   - OUTPUT_PROCESSING_DIR_REL defines where processing results are stored
#   - This file should remain stable across experiments
#
###############################################################################

# -----------------------------------------------------------------------------
# Base directory (required for non-Docker / local mode)
# -----------------------------------------------------------------------------
BASE_DIR="/path/to/your/dataset/root"

# -----------------------------------------------------------------------------
# Relative dataset paths
# -----------------------------------------------------------------------------

# Calibration files (instrument calibration + reference data)
CALIB_FILES_DIR_REL="01_complete_datasets/01_calib_files"

# Raw measurement data
RAW_DATA_DIR_REL="01_complete_datasets/02_raw_data"

# Output processing results
OUTPUT_PROCESSING_DIR_REL="01_complete_datasets/03_processed/test"

