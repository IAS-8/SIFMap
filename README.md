<!--
SPDX-FileCopyrightText: © 2025 Forschungszentrum Jülich GmbH
SPDX-FileContributor: Armagan Elibol
SPDX-FileContributor: Christian Hirt
SPDX-FileContributor: Jim Buffat

SPDX-License-Identifier: GPL-3.0-or-later
-->

# SIFMap overview

SIFMap is a python package to perform SIF retrieval, image registration and mapping with images acquired by SIFcam.
SIFcam is a UAV camera set-up allowing for high-throughput estimation of sun-induced fluorescence (SIF) from precise radiance measurements at 757 and 760 nm (see [Kneer et al. 2023](https://www.doi.org/10.1109/JSEN.2023.3297054)). 

The processing chain for SIFcam implemented in SIFMap is based on four steps: preprocessing, image registration, mapping and visualization. 
The individual processing steps are implemented independently such that SIFMap may be adapted to process and map images acquired by other camera systems as well.
SIFMap exhibits a significant level of parallelization and efficiency increases due to cached _numba_ compilation. 


## Outline of the SIFMap processing chain
### SIFcam acquisition preprocessing
The preprocessing step involves flat fielding (flat field correction), radiance calibration, and an atmospheric correction that gains reflectance at 760 and 757 nm and the SIF retrieval.
The 757 and 760 nm image paired in this step to correct for viewing angle differences in the acquisitions of the two SIFcam cameras.
The pairing is implemented as a homography matching with optional Farneback optical flow finetuning.
All substeps are defined in the UAVData class in `SIFcam/data.py`.

### Registration
In this step spatial relationships between images are detected by finding probable feature matches in different images.
It is implemented as parallelized projective matching of RANSAC-based SIFT features without any restrictions on the order on the sorting of the input images.
The SIFT features are created from reflectance of the 757 nm channel (can be changed optionally) with CLAHE-improved contrast. 
The registration step is implemented in the module `match`.

### Alignment
With the alignment step, SIFMap recovers the global homography of all SIFcam images. This step boils down to a least-squares
minimization of the symmetric transfer error with a preimplemented and numba-cached jacobian (in `align/jacobians.py`)
The alignment step is implemented in `align`.

### Visualization
The visualization step produces mosaics of SIF and reflectance with pixels in overlapping regions selected according to 
user specified aggregation methods (see `visualization_params.aggregation_methods`). Currently implemented methods are
* choosing the pixel value from the image with the closest image center (`closest`).
* choosing the maximum, mean or minimum value (`min`, `max`, `mean`).
* computing a weighted average with the weight of each overlapping image being 1/d, where d is the distance between the pixel and the image center (`closest_dist_weighted`).

## Process a SIFcam data set
### Configuration
To run the SIFMap processing chain, two configuration files must first be defined: a `config.yaml` and a `sensor_calib.yaml`. 
The _run configuration file_ `config.yaml` defines all options pertaining to the the processing chain. 
The _sensor configuration file_ `sensor_calib.yaml` defines paths to sensor calibration related files.
Templates for a _run configuration_ and a _sensor configuration files_ are provided in the data example included in this code package (in `example_data/01_mixed_crops_13062021`).
All parameters pertaining to these to configuration files are defined and explained in `CONFIGURATION.md`.
Note that user defined configuration files `config.yaml` will overwrite parameters defined in `conifg/default_config.yaml`.

### Required Files and Information
In addition to the SIFcam image pairs and the configuration files, SIFMap requires additional information to run. Again, for a practical example have a look at `example_data/01_mixed_crops_13062021/sensor_calib.yaml`. 
This includes 
1. the **flat fielding** parameterzation (see `example_data/01_mixed_crops_13062021/FFMap757.mat` and `FFMap760.mat`): these must be given as arrays with values ranging in [0, 1] formatted as mat files with the keys _FFmap757_ / _FFmap760_ or else as npy files. A standard set of flat fielding arrays are provided to you with the SIFcam set-up or here in the data example.
2. the **radiance calibration gain** (in mW nm-1 sr-1 m-2 / DN).
3. a set of **dark acquisitions** recorded immediately before or after flight
4. a set of **reflectance panel measurements** in the 757 and 760 nm channel allowing reflectance calibration. These are to be provided as matrices with row-wise measurements (first column is the reference, second the measurement). Practically, such measurements can be done after a first radiance preprocessing step of an image pair showing standardized panels.
5. **integration times** of the 757 and 760 nm channels given in ms.

### Run SIFMap
Create a venv from `requirements.txt` and activate it to run SIFMap.
To first gain the reflectance panel measurements to get reflectance calibration, call
```
python run_UAV_radiance_preprocessing.py config.yaml
```

Read out suitable radiance values over the panels and complete your `sensor_calib.yaml` file by updating `reflectance_calibration`. Then, run the full processing chain, calling
```
python run_UAV_processing.py config.yaml
```
to produce individual radiance, reflectance and SIF images as well as a non-geo-referenced mosaic of the data set.

Similarly, run 
```
python run_UAV_preprocessing.py config.yaml
```
to only produce individual radiance, reflectance and SIF images.

### Run SIFMap in a multi-workstation setup

It is recommended to run SIFMap through the provided docker/podman interface. 
IT ensures repeatability and a standardized configuration file management.
To set up a working environment, follow these steps for each group of datasets:
1. Adopt a specific data set shape, i.e.,

    ```text
    DATA_DIR/
    └── .../
        │
        ├── 01_CALIB_FILES_DIR/
        │   └── <campaign>/
        │       └── <flight>/
        │           ├── MW_CalibRef_760.mat
        │           ├── MW_CalibRef_757.mat
        │           ├── FFmap760.mat
        │           ├── FFmap757.mat
        │           ├── Radiometric_Coefficient*.mat
        │           └── dark*/
        │               └── ...
        │
        ├── 02_RAW_DATA_DIR/
        │   └── <campaign>/
        │       └── <flight>/
        │           └── <raw measurement files>
        │           └── ReadMe*
        │
        └── 03_PROCESSED_DATA_DIR/
            └── <generated processing results>
    ```
    on a server offering access to all users.
    Each user defines an environment configuration by copying and adapting `config/default_dataset_config.yaml`.

2. Each user installs docker/podman, copies the `config/env_config_template.sh` 
   and adapts it to map to the user specific paths and root of the dataset group `DATA_DIR`.
   A docker file and a detailed step-by-step introduction to the set-up of the SIFMap environment for the 
   unexperienced user is provided in `docker/Dockerdetails.md`.
 
3. Define config and sensor calibration files. With this set-up, use mounted directory names `data`, `app` and `user` 
   in the path definitions to refer to `env_config.DATA_DIR`, `env_config.USER_DIR` and `env_config.REPO_DIR`. \
   If needed, run the script `scripts/generate_config_files/generate_config_files.sh` to quickly generate config files
   for all datasets and all users under `dataset_config.BASE_DIR` derived from a base config `base_config.yaml`, e.g., 
   ```
    # For use with podman/docker
    bash scripts/generate_config_files/generate_config_files.sh \
            -s .../path/to/dataset_config.sh \
            -e .../path/to/env_config.sh \
            -c .../path/to/dir/containing/base_config \
            -o output_name \
            -d
    ```
    Note this can be done for a non-docker / non-podman setup as well
    ```
    # No docker/podman
    bash scripts/generate_config_files/generate_config_files.sh \
            -s .../path/to/your/dataset_config.sh \
            -c .../path/to/dir/containing/base_config \
            -o output_name

    ```
   In this case it will not use `env_config.DATA_DIR`, `env_config.USER_DIR` and `env_config.APP_DIR` to 
   generate user-independent config_files.
   The command will create all config files and a batch call under `dataset_config.OUTPUT_PROCESSING_DIR_REL/output_name`.

4. SIFMap can then be run with
    ```
    bash docker/SIFMap_runner.sh -c .../path/to/env_config.sh process /user/path_to_config_files/config.yaml
    ```
    Replace _process_ with _preprocess\_radiance_ or _preprocess_ to achieve the commands described earlier.
    See the documentation in `docker/SIFMap_runner.sh` for more information.

__Note__ that we provide a script `docker/run_wo_docker.sh` that let's you run SIFMap in the same way as in point 4 making 
use of standardized config files with the paths defined on your `data_config` even without a docker/podman installation. 
See the documentation in this script for more information.



### Data example
We provide a small data example with calibration files in `example_data/01_01_mixed_crops_13062021`. 
Adapt the configuration and sensor calibration files and run, as detailed above, 
```
python run_UAV_processing.py example_data/01_mixed_crops_13062021/config.yaml
```
or 
```
bash docker/SIFMap_runner.sh -c .../path/to/env_config.sh example_data/01_mixed_crops_13062021/config.yaml
```
to test your setup.

We provide larger dataset examples [here](https://doi.org/10.26165/JUELICH-DATA/2KDXUL). 
See the accompanying journal publication here (link will be updated) for SIFMap performance 
statistics reached on these datasets.



# License
The software contained in this repository is licensed under GPL-3.0 or later.
The data under `example_data` is licensed under CC-BY-NC-ND 4.0.
SIFMap depends on third-party libraries licensed under BSD, MIT, and Apache-2.0.