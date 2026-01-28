<!--
SPDX-FileCopyrightText: © 2025 Forschungszentrum Jülich GmbH
SPDX-FileContributor: Armagan Elibol
SPDX-FileContributor: Christian Hirt
SPDX-FileContributor: Jim Buffat

SPDX-License-Identifier: GPL-3.0-or-later
-->

# SIFMap overview

SIFMap is a python package to perform SIF retrieval, image registration and mapping with images acquired by SIFcam.
SIFcam is a UAV camera set-up allowing for high-throughput estimation of sun-induced fluorescence (SIF) from precise radiance measurements at 757 and 760 nm (see [Kneer et al.](https://www.doi.org/10.1109/JSEN.2023.3297054)). 

The processing chain for SIFcam implemented in SIFMap is based on four steps: preprocessing, image registration, image alignment and visualization. 
The individual processing steps are implemented independently such that SIFMap may be adapted to process and map images acquired by other camera systems as well.
SIFMap exhibits a significant level of parallelization and efficiency increases due to cached _numba_ compilation. 


## Process a SIFcam data set
### Configuration
To run the SIFMap processing chain, suitable configuration files must first be defined. 
Templates for _run configuration_ and _sensor configuration files_ are provided in the data example included in this code package (in `example_data/mixed_crops_13062021`). 
The _run configuration file_ `config.yaml` defines all options pertaining to the the processing chain. 
The _sensor configuration file_ `sensor_calib.yaml` defines paths to sensor calibration related files.

### Required Files and Information
In addition to the SIFcam image pairs and the configuration files, SIFMap requires additional information to run. Again, for a practical example have a look at `example_data/mixed_crops_13062021/sensor_calib.yaml`. 
This includes 
1. the **flat fielding** parameterzation (see `example_data/mixed_crops_13062021/FFMap757.mat` and `FFMap760.mat`): these must be given as arrays with values ranging in [0, 1] formatted as mat files with the keys _FFmap757_ / _FFmap760_ or else as npy files. A standard set of flat fielding arrays are provided to you with the SIFcam set-up or here in the data example.
2. the **radiance calibration gain** (in mW nm-1 sr-1 m-2 / DN).
3. a set of **dark acquisitions** recorded immediately before or after flight
4. a set of **reflectance panel measurements** in the 757 and 760 nm channel allowing reflectance calibration. These are to be provided as matrices with row-wise measurements (first column is the reference, second the measurement). Practically, such measurements can be done after a first radiance preprocessing step of an image pair showing standardized panels.
5. **integration times** of the 757 and 760 nm channels given in ms.

### Run SIFMap
To first gain the panel measurements to get the reflectance calibration, call
```
python run_UAV_radiance_preprocessing.py config.yaml
```

Read out suitable radiance values over the panels and complete your `sensor_calib.yaml` file. Then, run the full processing chain, calling
```
python run_UAV_processing.py config.yaml
```
to produce individual radiance, reflectance and SIF images as well as a non-geo-referenced mosaick of the data set.

Similarly, run 
```
python run_UAV_preprocessing.py config.yaml
```
to only produce individual radiance, reflectance and SIF images.

In a multi-workstation setup, the hard-coded paths in the configuration files are problematic. In this case, it is recommended to switch to a docker/podman setup. 
To this end, install docker/podman to run SIFMap, copy the `docker/SIFMap_runner.sh` script and configure the paths accordingly on individual machines. 
Then run commands such as (see `docker/entrypoint` and `docker/SIFMap_runner.sh` for explanation of function arguments) 
```
bash SIFMap/docker/SIFMap_runner_test.sh -ug process /user/path_to_config_files/config1.yaml
```
Replace _process_ with _preprocess\_radiance_ or _preprocess_ to achieve the commands described earlier.
With this set-up, use mounted directory names `data`, `app` and `user` in the path definitions of the config file, the sensor calibration file and the bash call. 
Proceeding in this way, allows to exchange config files across workstations.


### Data example
We provide a data example with calibration files in `example_data`. 
A docker file and a detailed step-by-step introduction to the set-up of the SIFMap environment for the unexperienced user is provided in `docker/Dockerdetails.md`.


## Outline of the SIFcam processing chain
### SIFcam acquisition preprocessing
The preprocessing step involves flat fielding, radiance calibration, and an atmospheric correction to gain reflectance in 760 and 757 nm and the SIF retrieval.
Additionally, 757 and 760 nm image pairs are matched in this step to correct for viewing angle differences in the acquisitions of the two SIFcam cameras.
All substeps are defined in the UAVData class in `SIFcam/data`.

### Registration
In this step spatial relationships between images are gained by finding probable feature matches in different images.
The matching is implemented as a parallelized RANSAC based SIFT feature matching without any restrictions on the order on the sorting of the input images.
RANSAC is run with Affine or Projective transforms depending on the configuration parameter `case` (affine for cases 1 & 2, projective for cases 3 & 4).
The registration step is implemented in the module `match`.

### Alignment
The global alignment of all images is implemented to obtain a global homography of SIFcam image datasets.
The alignment is obtained assuming either 2d or 3d geometry leveraging _scipy's_ least-squares optimization implementation. 
The alignment step is implemented in `align`.

### Visualization
The visualization step will produce mosaics of SIF and reflectance with pixels selected according to the following aggregation strategies
* choosing the pixel value from the image with the closest image center 
* choosing the maximum, mean or minimum value


# License
The software contained in this repository is licensed under GPL-3.0 or later.
The data under `example_data` is licensed under CC-BY-NC-ND 4.0.
SIFMap depends on third-party libraries licensed under BSD, MIT, and Apache-2.0.