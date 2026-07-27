# Configuration Reference

This document describes all configuration parameters used by the SIFMap processing pipeline.

## Notation

| Notation | Meaning |
|----------|---------|
| `bool` | Boolean value (`true` or `false`). |
| `path` | Path to an existing file or directory. |
| `null` | Parameter disabled or no value supplied. |
| `dict[key,value]<keys>` | YAML dictionary (mapping) over <keys>. |
| `arr` | 2d YAML array. |
| `<band>` | Spectral band identifier (currently `757` or `760`). |

---

# Run Configuration (`calib.yaml`)

## `data_params`

| Parameter | Type | Valid Values | Description                                                                                                                                                                                                                                                                              |
|-----------|------|--------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `dataset_path` | `path` | Existing directory | Directory containing the input dataset.                                                                                                                                                                                                                                                  |
| `result_path` | `path` | Existing directory | Directory where all outputs will be written.                                                                                                                                                                                                                                             |
| `camera_regex` | `dict[str,str]<band>` | Regular expressions | Maps sensor identifiers to regular expressions used to identify cameras from filenames. Must contain exactly one group targeting the `band_identifier`. Example: `.*?(C760).*?\.tif$`                                                                                                    |
| `datatake_regex` | `str` | Regular expression | Regular expression used to extract the datatake identifier from image filenames. Must contain exactly one group targeting the image `pair_identifier`.  Note: `pair_identifier` must be convertible to integer if `registration_params.protect_correspondences_in_close_images` is used. |

---

## `sensor_calibration_config`

| Parameter | Type | Valid Values | Description |
|-----------|------|--------------|-------------|
| `sensor_calibration_config` | `path` | YAML file | Path to the sensor calibration configuration file (`sensor_calib.yaml`). If not an absolute path, assumes same dir as config.yaml. |

---

## `preprocessing_params`

| Parameter | Type | Valid Values                 | Description                                                                                                                                                                                                                                     |
|-----------|------|------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `load_data` | `bool` | `true`, `false`              | Load previously preprocessed data instead of recomputing.                                                                                                                                                                                       |
| `load_features` | `bool` | `true`, `false`              | Load previously saved feature descriptors.                                                                                                                                                                                                      |
| `write` | `bool` | `true`, `false`              | Save preprocessing outputs to disk.                                                                                                                                                                                                             |
| `keep_data_in_memory` | `bool` | `true`, `false`              | Keep intermediate data in memory to reduce disk I/O.                                                                                                                                                                                            |
| `ransac_residual_threshold` | `float` | \> 0                         | Maximum reprojection error (pixels) used by RANSAC during image pairing.                                                                                                                                                                        |
| `pairing_min_samples` | `int` | \> 8                         | Minimum number of feature correspondences required to estimate a transform between image pairs.                                                                                                                                                 |
| `pairing_transform_type` | `str` | `affine`, `projective`       | Geometric transform model used for image registration.                                                                                                                                                                                          |
| `interpolation_method` | `str` | `nearest`, `linear`, `cubic` | Interpolation method used during image warping in channel matching                                                                                                                                                                              |
| `pairing_master_band` | `str` | `Rad757`, `Rad760` | Spectral band used as the reference during channel matching.                                                                                                                                                                                    |
| `pairing_optical_flow_refinement` | `str` | `null`, `farneback` | Optional optical flow refinement after feature matching. Only `farneback` is implemented.                                                                                                                                                       |
| `radiance_gaussian_blur` | `dict[str,float]<band>` | `null` or positive number | Gaussian blur (σ) applied independently to each spectral band before feature extraction. Applying gaussian blur to the 757 channel improves SIF retrieval as it approximately equalizes motion blur and higher noise levels in the 760 channel. |

---

## `registration_params`

| Parameter                                 | Type | Valid Values  | Description                                                                                                                                                                                       |
|-------------------------------------------|------|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `load`                                    | `bool` | `true`, `false` | Load previously computed registrations.                                                                                                                                                           |
| `write`                                   | `bool` | `true`, `false` | Save registration results.                                                                                                                                                                        |
| `min_movement`                            | `float` |  $\geq$ 0, null | Minimum estimated movement required before registering an image pair. With -1 the option is disabled.                                                                                             |
| `min_n_correspondences`                   | `int` | \> 0          | Minimum number of correspondences required for registration.                                                                                                                                      |
| `ransac_residual_threshold`               | `float` | \> 0          | RANSAC reprojection error threshold (pixels).                                                                                                                                                     |
| `n_features_trial`                        | `int` | \> 0          | Number of candidate features sampled during coarse matching.                                                                                                                                      |
| `protect_correspondences_in_close_images` | `int` | $\geq$ 0      | Minimum number of correspondences between close images, close being defined as having a difference in pair identifier smaller than `close_images_limit`. Note: `pair_identifier` must be integer. |
| `close_images_limit`                      | `int` | \> 0          | Maximum difference in `pair_identifier` for two images to be considered _close_                                                                                                                   |

---

## `alignment_params`

| Parameter | Type | Valid Values    | Description                                                                                                                                                                                                                                                         |
|-----------|------|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `load` | `bool` | `true`, `false` | Load an existing alignment solution.                                                                                                                                                                                                                                |
| `write` | `bool` | `true`, `false` | Save alignment results.                                                                                                                                                                                                                                             |
| `n_outlier_removal` | `int` | \> 0            | Number of outlier removal iterations. Outlier removal is disabled with 0.                                                                                                                                                                                                                              |
| `manual_outlier_mode` | `bool` | `true`, `false` | Enable manual residual-based outlier removal.                                                                                                                                                                                                                       |
| `outlier_threshold` | `float` | \> 3            | Residual threshold used to classify outliers.                                                                                                                                                                                                                       |
| `n_points` | `int` | \> 0            | Maximum number of correspondences used per image pair. Note: `registration_params.min_n_correspondences` is the lower limit.  Note:  You should make sure that `registration_params.min_n_correspondences` <  `n_points` < `registration_params.n_features_trial`. |
| `resume_optimization` | `bool` | `true`, `false` | Resume optimization from an existing mosaic solution. Note: load must be false to activate this option.                                                                                                                                                             |
| `remove_disconnected_threshold` | `int` | \> 0 | Removes disconnected graph partitions if they cluster less than `remove_disconnected_threshold` images.|                                                                

### `alignment_params.optim_params`
SIFMap runs scipy.optimize.least_squares with `trf` and `lsmr`. Provide additional options or choose different methods and solvers.
Any parameter (or dictionary of parameters) that can be passed to scipy.optimize.least_squares is valid, including different least_squares methods.
Here we list the most relevant.

| Parameter                     | Type    | Valid Values               | Description                                                                                                                                                                                                                   |
|-------------------------------|---------|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `xtol`                        | `float` | \> 0, `null`               | Parameter convergence tolerance. See scipy.optimize.least\_squares documentation.                                                                                                                                             |
| `ftol`                        | `float` | \> 0, `null`               | Cost-function convergence tolerance. See scipy.optimize.least\_squares documentation.                                                                                                                                         |
| `gtol`                        | `float` | \> 0, `null`               | Optimality convergence tolerance. See scipy.optimize.least\_squares documentation. For typical SIFcam imagery, an optimality of ca. 5e3 is usually sufficient.                                                                |
| `max_nfev`                    | `int`   | \> 0, `null`               | Maximum number of optimizer function evaluations. See scipy.optimize.least\_squares documentation.                                                                                                                            |
| `tr_options`                  | `dict`  | Optimizer-supported values | Solver options. See scipy.optimize.least\_squares documentation.                                                                                                                                                              |
| `lsmr_maxiter`                | `str`   | \> 50, `null`              | SIFMap-specific option. Defines the maximum steps of the `lsmr` solver. See scipy.optimize.least\_squares documentation. Note: for typical SIFcam imagery and  `tr_solver="lsmr"`, `tr_options.maxiter` can be set to \< 300. |
| `lsmr_dynamic_maxiter_factor` | `str`   | \> 1, `null`               | SIFMap-specific option. Defines a growth factor of maxiter, applied before every outlier detection iteration (see `alignment_params.n_outlier_removal`). `null` deactivates dynamic growth.                                   |
| `lsmr_max_maxiter`            | `str`   | \> 50, `null`              | SIFMap-specific option. Defines an upper limit to the dynamical growth of `lsmr`'s `maxiter`.                                                                                                                                 |

---

## `feature_params`

| Parameter | Type | Valid Values                                     | Description |
|-----------|------|--------------------------------------------------|-------------|
| `n_SIFT_features_per_image` | `int` | \> 0                                             | Maximum number of SIFT features extracted per image. |
| `n_features_threshold` | `int` | \> 0                                             | Minimum acceptable number of detected SIFT features. |
| `feature_product` | `str` | `Rad757`, `Rad760`, `Refl757`, `Refl760`, `Fluo` | Image product used for feature extraction. |

---

## `parallel_params`

| Parameter | Type | Valid Values    | Description |
|-----------|------|-----------------|-------------|
| `do_parallel` | `bool` | `true`, `false` | Enable process-based parallelization. |
| `n_processes` | `int` | `null` or \> 0    | Number of worker processes (`null` uses all logical CPU cores). |
| `n_threads` | `int` | `null`  or \> 0    | Number of Numba threads (`null` uses all available threads). |

---

## `flann_params`

| Parameter | Type | Valid Values | Description |
|-----------|------|--------------|-------------|
| `trees` | `int` | \> 0         | Number of KD-trees used for indexing. |
| `checks` | `int` | \> 0         | Number of search checks. Larger values improve matching accuracy at the expense of runtime. |

---

## `vis_params`

| Parameter              | Type   | Valid Values                                                       | Description                                                                                                                                                                                                                                                                                           |
|------------------------|--------|--------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `interpolation_method` | `str`  | `nearest`, `linear`, `cubic`                                       | Interpolation method used when generating the output mosaic.                                                                                                                                                                                                                                          |
| `n_processes`          | `int`  | `null` or \> 0                                                     | Number of worker processes for the visualization(`null` uses all logical CPU cores). The memory footprint per worker is much larger than for other processing steps, such that a smaller number than the global`parallel_params.n_processes` must probably be chosen, depending on the available RAM. |
| `run`                  | `bool` | `true` or `false`                                                  | Whether to run visualization.                                                                                                                                                                                                                                                                         |
| `products`             | `list` | `SIF`, `Ref757`, `Ref760`                                          | Products to visualize.                                                                                                                                                                                                                                                                                |
| `aggregation_methods`  | `list` | `min`, `max`, `mean`, `median`, `closest`, `closest_dist_averaged` | Aggregation methods to visualize.                                                                                                                                                                                                                                                                     |

---

# Sensor Calibration Configuration (`sensor_calib.yaml`)

This file contains the sensor-specific calibration parameters used during radiometric correction, reflectance calibration, masking, and geometric processing.

## DNs to Radiance

| Parameter | Type | Valid Values | Description |
|-----------|------|--------------|-------------|
| `flat_field` | `dict[str,path]<band>` | One `.mat` or `.npy` file per sensor | Flat-field correction maps used to compensate for pixel-to-pixel sensitivity variations. |
| `radiance_calibration` | `path` | `.mat` or `.npy` file or `dict[str,float]<band>` | Radiometric calibration coefficients used to convert raw digital numbers to radiance.|
| `dark_acquisitions` | `path` | Existing directory | Directory containing dark-frame acquisitions for dark-current correction. |
| `integration_times.<band>` | `float` or `int` | \> 0         | Camera integration (exposure) time for each spectral band. Time unit must match the unit used for the gain in radiance\_calibration. |

---

## Camera Intrinsics

| Parameter | Type | Valid Values | Description |
|-----------|------|--------------|-------------|
| `KMat` | `arr` | Valid camera intrinsic matrix | Camera intrinsic matrix in the form `[[fx,0,cx],[0,fy,cy],[0,0,1]]`, where `fx` and `fy` are focal lengths (pixels) and `cx`, `cy` are the principal point coordinates. Note: SIFcam's focal length is 25 mm.|

---

## Reflectance Calibration / Atmospheric Correction

There are two supported iputs for the reflectance calibration:

| Parameter | Type | Valid Values | Description |
|-----------|------|--------------|-------------|
| `reflectance_calibration.<band>.reference` | `dict[str,float]` | Values between 0 and 1 | Certified reflectance values of the calibration panel. Note: the key names are irrelevant, but the entries must be ordered from largest to smallest.|
| `reflectance_calibration.<band>.panel_measurements` | `path` | `.mat` file | Measured calibration panel acquisition for the corresponding spectral band. Note: the entries must be ordered from largest to smallest.|


| Parameter | Type | Valid Values | Description |
|-----------|------|--------------|-------------|
| `reflectance_calibration.<band>` | `dict[str,arr]` | Values between 0 and 1 | Array for each band. First column: certified reflectance values of the calibration panel. Second column: measured calibration panel acquisition for the corresponding spectral band. Note: the rows must be ordered from largest to smallest.|


## Masks over the input

| Parameter | Type | Valid Values              | Description |
|-----------|------|---------------------------|-------------|
| `mask.<band>` | `dict` or `null` | `null` or mask definition | Optional exclusion mask applied to the specified spectral band. Note: the only currently supported mask definition is `point_mask`. |
| `point_mask.locs` | List of pixel coordinates | Image coordinates         | Centers of circular exclusion regions. |
| `point_mask.buffer` | `int` | \> 0                      | Radius (pixels) of each exclusion region. |

---
