# SPDX-FileCopyrightText: © 2025 Forschungszentrum Jülich GmbH
# SPDX-FileContributor: Armagan Elibol
# SPDX-FileContributor: Christian Hirt
# SPDX-FileContributor: Jim Buffat <j.buffat@fz-juelich.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import glob
import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import re
import cv2
import yaml

from joblib import Parallel

import numpy as np
import scipy.io as scio
from os.path import join as pjoin

from skimage import io, transform
from skimage.transform import warp
import scipy.ndimage as ndimage

from match.epipolar import epipolar_warp
from match.utils import accurate_image_matcher, run_SIFT, interpolation_flags_CV2, interpolation_flags_skimage
from data.data import _ImageData
from data.utils import run_jobs, init_with_valid_kwargs, replace_ext, write_tif, profile
from data.utils import get_from_dict, extract_patches, reconstruct_from_patches, cast_to_int_type

from functools import partial
from contextlib import nullcontext

import warnings
from scipy.optimize import OptimizeWarning
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", message=".*low contrast image.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=OptimizeWarning, message="Covariance of the parameters could not be estimated")
warnings.filterwarnings("ignore", message="No inliers found.", category=UserWarning, module=r"skimage\.measure\.fit")
warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Dataset has no geotransform", category=NotGeoreferencedWarning)

def clean_dictionary(d):
    """
    Recursively removes keys from a nested dictionary if all values are None.
    Returns None if the entire dictionary is empty after cleaning.
    """
    if not isinstance(d, dict):
        return d  # Return non-dict values as they are

    cleaned_dict = {k: v for k, v in ((k, clean_dictionary(v)) for k, v in d.items()) if v is not None}

    return None if not cleaned_dict else cleaned_dict


class SIFcam(_ImageData):
    """
    This class handles the preprocessing (flat fielding, radiance calibration, 757/760 image pair matching,
    reflectance derivation and SIF retrieval) and creates SIFT features that can be used for global alignment.

    """
    def __init__(self, dataset_path, result_path, sensor_params, flann=None, parallel_params=None,
                 n_SIFT_features_per_image=10000, n_features_threshold=100, camera_regex='',
                 datatake_regex='', ransac_residual_threshold=0.5, feature_product='Rad757',
                 feature_mask=None, interpolation_method='linear', rescale_interpolation_method='linear', 
                 pairing_min_samples=20, pairing_transform_type='affine', radiance_gaussian_blur=None, 
                 pairing_finetune_metric='ssd', pairing_master_band='Rad757', sif_computation_method='fld', 
                 pairing_optical_flow_refinement=None, mask=None, **kwargs):

        super().__init__()
        self.BASE_KEYS += ['OrgRad760', 'OrgRad757',
                           'Rad760', 'Rad757',
                           'Irr760', 'Irr757',
                           'rawRad760', 'rawRad757',
                           'Refl760', 'Refl757',
                           'Fluo', 'H', 'FAILURE']

        self.dataset_path = dataset_path
        self.result_path = result_path
        self.feature_params = dict(n_SIFT_features_per_image=n_SIFT_features_per_image,
                                   n_features_threshodld=n_features_threshold, 
                                   feature_product=feature_product, feature_mask=feature_mask)

        self.flann = flann
        self.flann.update(ransac_residual_threshold=ransac_residual_threshold)

        self.interpolation_method = interpolation_method
        self.radiance_gaussian_blur = radiance_gaussian_blur

        self.rescale_interpolation_method = rescale_interpolation_method
        self.pairing_min_samples = pairing_min_samples
        self.pairing_transform_type = pairing_transform_type
        self.pairing_finetune_metric = pairing_finetune_metric
        self.pairing_master_band = pairing_master_band
        self.pairing_optical_flow_refinement = pairing_optical_flow_refinement

        self.sif_computation_method = sif_computation_method

        if parallel_params is None:
            parallel_params = dict(do_parallel=False)
        self.parallel_params = parallel_params
        
        self.sensor_params = self.sensor_setup(**sensor_params)
        self.sensor_params.update(kwargs)

        self._ids, self._paths, self._dark = self.read_paths(dataset_path, self.sensor_params['DarkPath'],
                                                             camera_regex, datatake_regex)
        self.dark757, self.dark760 = SIFcam.dark_computation(self.sensor_params['DarkPath'], self._dark)
        
        self.sensor_params['dark757'] = self.dark757
        self.sensor_params['dark760'] = self.dark760

        self.removed_paths = []
        self.removed_ids = []

        self._data = self.setup(self._paths, self._ids)
        self.removed_save_data = []

    def __len__(self):
        return len(self.data)

    @property
    def data(self):
        return self._data

    @property
    def paths(self):
        return self._paths

    @property
    def ids(self):
        return self._ids

    @property
    def shapes(self):
        return [p['shape'] for p in self._data]

    def is_initialized(self):
        return self._is_initialized

    @staticmethod
    def _save(arrs, filnames, save_with='tif', dtype=None, cast_to_int=True):
        for arr, fil in zip(arrs, filnames):
            os.makedirs(os.path.dirname(fil), exist_ok=True)
            
            if cast_to_int:
                mask = np.isnan(arr)
                arr[mask] = 0
                arr = cast_to_int_type(arr * 1000)
                arr[mask] = np.iinfo(np.int16).min

            if np.issubdtype(arr.dtype, np.floating) and np.any(np.isnan(arr)):
                arr = arr.copy()
                arr[np.isnan(arr)] = -9999

            if save_with == 'npy':
                if dtype is not None:
                    np.save(fil, arr.astype(dtype), allow_pickle=False)

                else:
                    np.save(fil, arr, allow_pickle=False)

            elif save_with == 'cv2':
                cv2.imwrite(fil, arr)

            elif save_with == 'tif':
                if dtype is not None:
                    arr = arr.astype(dtype)

                io.imsave(fil, arr)

            else:
                raise ValueError(f'"{save_with}" is not a valid save_with argument')

        return filnames

    @staticmethod
    def _load(filname, divide_by=1000, **kwargs):
        if filname.endswith('npy'):
            arr = np.load(filname)

        else:
            arr = io.imread(filname, **kwargs)

        if np.issubdtype(arr.dtype, np.floating):
            arr[arr < -1000] = np.nan

        if divide_by is not None:
            mask = arr < np.iinfo(np.int16).min + 1000
            arr = arr.astype(float) / divide_by
            arr[mask] = np.nan

        return arr

    def match_regex(self, fils, regex):
        """
        Match a regex to a list of file names. The regex must include exactly one group. The function will only succeed
        if exactly one match per file is found.

        Args:
            fils:
            regex:

        Returns:
            list of matches
        """
        ms  = []
        for fil in fils:
            matches = re.findall(regex, fil)

            if len(matches) == 0:
                ms.append('')

            elif len(matches) == 1:
                ms.append(matches[0])

            else:
                raise Exception(f'File {fil} and regex {regex} lead to ambiguous results: {matches}. '
                                f'Must match a single group')

        return np.asarray(ms)

    def read_paths(self, base_path, dark_path, camera_regex, datatake_regex):
        """
        Find the paths to the data takes.

        Args:
            base_path:
            camera_regex:
            datatake_regex:

        Returns:

        """
        fils = os.listdir(base_path)
        datatakes = self.match_regex(fils, datatake_regex)

        uniqs, mapping = np.unique(datatakes, return_inverse=True)
        datatakes = {val: np.where(mapping == key)[0] for key, val in enumerate(uniqs)}
        
        if '' in datatakes:
            non_matched_files = np.asarray(fils)[datatakes['']]
            non_matched_str = f'{non_matched_files[:3]}'
            if len(non_matched_files) > 3:
                non_matched_str += f' ...{non_matched_files[-3:]}'
            print(f'Files {non_matched_str} do not correspond to datatake_regex. Will ignore them.')

        datatakes.pop('', None)

        print(f'{len(datatakes)} data take ids have been identified. Here are the first ones: ', list(datatakes.keys())[:3])

        ignored_keys = []
        for key in datatakes:
            dfils = datatakes[key]

            if len(dfils) > 2:
                raise Exception(f'Datatake {key} from regex {datatake_regex} points to more than two files: {np.array(fils)[dfils]}')

            elif len(dfils) == 1:
                print(f'Datatake {key} from regex {datatake_regex} points to only one file: {dfils}. Will ignore it.')

                ignored_keys.append(key)

        for key in ignored_keys:
            datatakes.pop(key, None)
        
        paths = []
        for key in datatakes:
            key_fils = np.asarray(fils)[datatakes[key]]
            fil760 = key_fils[np.where(self.match_regex(key_fils, camera_regex['760']) != '')[0]]
            fil757 = key_fils[np.where(self.match_regex(key_fils, camera_regex['757']) != '')[0]]
            
            if len(fil760) != 1:
                raise Exception(f'Your camera_regex["760"] matches either none or both of the datatake files: {fil760}'
                                f'It must match exactly one.')

            if len(fil757) != 1:
                raise Exception(f'Your camera_regex["757"] matches either none or both of the datatake files: {fil757}'
                                f'It must match exactly one.')

            paths.append((pjoin(base_path, fil760[0]), 
                          pjoin(base_path, fil757[0])))

        print(f'We\'re running with {len(paths)} paths.')

        all_files = os.listdir(dark_path)
        matches757 = np.where(self.match_regex(all_files, camera_regex['757']) != '')[0]
        matches760 = np.where(self.match_regex(all_files, camera_regex['760']) != '')[0]

        dark_files = {'757': np.asarray(all_files)[matches757], 
                      '760': np.asarray(all_files)[matches760]}
        
        return list(datatakes.keys()), paths, dark_files

    def sensor_setup(self, sensor_calibration_config, **kwargs):
        """
        Read the sensor_calibration_config file
        Args:
            sensor_calibration_config:
            **kwargs:

        Returns:

        """
        with open(sensor_calibration_config, 'r') as file:
            print('\nLOADING SENSOR CALIB', sensor_calibration_config, '\n')
            config = yaml.safe_load(file)

        with open(pjoin(os.path.dirname(__file__), '../config/default_sensor_calib.yaml'), 'r') as file:
            default = yaml.safe_load(file)

        required_sensor_params = ('flat_field', 'radiance_calibration',
                                  'dark_acquisitions', 'KMat',
                                  'reflectance_calibration',
                                  'integration_times', 'mask',
                                  )
        sensor_params = get_from_dict(default, required_sensor_params, need_all=True)
        sensor_params.update(get_from_dict(config, required_sensor_params))

        sensor_params.update(KMat=np.asarray(sensor_params['KMat']))

        if sensor_params['flat_field']['757'].endswith('.mat'):
            tFile = scio.loadmat(sensor_params['flat_field']['757'])
            sensor_params['FFmap757'] = tFile['FFmap757']
        else:
            sensor_params['FFmap757'] = np.load(sensor_params['flat_field']['757'])

        if sensor_params['flat_field']['760'].endswith('.mat'):
            tFile = scio.loadmat(sensor_params['flat_field']['760'])
            sensor_params['FFmap760'] = tFile['FFmap760']
        else:
            sensor_params['FFmap760'] = np.load(sensor_params['flat_field']['760'])

        # old mat input type
        if 'panel_measurements' in sensor_params['reflectance_calibration']['757']:
            S757 = scio.loadmat(sensor_params['reflectance_calibration']['757']['panel_measurements'],
                                variable_names=['MW_ROI', 'STD_ROI'])
            S760 = scio.loadmat(sensor_params['reflectance_calibration']['760']['panel_measurements'],
                                variable_names=['MW_ROI', 'STD_ROI'])

            sensor_params['panel_vals_757'] = np.squeeze(S757['MW_ROI'].T)
            sensor_params['panel_vals_760'] = np.squeeze(S760['MW_ROI'].T)

            sensor_params['ref_vals_757'] = np.array(list(sensor_params['reflectance_calibration']['757']['reference'].values()))
            sensor_params['ref_vals_760'] = np.array(list(sensor_params['reflectance_calibration']['760']['reference'].values()))

            assert sensor_params['panel_vals_757'].shape == sensor_params['ref_vals_757'].shape
            assert sensor_params['panel_vals_760'].shape == sensor_params['ref_vals_760'].shape

        else:
            sensor_params['reflectance_calibration']['757'] = np.asarray(sensor_params['reflectance_calibration']['757'])
            sensor_params['reflectance_calibration']['760'] = np.asarray(sensor_params['reflectance_calibration']['760'])

            sensor_params['ref_vals_757'] = sensor_params['reflectance_calibration']['757'][:, 0]
            sensor_params['ref_vals_760'] = sensor_params['reflectance_calibration']['760'][:, 0]

            sensor_params['panel_vals_757'] = sensor_params['reflectance_calibration']['757'][:, 1]
            sensor_params['panel_vals_760'] = sensor_params['reflectance_calibration']['760'][:, 1]

        # old mat input format
        if type(sensor_params['radiance_calibration']) is str:
            tFile = scio.loadmat(sensor_params['radiance_calibration'], variable_names=['g760', 'g757'])
            g760, g757  = tFile['g760'], tFile['g757']

        else:
            g760, g757 = sensor_params['radiance_calibration']['760'], sensor_params['radiance_calibration']['757']

        sensor_params['g760'] = g760
        sensor_params['g757'] = g757
        sensor_params['DarkPath'] = sensor_params['dark_acquisitions']
        sensor_params['Target'] = 3

        sensor_params['Int760'] = sensor_params['integration_times']['760']
        sensor_params['Int757'] = sensor_params['integration_times']['757']

        sensor_params['mask'] = {"757": clean_dictionary(sensor_params['mask']['757']),
                                 "760": clean_dictionary(sensor_params['mask']['760'])} \
                                 if sensor_params['mask'] is not None else None

        sensor_params['interpolation_method'] = self.interpolation_method
        sensor_params['rescale_interpolation_method'] = self.rescale_interpolation_method
        sensor_params['radiance_gaussian_blur'] = self.radiance_gaussian_blur

        sensor_params['pairing_min_samples'] = self.pairing_min_samples
        sensor_params['pairing_transform_type'] = self.pairing_transform_type
        sensor_params['pairing_finetune_metric'] = self.pairing_finetune_metric
        sensor_params['pairing_optical_flow_refinement'] = self.pairing_optical_flow_refinement

        sensor_params['sif_computation_method'] = self.sif_computation_method

        return sensor_params

    def remove(self, remove):
        """
        Remove an image from the active data pool.

        Args:
            remove:

        Returns:

        """
        remove = [i for i, r in enumerate(remove) if r]

        self.removed_paths = self.removed_paths + [p for i, p in enumerate(self._paths) if i in remove]
        self.removed_save_data = self.removed_save_data + [p for i, p in enumerate(self._data) if i in remove]
        self.removed_ids = self.removed_ids + [p for i, p in enumerate(self._ids)]

        self._paths = [p for i, p in enumerate(self._paths) if not i in remove]
        self._data = [s for i, s in enumerate(self._data) if not i in remove]
        self._ids = [s for i, s in enumerate(self._ids) if not i in remove]

    def compute_radiance(self, *args, **kwargs):
        return self.compute(*args, **kwargs, process_only_radiance=True)

    def compute(self, load_data=True, load_features=True, write=True, keep_data_in_memory=True, write_nonmatched=True, **kwargs):
        """
        This function runs the logic of this class by (i) creating a worker pool (ii) running he preprocessing and (iii)
        creating the features.
        Args:
            load_data: load precomputed results
            write: write the result to disk
            keep_data_in_memory: keep the data in memory between the individual processing steps

        Returns:

        """

        # PARALLEL computation handling
        # Start a Parallel context if parallel computation is requested
        _create_parallel_context = (self.parallel_params['do_parallel']
                                    and not self.parallel_params['individual_worker_pools'])
        if _create_parallel_context:
            context = init_with_valid_kwargs(Parallel, **self.parallel_params)
        else:
            context = nullcontext()

        # COMPUTE
        # Call process and create_features
        with context:
            parallel_context = context if _create_parallel_context else None

            # Run processing on input radiances
            self._data = self.preprocess(write=write, load=load_data, keep_data_in_memory=keep_data_in_memory,
                                         parallel_context=parallel_context)

            # Run feature creation
            self._data, remove = self.process_features(parallel_context=parallel_context, write=write, load=load_features,
                                                       keep_data_in_memory=keep_data_in_memory, **self.feature_params)
            self.remove(remove)

        return self
    
    @profile()
    def preprocess(self, load=True, write=True, write_nonmatched=True, keep_data_in_memory=False, parallel_context=None, 
                   process_only_radiance=False):
        """
        Runs the preprocessing. (i) Radiance is flat fielded and calibrated. (ii) Reflectance and SIF are derived.

        Args:
            load:
            write:
            keep_data_in_memory:
            parallel_context:

        Returns:

        """
        # Run Radiance Processing Pipeline
        self._data, remove = self._process_radiance(load=load, write=write, write_nonmatched=write_nonmatched,
                                                    keep_data_in_memory=keep_data_in_memory,
                                                    parallel_context=parallel_context)
        
        if np.sum(remove) > 0:
            print(f'Removing {np.sum(remove)} images after radiance processing.')
            print(dict([(self._data[i]['id'], self._data[i]['FAILURE']) for i, r in enumerate(remove) if r]))

        self.remove(remove)

        if process_only_radiance:
            return self._data

        # Run Reflectance and Fluorescence Processing Pipeline
        self._data, remove = self._process_reflectance_fluorescence(load=load, write=write, write_nonmatched=write_nonmatched,
                                                                    keep_data_in_memory=keep_data_in_memory,
                                                                    parallel_context=parallel_context)
        self.remove(remove)

        self._is_initialized = True

        return self._data
    
    @profile()
    def process_features(self, parallel_context=None, write=True, load=True, keep_data_in_memory=False,
                         n_SIFT_features_per_image=10000, n_features_threshold=100, feature_product='Rad757',
                         feature_mask=None, **kwargs):
        """
        Parallel call of self.create_features
        """
        assert self.is_initialized()

        jobs = []

        # register image pairs from C757-C760 and create SIF image
        for i, (file760, file757) in enumerate(self.paths):
            pair_data = self._data[i]
            file_name = os.path.basename(file760)

            io_kwargs = dict(load=load, write=write, keep_data_in_memory=keep_data_in_memory,
                             out_dir=self.result_path, file_name=file_name)

            jobs.append(partial(SIFcam.create_features, pair_data, n_features_threshold, **io_kwargs,
                                n_SIFT_features=n_SIFT_features_per_image, feature_prod=feature_product,
                                feature_mask=feature_mask))

        parallel_out = run_jobs(jobs, **self.parallel_params, parallel_context=parallel_context)
        save_data, remove = zip(*parallel_out)

        return save_data, remove

    @staticmethod
    def process_reflectance_fluorescence(pair_data, sensor_params, load, write, write_nonmatched, 
                                         keep_data_in_memory, path760, path757, out_path):
        do_remove = False

        # Prepare saving REFLECTANCE
        base = pjoin(out_path, 'REFLECTANCE')
        path760_ = replace_ext(pjoin(base, os.path.basename(path760)))
        path757_ = replace_ext(pjoin(base, os.path.basename(path757)))
        refl_paths = [path760_, path757_]

        base_nonmatched = pjoin(out_path, 'REFLECTANCE_nonmatched')
        path760_ = replace_ext(pjoin(base_nonmatched, os.path.basename(path760)))
        path757_ = replace_ext(pjoin(base_nonmatched, os.path.basename(path757)))
        refl_paths_nm = [path760_, path757_]

        # Prepare saving IRRADIANCE
        base = pjoin(out_path, 'IRRADIANCE')
        path760_ = replace_ext(pjoin(base, os.path.basename(path760)))
        path757_ = replace_ext(pjoin(base, os.path.basename(path757)))
        irr_paths = [path760_, path757_]

        # Prepare saving REFLMAP
        #base = pjoin(out_path, 'REFLMAP')
        #refl_map_path = replace_ext(pjoin(base,  f'ReflMap_{pair_data["id"]}.tif'))

        # Prepare saving FLUORESCENCE
        base = pjoin(out_path, 'FLUO')
        fluo_path = replace_ext(pjoin(base, f'SIF_{pair_data["id"]}.tif'))

        base = pjoin(out_path, 'RADIANCE')
        path760_ = replace_ext(pjoin(base, os.path.basename(path760)))
        path757_ = replace_ext(pjoin(base, os.path.basename(path757)))
        rad_paths = [path760_, path757_]
        
        base_nonmatched = pjoin(out_path, 'RADIANCE_nonmatched')
        path760_nm = replace_ext(pjoin(base_nonmatched, os.path.basename(path760)))
        path757_nm = replace_ext(pjoin(base_nonmatched, os.path.basename(path757)))
        rad_paths_nm = [path760_nm, path757_nm]


        all_out_paths = irr_paths + rad_paths + [fluo_path]

        if not load or not np.all([os.path.exists(p) for p in all_out_paths]):
            if type(pair_data['Rad760']) is str:
                pair_data['Rad760'] = SIFcam._load(rad_paths[0])
                pair_data['Rad757'] = SIFcam._load(rad_paths[1])
            
            ############### Compute Reflectance
            _, pair_data['Irr760'], pair_data['Refl760'] = SIFcam.process_reflectance(pair_data['Rad760'],
                                                                                      sensor_params['panel_vals_760'],
                                                                                      reflectance_reference=sensor_params['ref_vals_760'])

            _, pair_data['Irr757'], pair_data['Refl757'] = SIFcam.process_reflectance(pair_data['Rad757'],
                                                                                      sensor_params['panel_vals_757'],
                                                                                      reflectance_reference=sensor_params['ref_vals_757'])

            if write_nonmatched:
                _, _, pair_data['Refl760_nonmatched'] = SIFcam.process_reflectance(pair_data['Rad760_nonmatched'],
                                                                                          sensor_params['panel_vals_760'],
                                                                                          reflectance_reference=sensor_params['ref_vals_760'])

                _, _, pair_data['Refl757_nonmatched'] = SIFcam.process_reflectance(pair_data['Rad757_nonmatched'],
                                                                                          sensor_params['panel_vals_757'],
                                                                                          reflectance_reference=sensor_params['ref_vals_757'])
            ##########################################################

            ############# Compute Fluorescence
            pair_data = SIFcam.process_fluorescence(pair_data, mode=sensor_params['sif_computation_method'])
            #########################################################

            # IO
            if write or not keep_data_in_memory:
                # SAVE REFLECTANCE
                SIFcam._save([pair_data['Refl760'], pair_data['Refl757']], refl_paths)
                
                if write_nonmatched:
                    SIFcam._save([pair_data['Refl760_nonmatched'], pair_data['Refl757_nonmatched']], refl_paths_nm)


                # SAVE IRRADIANCE
                SIFcam._save([pair_data['Irr760'], pair_data['Irr757']], irr_paths)

                # SAVE REFLECTANCE
                #SIFcam._save([pair_data['ReflMap']], [refl_map_path])

                # SAVE FLUORESCENCE
                SIFcam._save([pair_data['Fluo']], [fluo_path])

            if not keep_data_in_memory:
                dels = [('Refl760', refl_paths[0]), ('Refl757', refl_paths[1]),
                        ('Refl760_nonmatched', refl_paths_nm[0]), ('Refl757_nonmatched', refl_paths_nm[1]),
                        ('Irr760', irr_paths[0]), ('Irr757', irr_paths[1]),
                        ('Fluo', fluo_path), ('Rad760', rad_paths[0]),
                        ('Rad757', rad_paths[1]), ('Rad760_nonmatched', rad_paths_nm[0]),
                        ('Rad757_nonmatched', rad_paths_nm[1])]
                        # ('ReflMap', refl_map_path),

                for key, val in dels:
                    pair_data[key] = val

        elif load and keep_data_in_memory:

            if type(pair_data['Rad760']) is str:
                pair_data['Rad760'] = SIFcam._load(rad_paths[0]).astype(float)
                pair_data['Rad757'] = SIFcam._load(rad_paths[1]).astype(float)

            pair_data['Refl760'] = SIFcam._load(refl_paths[0]).astype(float)
            pair_data['Refl757'] = SIFcam._load(refl_paths[1]).astype(float)

            pair_data['Irr760'] = SIFcam._load(irr_paths[0]).astype(float)
            pair_data['Irr757'] = SIFcam._load(irr_paths[1]).astype(float)

            #pair_data['ReflMap'] = SIFcam._load(refl_map_path).astype(float)
            pair_data['Fluo'] = SIFcam._load(fluo_path).astype(float)

        else:
            dels = [('Refl760', refl_paths[0]), ('Refl757', refl_paths[1]),
                        ('Refl760_nonmatched', refl_paths_nm[0]), ('Refl757_nonmatched', refl_paths_nm[1]),
                        ('Irr760', irr_paths[0]), ('Irr757', irr_paths[1]),
                        ('Fluo', fluo_path), ('Rad760', rad_paths[0]),
                        ('Rad757', rad_paths[1]), ('Rad760_nonmatched', rad_paths_nm[0]),
                        ('Rad757_nonmatched', rad_paths_nm[1])]

            for key, val in dels:
                pair_data[key] = val
    
        return pair_data, do_remove

    @staticmethod
    def process_fluorescence(pair_data, mode='fld'):

        if mode == 'fld':
            ######## Fluorescence Computation
            subsI = np.subtract(pair_data['Irr757'], pair_data['Irr760']) # I0 - I1
            subsRe = np.subtract(pair_data['Refl760'], pair_data['Refl757']) # Re1 - Re0
            subsRa = np.subtract(pair_data['Rad757'], pair_data['Rad760']) # Ra0 - Ra1
            CRatio = np.divide(np.multiply(pair_data['Irr760'], pair_data['Irr757']), subsI) # I0 I1 / (I0 - I1)

            # (Re1 - Re0) I0 I1 / (I0 - I1)
            # = (Ra1 / I1 * (I0 I1) - Ra0 / I0 * (I0 I1)) / (I0 - I1)
            # = (Ra1 I0 - Ra0 I1) / (I0 - I1) = Fluo
            pair_data['Fluo'] = np.multiply(subsRe, CRatio)

            # (Ra0 - Ra1) / (I0 - I1)
            # = (Re0 I0 - Re1 I1) / (I0 - I1)
            # = ReflMap
            #  pair_data['ReflMap'] = np.abs(np.divide(subsRa, subsI))
            ###################################################

        else:
            raise NotImplementedError('Only FLD is implemented (set sif_computation_method="fld")')

        return pair_data

    @staticmethod
    def process_reflectance(rad, panel_radiance, reflectance_reference):
        ReflVek_R100 = np.zeros(len(reflectance_reference) + 2)
        ReflVek_R100[1:-1] = reflectance_reference
        ReflVek_R100[0] = 1
        ReflVek_R100[-1] = 0

        p = np.polyfit(panel_radiance, reflectance_reference, 1)
        Refl = np.polyval(p, rad)

        Refl[Refl > 1] = 1
        Refl = np.abs(Refl)

        p1 = np.polyfit(reflectance_reference, panel_radiance, 1)
        y1 = np.polyval(p1, ReflVek_R100)
        Irr = np.ones_like(rad) * y1[0]
        Rad = np.multiply(Refl, Irr)

        return Rad, Irr, Refl

    @staticmethod
    def process_align_bands(pair_data, sensor_params, flann, mask=None):
        ############ Get transform between the two bands
        HtMat, status = accurate_image_matcher(pair_data['Rad757'].copy(),
                                               pair_data['Rad760'].copy(),
                                               flann, mask=mask,
                                               transform_type=sensor_params['pairing_transform_type'],
                                               min_samples=sensor_params['pairing_min_samples'],
                                               finetune_metric=sensor_params['pairing_finetune_metric'],
                                               interpolation_method=sensor_params['interpolation_method'])

        if status != 0:
            return None, status

        ############ Apply transform to the two bands
        pair_data['Rad757'], pair_data['Rad760'], status \
            = SIFcam.process_homography_pairing(pair_data['Rad757'],
                                                pair_data['Rad760'],
                                                interpolation_method=sensor_params['interpolation_method'],
                                                rescale_interpolation_method=sensor_params['rescale_interpolation_method'],
                                                H=HtMat, transform_type=sensor_params['pairing_transform_type'])

        pair_data['HtMat'] = HtMat
        

        ########### Optical flow refinement
        if sensor_params['pairing_optical_flow_refinement'] is not None:
            # 757 channel is to be warped
            gray1 = ((pair_data['Rad757'] - np.nanmin(pair_data['Rad757'])) \
                        / (np.nanmax(pair_data['Rad757']) - np.nanmin(pair_data['Rad757'])) * 256)#.astype(np.uint8)
            gray2 = ((pair_data['Rad760'] - np.nanmin(pair_data['Rad760'])) \
                        / (np.nanmax(pair_data['Rad760']) - np.nanmin(pair_data['Rad760'])) * 256)#.astype(np.uint8)
        
            gray1[np.isnan(gray1)] = 0
            gray2[np.isnan(gray2)] = 0

            if sensor_params['pairing_optical_flow_refinement'] == 'farneback':
                flow = SIFcam.optical_flow_farneback(gray1, gray2)

            else:
                raise NotImplementedError()

            pair_data['Rad757'] = SIFcam.align_w_flow(pair_data['Rad757'], flow, 
                                                      interpolation_flags_CV2[sensor_params['interpolation_method']])

        return pair_data, status

    @staticmethod
    def align_w_flow(arr, flow, interpolation_flag):
        h, w = arr.shape
        xx, yy = np.meshgrid(
                np.arange(w),
                np.arange(h)
            )

        map_x = (xx - flow[:, :, 0]).astype(np.float32)
        map_y = (yy - flow[:, :, 1]).astype(np.float32)

        aligned = cv2.remap(
            arr,
            map_x,
            map_y,
            interpolation_flag 
            )

        return aligned

    @staticmethod
    def optical_flow_farneback(image1, image2):
        flow = cv2.calcOpticalFlowFarneback(
            image1,
            image2,
            None,
            pyr_scale=0.5,
            levels=5,
            winsize=25,
            iterations=5,
            poly_n=7,
            poly_sigma=1.5,
            flags=0
        )
        
        return flow 

    @staticmethod
    def process_radiance(pair_data, sensor_params, flann, load, write, write_nonmatched, 
                         keep_data_in_memory, path760, path757, pairing_master_band, out_path):

        do_remove = False

        base = pjoin(out_path, 'RADIANCE')
        path760_ = replace_ext(pjoin(base, os.path.basename(path760)))
        path757_ = replace_ext(pjoin(base, os.path.basename(path757)))
        rad_paths = [path760_, path757_]

        base_nonmatched = pjoin(out_path, 'RADIANCE_nonmatched')
        path760_nm = replace_ext(pjoin(base_nonmatched, os.path.basename(path760)))
        path757_nm = replace_ext(pjoin(base_nonmatched, os.path.basename(path757)))
        rad_paths_nm = [path760_nm, path757_nm]

        if not load or not np.all([os.path.exists(p) for p in rad_paths]):
            ############ Load data paths
            if pair_data['rawRad760'] is None or type(pair_data['rawRad760']) is str:
                pair_data['rawRad760'] = SIFcam._load(path760, divide_by=None).astype(float)

            if pair_data['rawRad757'] is None or type(pair_data['rawRad757']) is str:
                pair_data['rawRad757'] = SIFcam._load(path757, divide_by=None).astype(float)

            ############ Get dark noise statistics 
            minV757 = np.nanmin(pair_data['rawRad757'])
            minV760 = np.nanmin(pair_data['rawRad760'])

            pair_data['rawRad760'] = pair_data['rawRad760'] - sensor_params['dark760']
            pair_data['rawRad757'] = pair_data['rawRad757'] - sensor_params['dark757']

            pair_data['rawRad757'][pair_data['rawRad757'] < 0] = minV757
            pair_data['rawRad760'][pair_data['rawRad760'] < 0] = minV760
            
            ############ Mask raw radiance
            mask = sensor_params['mask']
            if mask is not None and mask['757'] is not None:
                pair_data['rawRad757'] = SIFcam.mask(pair_data['rawRad757'], **mask['757'])

            if mask is not None and mask['760'] is not None:
                pair_data['rawRad760'] = SIFcam.mask(pair_data['rawRad760'], **mask['760'])
 
            ############ Calibrate radiance to physical units
            pair_data['Rad757'] = SIFcam.calibrate_radiance(pair_data['rawRad757'], sensor_params['FFmap757'],
                                                            sensor_params['Int757'], sensor_params['g757'])
            pair_data['Rad760'] = SIFcam.calibrate_radiance(pair_data['rawRad760'], sensor_params['FFmap760'],
                                                            sensor_params['Int760'], sensor_params['g760'])


            pair_data['Rad757'], pair_data['Rad760'] = SIFcam.gaussian_blurring(pair_data['Rad757'], pair_data['Rad760'],
                                                                                gaussian_blur=sensor_params['radiance_gaussian_blur'])

            #Pass along nonmatched radiance
            pair_data['Rad757_nonmatched'], pair_data['Rad760_nonmatched']  = pair_data['Rad757'].copy(), pair_data['Rad760'].copy()
            
            assert pairing_master_band in ('Rad757', 'Rad760'), ('Param master_band must be "Rad757" or "Rad760, '
                                                         f'but found {pairing_master_band}"')

            ############ Switch logic if the master band is Rad757
            if pairing_master_band == 'Rad757':
                _band757 = pair_data['Rad757'].copy()
                pair_data['Rad757'] = pair_data['Rad760'].copy()
                pair_data['Rad760'] = _band757

            ############ Match bands
            pair_data_updated, status = SIFcam.process_align_bands(pair_data, sensor_params, flann)

            ############ Catch exceptions
            if status != 0:
                pair_data['Rad760'], pair_data['Rad757'] = None, None
                pair_data['rawRad760'], pair_data['rawRad757'] = None, None

                pair_data['FAILURE'] = 'BAND_MATCHING'
                do_remove = True

                return pair_data, do_remove

            else:
                pair_data = pair_data_updated

            ############ Switch back order
            if pairing_master_band == 'Rad757':
                _band760 = pair_data['Rad757'].copy()
                pair_data['Rad757'] = pair_data['Rad760'].copy()
                pair_data['Rad760'] = _band760

            ############ Save to disk
            pair_data['shape'] = pair_data['Rad760'].shape

            if write or not keep_data_in_memory:
                SIFcam._save([pair_data['Rad760'], pair_data['Rad757']], rad_paths)
                
                if write_nonmatched:
                    SIFcam._save([pair_data['Rad760_nonmatched'], pair_data['Rad757_nonmatched']], rad_paths_nm)

                # delete loaded radiances if load is False
                if not keep_data_in_memory:
                    pair_data['Rad760'] = rad_paths[0]
                    pair_data['Rad757'] = rad_paths[1]
                    pair_data['Rad760_nm'] = rad_paths[0]
                    pair_data['Rad757_nm'] = rad_paths[1]

                    del pair_data['rawRad760']
                    del pair_data['rawRad757']

        elif load and keep_data_in_memory:
            pair_data['Rad760'] = SIFcam._load(rad_paths[0]).astype(float)
            pair_data['Rad757'] = SIFcam._load(rad_paths[1]).astype(float)
            pair_data['Rad760_nonmatched'] = SIFcam._load(rad_paths_nm[0]).astype(float)
            pair_data['Rad757_nonmatched'] = SIFcam._load(rad_paths_nm[1]).astype(float)

            pair_data['shape'] = pair_data['Rad760'].shape


        else:
            pair_data['Rad760'] = rad_paths[0]
            pair_data['Rad757'] = rad_paths[1]
            pair_data['Rad760_nm'] = rad_paths[0]
            pair_data['Rad757_nm'] = rad_paths[1]

            pair_data['shape'] = SIFcam._load(rad_paths[0]).shape

        return pair_data, do_remove

    @staticmethod
    def gaussian_blurring(Im757, Im760, gaussian_blur=None):
        if gaussian_blur is not None:
            if '760' in gaussian_blur and gaussian_blur['760'] is not None and gaussian_blur['760'] > 0:
                mask = np.isnan(Im760)
                Im760 = ndimage.gaussian_filter(Im760, gaussian_blur['760'])
                Im760[mask] = np.nan

            if '757' in gaussian_blur and gaussian_blur['757'] is not None and gaussian_blur['757'] > 0:
                mask = np.isnan(Im757)
                Im757 = ndimage.gaussian_filter(Im757, gaussian_blur['757'])
                Im757[mask] = np.nan

        return Im757, Im760
    
    @staticmethod
    def warp(H, Im, shape, interpolation_method, translation_before=None, translation_after=None):
            H = np.asarray(H)

            u = [0, shape[1], 0, shape[1]]
            v = [0, 0, shape[0], shape[0]]
            points = np.array([[u[0], v[0]], 
                               [u[1], v[1]], 
                               [u[2], v[2]], 
                               [u[3], v[3]]], 
                              dtype=np.float32).reshape(-1, 1, 2)
            
            if translation_before is not None:
                translation_before = np.asarray([[1, 0, translation_before[0]], [0, 1, translation_before[1]], [0, 0, 1]])
                H = np.matmul(H, translation_before)

            if translation_after is not None:
                translation_after = np.asarray([[1, 0, translation_after[0]], [0, 1, translation_after[1]],  [0, 0, 1]])
                H = np.matmul(translation_after, H)

            transformed_points = cv2.perspectiveTransform(points, H)             
            Im_registered = cv2.warpPerspective(Im, H, (shape[1], shape[0]),
                                                flags=interpolation_flags_CV2[interpolation_method],
                                                borderMode=cv2.BORDER_REPLICATE)

            x = transformed_points[:, 0, 0]
            y = transformed_points[:, 0, 1]
            
            x[x < 0] = 0
            y[y < 0] = 0

            x[x > shape[1] - 1] = shape[1] - 1
            y[y > shape[0] - 1] = shape[0] - 1

            xMin = int(np.ceil(max(x[0], x[2])))
            yMin = int(np.ceil(max(y[0], y[1])))
            xMax = int(np.floor(min(x[1], x[3])))
            yMax = int(np.floor(min(y[2], y[3])))
            
            Imreg = Im_registered[yMin:yMax, xMin:xMax]
            return Imreg, (yMin, yMax, xMin, xMax) 

    @staticmethod
    def process_homography_pairing(Im757, Im760, H, interpolation_method='linear', rescale_interpolation_method='linear',
                                   gaussian_blur=None, transform_type=None):
        """
            This function is to register data from two cameras
        Args:
            Im1 ():
            Im2 ():
            H ():
            interpMethod ():
            Corners ():

        Returns:

        """
        status = 0
        if transform_type in ('projective', 'affine', 'similarity', None):
            Im757reg, bounds = SIFcam.warp(H, Im757, Im760.shape, interpolation_method)
            yMin, yMax, xMin, xMax = bounds

            Im760reg = Im760[yMin:yMax, xMin:xMax]

        elif transform_type in ('piecewise_affine', ):
            try:
                Im757_registered = warp(Im757, H.inverse, order=interpolation_flags_skimage[interpolation_method],
                                        output_shape=(Im760.shape[1], Im760.shape[0]))

                valid_mask = Im757_registered != 0  # Inverts NaN mask to identify valid pixels
                Im757_registered[~valid_mask] = np.nan
 
                Im757reg = Im757_registered
                Im760reg = Im760
                
            except Exception as e:
                status = 1
                return None, None, status

        elif transform_type == 'epipolar':
            Im757reg, Im760reg, status = epipolar_warp(Im757, Im760, camera_matrix, **H)

        else:
            raise NotImplementedError()

        if rescale_interpolation_method is not None:
            Corners = np.array([[0, 0, Im757.shape[1], Im757.shape[1], 0],
                                [0, Im757.shape[0], Im757.shape[0], 0, 0]], 
                               dtype=np.float32)

            Corners2 = np.array([[0, 0, Im757reg.shape[1], Im757reg.shape[1], 0], 
                                 [0, Im757reg.shape[0], Im757reg.shape[0], 0, 0]],
                                dtype=np.float32)

            Hp, _ = cv2.findHomography(Corners2.T, Corners.T)
            ntform = np.array(Hp)  

            Im760reg = cv2.warpPerspective(Im760reg, ntform, (Im760.shape[1], Im760.shape[0]), 
                                           flags=interpolation_flags_CV2[rescale_interpolation_method],
                                           borderMode=cv2.BORDER_REPLICATE)  # borderMode=cv2.BORDER_CONSTANT, borderValue=fillVal)
            Im757reg = cv2.warpPerspective(Im757reg, ntform, (Im757.shape[1], Im757.shape[0]), 
                                           flags=interpolation_flags_CV2[rescale_interpolation_method],
                                           borderMode=cv2.BORDER_REPLICATE)  # borderMode=cv2.BORDER_CONSTANT, borderValue=fillVal)
         
        return Im757reg, Im760reg, status

    @staticmethod
    def create_features(pair_data, thres, load, write, keep_data_in_memory, out_dir, file_name,
                        n_SIFT_features=10000, feature_prod='Rad757', feature_mask=None):
        """
        Creates SIFT features of each image pair.

        Args:
            pair_data: dictionary with all necessary information of the image pair. Holds the reflectance, SIF and paths
            thres: minimum number of SIFT features necessary for an image pair to be
            load: whether to load the features from a precomputed cache in out_dir
            write: whether to write the features to out_memory
            keep_data_in_memory: whether to keep the image data in memory after function completion
            out_dir:
            file_name: file_name of the 760 nm image in the image pair
            n_SIFT_features: number of SIFT features to be created

        Returns:

        """
        do_remove = False

        #feature_file_path =  pjoin(out_dir, 'FEATURE_FILES', f'st{file_name[-31:-4]}.bmp')
        points_path =  pjoin(out_dir, 'FEATURE_FILES', f'{file_name[-31:-4]}_points.npy')
        features_path =  pjoin(out_dir, 'FEATURE_FILES', f'{file_name[-31:-4]}_features.npy')

        if not load:
            pths = dict()
            
            # Load data if it is not in memory
            _unload = False
            loaded_vars = [feature_prod]
            for var in loaded_vars:
                # Keep track of paths if not loaded
                if type(pair_data[var]) is str:
                    _unload = True
                    pths[var] = pair_data[var]
                    pair_data[var] = SIFcam._load(pair_data[var]).astype(float)
                
            # Start feature creation
            if pair_data[loaded_vars[0]] is not None:
                # run SIFT on grayscale image
                fprod = pair_data[feature_prod].copy()
                if feature_mask is not None:
                    fprod = SIFcam.mask(fprod, **feature_mask)

                features, valid_points, status = run_SIFT(fprod, n_SIFT_features)
                if features is None or status > 0:
                    do_remove = True
                    pair_data['FAILURE'] = 'SIFT: NOT ENOUGH FEATURES DETECTED'

                if not do_remove and len(features) > thres:
                    if write or not keep_data_in_memory:
                        SIFcam._save([valid_points, features], [points_path, features_path], save_with='npy')

                    #:pair_data['file_name'] = feature_file_path
                    pair_data['points'] = valid_points
                    pair_data['features'] = features
                    pair_data['valid_points'] = valid_points

                else:
                    do_remove = True
                    pair_data['FAILURE'] = 'SIFT: NOT ENOUGH FEATURES DETECTED'

            else:
                do_remove = True
                pair_data['FAILURE'] = 'NO FLUORESCENCE DETECTED'

            if (_unload and not keep_data_in_memory) or do_remove:
                for var in pths.keys():
                    pair_data[var] = pths[var]

        elif load and not os.path.exists(points_path):
            print(f'Loading features failed for {os.path.basename(file_name)}. Could not find {points_path}')
            do_remove = True

        else:
            pair_data['points'] = SIFcam._load(points_path)
            pair_data['features'] = SIFcam._load(features_path)
            pair_data['valid_points'] = SIFcam._load(points_path)

        return pair_data, do_remove

    @staticmethod
    def calibrate_radiance(raw_radiance, flat_field, integration_time, gain):
        resize_factor = raw_radiance.shape[0] / flat_field.shape[0]
        resized_ffmap = transform.resize(flat_field, (flat_field.shape[0] * resize_factor,
                                                      flat_field.shape[1] * resize_factor))

        rad = np.multiply(raw_radiance, resized_ffmap)
        cal_factor = 1. / integration_time
        coef = gain * cal_factor
        return coef * rad

    @staticmethod
    def dark_computation(dark_path, files):
        """

        Args:
            dark_path ():

        Returns:

        """
        def _load(dark_path, fils):
            maps = []
            for file in fils:
                maps.append(cv2.imread(os.path.join(dark_path, file),
                                       cv2.IMREAD_UNCHANGED).astype(float))

            dark = np.nanmean(maps, axis=0)
            return dark
        
        dark760 = _load(dark_path, files['760'])
        dark757 = _load(dark_path, files['757'])
        
        return dark757, dark760

    @staticmethod
    def mask(arr, point_mask=None, threshold_mask=None, **kwargs):
        if point_mask is not None:
            locs = point_mask['locs']
            buffer = point_mask['buffer']

            for (x, y) in locs:
                arr[x-buffer // 2:x+buffer//2, y-buffer // 2:y+buffer//2] = np.nan
            
        if threshold_mask is not None:
            if 'upper_th' in threshold_mask:
                arr[arr > threshold_mask['upper_th']] = np.nan

            if 'lower_th' in threshold_mask:
                arr[arr < threshold_mask['lower_th']] = np.nan
        
        return arr

    def _process_radiance(self, load=True, write=False, write_nonmatched=True,
                          keep_data_in_memory=False, parallel_context=None):
        """
        Parallel call of preprocess_radiance_and_pair_bands
        Args:
            load:
            write:
            keep_data_in_memory:
            parallel_context:

        Returns:

        """
        jobs = []

        for i, pair_data in enumerate(self._data):
            path760, path757 = self.paths[i]
            io_kwargs = dict(path760=path760, path757=path757, load=load, write=write,
                             keep_data_in_memory=keep_data_in_memory,
                             out_path=self.result_path, pairing_master_band=self.pairing_master_band, 
                             write_nonmatched=write_nonmatched)

            # Prepare parallel execution`of
            # self._preprocess_radiance_and_pair(pair_data, self.sensor_params)
            jobs.append(partial(SIFcam.process_radiance,
                                pair_data, self.sensor_params, self.flann,
                                **io_kwargs))

        save_data, remove = zip(*run_jobs(jobs, **self.parallel_params, parallel_context=parallel_context))
        return save_data, remove

    def _process_reflectance_fluorescence(self, load=True, write=False, write_nonmatched=True, 
                                          keep_data_in_memory=False, parallel_context=None):
        """
        Parallel call of self.process_reflectance_fluorescence
        Args:
            load:
            write:
            keep_data_in_memory:
            parallel_context:

        Returns:

        """
        jobs = []
        for i, pair_data in enumerate(self._data):
            path760, path757 = self.paths[i]
            io_kwargs = dict(path760=path760, path757=path757, load=load, write=write,
                             keep_data_in_memory=keep_data_in_memory, out_path=self.result_path, 
                             write_nonmatched=write_nonmatched)

            # Prepare parallel execution of
            # self._process_reflectance_and_fluorescence(pair_data)
            jobs.append(partial(SIFcam.process_reflectance_fluorescence,
                                pair_data, self.sensor_params, **io_kwargs))

        save_data, remove = zip(*run_jobs(jobs, **self.parallel_params, parallel_context=parallel_context))
        return save_data, remove
