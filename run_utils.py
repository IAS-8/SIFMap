from SIFcam.data import SIFcam
from SIFcam.visual import visualize
from align.align import align
from data.data import Datastruct
from data.utils import get_from_dict
from match.register import register_images

import os
import os.path
from os.path import join as pjoin
import sys
import numba
import pickle as pkl
import yaml
import numpy as np
import json


def run_preprocessing(data_params, preproc_params, sensor_params, flann, parallel_params, feature_params, **kwargs):
    # 0. Prepare run ###################################################################################################
    print("#####################################################################")
    print("Preparing Run \n")
    numba.set_num_threads(parallel_params['n_threads'])

    print('\n--------------------- \nData Parameters \n', json.dumps(data_params, indent=4),
          '\n--------------------- \n ')
    print('\n--------------------- \nSensor Parameters \n', json.dumps(sensor_params, indent=4),
          '\n--------------------- \n ')
    print('\n--------------------- \nPreprocessing Parameters \n', json.dumps(preproc_params, indent=4),
          '\n--------------------- \n ')


    # 1. SIF image generation ##########################################################################################
    print("#####################################################################")
    print("SIF image generation starts now by registering C757-C760 pairs \n")

    uav_data = SIFcam(**data_params,
                      **preproc_params,
                      **feature_params,
                      sensor_params=sensor_params,
                      flann=flann,
                      parallel_params=parallel_params).compute(**preproc_params)


def run_radiance_preprocessing(data_params, preproc_params, sensor_params, flann, parallel_params, feature_params, **kwargs):
    # 0. Prepare run ###################################################################################################
    print("#####################################################################")
    print("Preparing Run \n")
    numba.set_num_threads(parallel_params['n_threads'])

    print('\n--------------------- \nData Parameters \n', json.dumps(data_params, indent=4),
          '\n--------------------- \n ')
    print('\n--------------------- \nSensor Parameters \n', json.dumps(sensor_params, indent=4),
          '\n--------------------- \n ')
    print('\n--------------------- \nPreprocessing Parameters \n', json.dumps(preproc_params, indent=4),
          '\n--------------------- \n ')


    # 1. SIF image generation ##########################################################################################
    print("#####################################################################")
    print("SIF image generation starts now by registering C757-C760 pairs \n")

    uav_data = SIFcam(**data_params,
                      **preproc_params,
                      **feature_params,
                      sensor_params=sensor_params,
                      flann=flann,
                      parallel_params=parallel_params).compute_radiance(**preproc_params)



def run(data_params, preproc_params, register_params, align_params, sensor_params,
        visualization_params, flann, parallel_params, feature_params, **kwargs):

    # 0. Prepare run ###################################################################################################
    print("#####################################################################")
    print("Preparing Run \n")
    numba.set_num_threads(parallel_params['n_threads'])
    flann.update(alignment_dim=2 if align_params['case'] in (1, 2) else 3)

    print('\n--------------------- \nData Parameters \n', json.dumps(data_params, indent=4), '\n--------------------- \n ')
    print('\n--------------------- \nSensor Parameters \n', json.dumps(sensor_params, indent=4), '\n--------------------- \n ')
    print('\n--------------------- \nPreprocessing Parameters \n', json.dumps(preproc_params, indent=4), '\n--------------------- \n ')
    print('\n--------------------- \nRegistration Parameters \n', json.dumps(register_params, indent=4), '\n--------------------- \n ')
    print('\n--------------------- \nAlignment Parameters \n', json.dumps(align_params, indent=4), '\n--------------------- \n ')
    print('\n--------------------- \nVisualization Parameters \n', json.dumps(visualization_params, indent=4), '\n--------------------- \n ')

    os.makedirs(pjoin(data_params['result_path'],'mapping_data'), exist_ok=True)
    print("#####################################################################\n")
    ####################################################################################################################




    # 1. SIF image generation ##########################################################################################
    print("#####################################################################")
    print("SIF image generation starts now by registering C757-C760 pairs \n")

    uav_data = SIFcam(**data_params,
                      **preproc_params,
                      **feature_params,
                      sensor_params=sensor_params,
                      flann=flann,
                      parallel_params=parallel_params).compute(**preproc_params)

    print("#####################################################################\n")
    ####################################################################################################################


    hglobal_path = pjoin(data_params['result_path'], 'mapping_data', 'HGlobal.npy')
    x_path = pjoin(data_params['result_path'], 'mapping_data', 'x_optim_params.npy')
    match_matrix_path2 = pjoin(data_params['result_path'], 'mapping_data', 'match_matrix_final.pkl')


    # 2. Registering images ###########################################################################################
    print("#####################################################################")
    print("Image registration starts now \n")

    assert not (align_params['load'] and not register_params['load']), \
                'Alignment must be run and cannot be loaded if registration is changed'

    if not align_params['load']:
        match_matrix_path = pjoin(data_params['result_path'], 'mapping_data', 'match_matrix_pre_align.pkl')
        from_file = match_matrix_path if register_params['load'] else None

        data = Datastruct(uav_data, from_file=from_file)
        if from_file is None:
            register_images(data=data,
                            flann=flann,
                            parallel_params=parallel_params,
                            **feature_params,
                            **register_params)

        if register_params['write']:
            data.to_file(match_matrix_path)

        print("#####################################################################\n")
        ####################################################################################################################




    # 3. Aligning images ###############################################################################################
        print("#####################################################################")
        print("Estimation of HGlobal starts now \n")

        if align_params['resume_optimization']:
            assert os.path.exists(x_path), 'There is no existing checkpoint from which optimization can be restarted'
            init_x = np.load(x_path)
            data = Datastruct(uav_data, from_file=match_matrix_path2)
            init_H = np.load(hglobal_path)[data.load_indices]

        else:
            init_x = None
            init_H = None

        params = uav_data.sensor_params
        params.update(feature_params)
        params.update(align_params)
        params.update(visualization_params)

        data, HGlobal, x = align(data=data,
                                 **params,
                                 min_n_correspondences=register_params['min_n_correspondences'],
                                 min_movement=register_params['min_movement'],
                                 init_x=init_x,
                                 init_H=init_H)

        if align_params['write']:
            np.save(hglobal_path, HGlobal)
            np.save(x_path, x)
            data.to_file(match_matrix_path2)

    else:
        data = Datastruct(uav_data, from_file=match_matrix_path2)
        HGlobal = np.load(hglobal_path)[data.load_indices]

    print("#####################################################################\n")
    ####################################################################################################################




    # 4. Visualization #################################################################################################
    print("#####################################################################")
    print("Visualization starts now \n")
    visualize(data=data, HGlobal=HGlobal, **visualization_params, parallel_params=parallel_params)
    print("#####################################################################\n")
    ####################################################################################################################


def read_params(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # FLANN parameters 
    if 'flann_params' in config:
        index_params = get_from_dict(config['flann_params'], ("algorithm", "trees"))
        search_params = get_from_dict(config['flann_params'], ("checks",))

        flann = dict(type=config['flann_params']['type'], args=(index_params, search_params))

    else:
        flann = dict()

    # PARALLELIZATION params
    if 'parallel_params' in config:
        required_parallel_params = ('do_parallel', 'n_processes', 'n_threads')
        parallel_params = get_from_dict(config['parallel_params'], required_parallel_params)

        parallel_params.update(n_jobs=parallel_params['n_processes'] if parallel_params['n_processes'] >= 0
                                            else os.cpu_count(),
                               n_threads=parallel_params['n_threads'] if parallel_params['n_threads'] >= 0
                                            else numba.config.NUMBA_NUM_THREADS,
                               prefer='processes',
                               individual_worker_pools=False)
    else:
        parallel_params = dict()

    # PREPROC params
    if 'preprocessing_params' in config:
        required_preproc_params = ('load_data', 'load_features', 'write', 'keep_data_in_memory',
                                   'ransac_residual_threshold', 'rescale_interpolation_method',
                                   'interpolation_method', 'pairing_min_samples',
                                   'pairing_transform_type', 'pairing_finetune_metric',
                                   'pairing_master_band', 'radiance_gaussian_blur')
        preproc_params = get_from_dict(config['preprocessing_params'], required_preproc_params)

    else:
        preproc_params = dict()

    # REGISTRATION params
    if 'registration_params' in config:
        required_register_params = ('load', 'write', 'min_n_correspondences', 'ransac_residual_threshold',
                                    'min_movement')
        register_params = get_from_dict(config['registration_params'], required_register_params)
        register_params.update(min_movement=register_params['min_movement']
                                                if register_params['min_movement'] > 0 else None)

    else:
        register_params = dict()

    # ALIGN params
    if 'alignment_params' in config:
        required_align_params = ('load', 'write',
                                 'case', 'optim_params',
                                 'n_outlier_removal',
                                 'manual_outlier_mode',
                                 'n_points', 'outlier_threshold',
                                 'resume_optimization')
        align_params = get_from_dict(config['alignment_params'], required_align_params)

    else:
        align_params = dict()

    # VISUALIZATION params
    if 'vis_params' in config:
        required_vis_params = ('mosaic_resolution', 'mosaic_origin', 'interpolation_method')
        vis_params = get_from_dict(config['vis_params'], required_vis_params)

    else:
        vis_params = dict()

    # SENSOR params
    if 'sensor_calibration_config' in config:
        sensor_params = dict(sensor_calibration_config=config['sensor_calibration_config'])

    else:
        sensor_params = dict()


    # DATA params
    required_data_params = ('dataset_path', 'result_path',
                            'camera_regex', 'datatake_regex')
    data_params = get_from_dict(config['data_params'], required_data_params)


    # FEATURE params
    if 'feature_params' in config:
        required_feature_params = ('n_SIFT_features_per_image', 'n_features_threshold',
                                   'n_features_trial', 'feature_product', 'mask')
        feature_params = get_from_dict(config['feature_params'], required_feature_params)
        feature_params['feature_mask'] = feature_params['mask']
        del feature_params['mask']

    else:
        feature_params = dict()


    out_dict = dict(sensor_params=sensor_params,
                    data_params=data_params,
                    preproc_params=preproc_params,
                    register_params=register_params,
                    align_params=align_params,
                    visualization_params=vis_params,
                    flann=flann,
                    parallel_params=parallel_params,
                    feature_params=feature_params
                    )

    return out_dict


def read_config(config):
    default_params = read_params(pjoin(os.path.dirname(__file__), 'config/default_config.yaml'))
    params = read_params(config)
    for kdict in params.keys():
        default_params[kdict].update(params[kdict])
    
    return default_params
