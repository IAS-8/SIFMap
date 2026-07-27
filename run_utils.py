import os
import os.path
from os.path import join as pjoin
import yaml
import numpy as np
import json

def prepare_numba_import(n_threads):
    n_threads = str(n_threads)
    for var in ("NUMBA_NUM_THREADS",
                "MKL_NUM_THREADS",
                #"OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS"):
        os.environ[var] = n_threads

    os.environ["NUMBA_THREADING_LAYER"] = "omp"


def run_preprocessing(data_params, preproc_params, sensor_params, flann, parallel_params, feature_params, **kwargs):
    # 0. Prepare run ###################################################################################################
    print("#####################################################################")
    print("Preparing Run \n")

    prepare_numba_import(parallel_params['n_threads'])
    import numba
    numba.set_num_threads(parallel_params['n_threads'])
    print('Running with', numba.get_num_threads(), f'threads and threading layer `{numba.config.THREADING_LAYER}` ')

    print('\n--------------------- \nData Parameters \n', json.dumps(data_params, indent=4),
          '\n--------------------- \n ')
    print('\n--------------------- \nSensor Parameters \n', json.dumps(sensor_params, indent=4),
          '\n--------------------- \n ')
    print('\n--------------------- \nPreprocessing Parameters \n', json.dumps(preproc_params, indent=4),
          '\n--------------------- \n ')


    # 1. SIF image generation ##########################################################################################
    print("#####################################################################")
    print("SIF image generation starts now by registering C757-C760 pairs \n")

    from SIFcam.data import SIFcam

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

    prepare_numba_import(parallel_params['n_threads'])
    import numba
    numba.set_num_threads(parallel_params['n_threads'])
    print('Running with', numba.get_num_threads(), f'threads and threading layer `{numba.config.THREADING_LAYER}` ')

    print('\n--------------------- \nData Parameters \n', json.dumps(data_params, indent=4),
          '\n--------------------- \n ')
    print('\n--------------------- \nSensor Parameters \n', json.dumps(sensor_params, indent=4),
          '\n--------------------- \n ')
    print('\n--------------------- \nPreprocessing Parameters \n', json.dumps(preproc_params, indent=4),
          '\n--------------------- \n ')


    # 1. SIF image generation ##########################################################################################
    print("#####################################################################")
    print("SIF image generation starts now by registering C757-C760 pairs \n")

    from SIFcam.data import SIFcam

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

    prepare_numba_import(parallel_params['n_threads'])
    import numba
    numba.set_num_threads(parallel_params['n_threads'])
    print('Running with', numba.get_num_threads(), f'threads and threading layer `{numba.config.THREADING_LAYER}` ')

    from SIFcam.data import SIFcam
    from SIFcam.visual import visualize
    from align.align import align
    from data.data import Datastruct
    from match.register import register_images

    flann.update(alignment_dim=3)

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

        data = Datastruct(uav_data, from_file=from_file, **register_params)
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
            data = Datastruct(uav_data, from_file=match_matrix_path2, **register_params)
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
        data = Datastruct(uav_data, from_file=match_matrix_path2, **register_params)
        HGlobal = np.load(hglobal_path)[data.load_indices]

    print("#####################################################################\n")
    ####################################################################################################################




    # 4. Visualization #################################################################################################
    print("#####################################################################")
    print("Visualization starts now \n")
    if visualization_params['run']:
        if 'n_processes' in visualization_params and visualization_params['n_processes'] is not None:
            parallel_params.update(n_processes=visualization_params['n_processes'])

        visualize(data=data, HGlobal=HGlobal, **visualization_params, parallel_params=parallel_params)

    print("#####################################################################\n")
    ####################################################################################################################


def read_params(config_file, need_all=False):
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
        parallel_params = get_from_dict(config['parallel_params'], required_parallel_params, need_all)
        
        if 'n_processes' in parallel_params:
            parallel_params.update(n_jobs=parallel_params['n_processes'] if parallel_params['n_processes'] is not None
                                          else os.cpu_count())

        if 'n_threads' in parallel_params:
            parallel_params.update(n_threads=parallel_params['n_threads'] if parallel_params['n_threads'] is not None
                                             else os.cpu_count())

        parallel_params.update(prefer='processes',
                               individual_worker_pools=False)

    else:
        parallel_params = dict()

    # PREPROC params
    if 'preprocessing_params' in config:
        required_preproc_params = ('load_data', 'load_features', 'write', 'keep_data_in_memory',
                                   'ransac_residual_threshold', 'rescale_interpolation_method',
                                   'interpolation_method', 'pairing_min_samples',
                                   'pairing_transform_type', 'pairing_finetune_metric',
                                   'pairing_master_band', 'pairing_optical_flow_refinement',
                                   'radiance_gaussian_blur', 'sif_computation_method')
        preproc_params = get_from_dict(config['preprocessing_params'], required_preproc_params, need_all)

    else:
        preproc_params = dict()

    # REGISTRATION params
    if 'registration_params' in config:
        required_register_params = ('load', 'write', 'min_n_correspondences', 'ransac_residual_threshold',
                                    'n_features_trial', 'min_movement', 'protect_correspondences_in_close_images', 
                                    'close_images_limit', 'spread_points', 'min_movement')
        register_params = get_from_dict(config['registration_params'], required_register_params, need_all)

    else:
        register_params = dict()

    # ALIGN params
    if 'alignment_params' in config:
        required_align_params = ('load', 'write',
                                 'optim_params',
                                 'n_outlier_removal',
                                 'manual_outlier_mode',
                                 'n_points', 'outlier_threshold',
                                 'resume_optimization', 
                                 'remove_disconnected_threshold')
        align_params = get_from_dict(config['alignment_params'], required_align_params, need_all)

    else:
        align_params = dict()

    # VISUALIZATION params
    if 'vis_params' in config:
        required_vis_params = ('mosaic_resolution', 'mosaic_origin', 'interpolation_method', 'run', 'n_processes',
                               'products', 'aggregation_methods')
        vis_params = get_from_dict(config['vis_params'], required_vis_params, need_all)

    else:
        vis_params = dict()

    # SENSOR params
    if 'sensor_calibration_config' in config:
        if not os.path.isabs(config['sensor_calibration_config']):
            config['sensor_calibration_config'] = os.path.join(os.path.dirname(config_file), config['sensor_calibration_config'])
        
        sensor_params = dict(sensor_calibration_config=config['sensor_calibration_config'])

    else:
        sensor_params = dict()


    # DATA params
    required_data_params = ('dataset_path', 'result_path',
                            'camera_regex', 'datatake_regex')
    data_params = get_from_dict(config['data_params'], required_data_params, need_all)


    # FEATURE params
    if 'feature_params' in config:
        required_feature_params = ('n_SIFT_features_per_image', 'n_features_threshold',
                                   'feature_product', 'mask')
        feature_params = get_from_dict(config['feature_params'], required_feature_params, need_all)
        if 'mask' in feature_params:
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


def deep_update(dest, src):
    """
    Recursively update ``dest`` with the contents of ``src``.

    * If a key in ``src`` maps to a dictionary, the corresponding value in ``dest``
      is also turned into a dictionary (created if it does not exist) and the
      recursion continues.
    * Otherwise the value from ``src`` overwrites whatever is stored in ``dest``.
    * The function mutates ``dest`` in‑place and returns ``None`` – this mirrors the
      behaviour of ``dict.update`` while adding recursion.
    """
    for key, src_val in src.items():
        if isinstance(src_val, dict):
            # Make sure the destination has a dict to merge into
            dest_val = dest.get(key)
            if not isinstance(dest_val, dict):
                dest[key] = {}                # create a fresh dict if missing / wrong type
                dest_val = dest[key]
            deep_update(dest_val, src_val)   # recurse
        else:
            dest[key] = src_val               # leaf – simple overwrite


def read_config(config_path):
    """
    Load the *default* configuration from ``config/default_config.yaml`` and
    overlay the user supplied ``config_path`` on top of it.

    The merge is **deep**, i.e. nested dictionaries are merged recursively.
    Missing sections in the default file are created on‑the‑fly so the result
    always contains a full configuration tree.

    Parameters
    ----------
    config_path:
        Path to a YAML file that contains user‑provided overrides.

    Returns
    -------
    dict
        The resulting configuration dictionary.
    """
    default_file = pjoin(os.path.dirname(__file__), "config", "default_config.yaml")
    default_params = read_params(default_file, need_all=True)   
    user_params    = read_params(config_path)                 

    deep_update(default_params, user_params)
    return default_params


def get_from_dict(dic, keys, need_all=False):
    ret = dict()

    for key in keys:
        if need_all:
            ret[key] = dic.get(key, None)

        elif not need_all and key in dic:
            ret[key] = dic[key]

        else:
            pass

    return ret

