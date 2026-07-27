# SPDX-FileCopyrightText: © 2025 Forschungszentrum Jülich GmbH
# SPDX-FileContributor: Armagan Elibol
# SPDX-FileContributor: Christian Hirt
# SPDX-FileContributor: Jim Buffat <j.buffat@fz-juelich.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import skimage as ski
import networkx as nx
import copy
from scipy.optimize import least_squares

from align.calc import vector_to_affine_homography, get_mosaic_size, \
        convert_image_to_map_to_map_to_image_2d, convert_image_to_map_to_map_to_image_3d
from align.jacobians import global_2d_jacobian
from align.optimize import optimize_alignment_3d, point_match_residual_2d
from data.utils import timeit, profile 
from data.geometry import get_pose_from_absolute, convert_bundle_to_gmml, convert_bundle_to_data


def get_user_input(message, expected_dtype):
    while True:
        try:
            out = expected_dtype(input(message))
        except ValueError:
            print(f"Error: Input could not be parsed to {expected_dtype}")
            continue
        else:
            break

    return out


def global_align_2d(IniH, data, n_points=50, min_n_correspondences=14, optim_params=None,
                    min_movement=None, init_x=None, **kwargs):
    """
    This function finds image motions when they are all represented in
    a common map (or global) frame.
    It minimizes symmetric difference error between correspondences
    positions. Image-to-map motions are unknowns and 
    they are modeled as 2D affine planar transformations. First image frame is
    considered as map (or global) frame
    Args:
        IniH (): Initial estimate of image-to-map 2D planar transformations
        data (): Data structure to keep the successfully matched
            correspondences positions (x,y) between overlapping image pairs
        n_points (): number of correspondences per image pair to be used in
            global alignment procedure

    Returns:
        GlobalH : Final estimate of map-to-image image transformations
    """
    # use n_points, min_n_correspondences
    remove = data.update_matches(min_n_correspondences=min_n_correspondences, n_points=n_points,
                                 min_movement=min_movement)
    IniH = np.delete(IniH, remove, axis=0)
    if init_x is not None:
        dels = [np.arange(6*i, 6*(i+1)) for i in remove]
        init_x = np.delete(init_x, dels)
    
    if init_x is None:
        xini = []
        for mm in range(len(data)):
            t1 = np.linalg.inv(IniH[0][:]) @ IniH[mm][:]
            xini.append(t1.flatten())
        x = np.asarray(xini).flatten()

    else:
        x = init_x

    optim_params, _optim_params = prepare_optim_params(optim_params)
    print('Running least squares optimization with the followig parameters:', _optim_params)
    
    out = least_squares(fun=point_match_residual_2d, x0=x, jac=global_2d_jacobian, args=(data, ), **_optim_params)

    #convert image-to-map to map-to-image
    x_reshaped = np.concatenate((orig, out.x))
    GlobalH = convert_image_to_map_to_map_to_image_2d(data, x_reshaped)
    return GlobalH, out.x


def global_align_2d_w_outlier_analysis(IniH, data, min_n_correspondences=14, n_points=50,
                                       n_outlier_removal=0, manual_outlier_mode=True, optim_params=None,
                                       min_movement=None, outlier_threshold=5, init_x=None, remove_disconnected_threshold=None,
                                       **kwargs):
    """
    This function does global alignment and check residuals afterwards
    in case of there exists residuals outside of 5*standard deviation,
    it removes them and re-do the global alignment. As a result of removing,
    some overlapping image pairs and/or some images migth be removed as well

    Args:
        IniH (): Initial estimate of image-to-map 2D planar transformations
        data (): Data structure to keep the successfully matched
            correspondences positions (x,y) between overlapping image pairs
        n_points (): number of correspondences per image pair to be used in
            global alignment procedure

    Returns:
        HGlobal : Final estimate of map-to-image image transformations
        data : Data structure after some correspondences,
            image pairs and/or image removed as a result of outlier analysis
        remove : Images that are removed as a result of outlier analysis
    """
    # use n_points, min_n_correspondences
    remove = data.update_matches(min_n_correspondences=min_n_correspondences, n_points=n_points,
                                 min_movement=min_movement)
    IniH = np.delete(IniH, remove, axis=0)

    if init_x is not None:
        dels = [np.arange(6*i, 6*(i+1)) for i in remove]
        init_x = np.delete(init_x, dels)

    if init_x is None:
        xini = []
        for mm in range(len(data)):
            t = np.linalg.inv(IniH[0][:]) @ IniH[mm][:]
            xini.append(t.flatten()[:6])

        x_reshaped = np.asarray(xini)
        x = x_reshaped.flatten()

    else:
        x = init_x
        x_reshaped = x.reshape(len(data), 6)

    stop = False
    i = 0
    while not stop:

        optim_params, _optim_params = prepare_optim_params(optim_params)
        print('Running least squares optimization with the followig parameters:', _optim_params)

        result = least_squares(fun=point_match_residual_2d, x0=x, jac=global_2d_jacobian, args=(data, ),  # orig),
                               **__optim_params)

        if manual_outlier_mode:
            manual_continue = get_user_input('Do you want to continue with outlier removal (0/1): ',
                                             expected_dtype=int)

        else:
            manual_continue = False

        x = result.x
        x_reshaped = x.reshape(len(data), 6)
        
        data, x_reshaped = handle_disconnected(data,
                                           remove_disconnected_threshold,
                                           min_n_correspondences,
                                           n_points,
                                           min_movement,
                                           x=x_reshaped)

        data, x_reshaped, n_corr_removals = outlier_analysis(data,
                                                         result.fun,
                                                         outlier_threshold=outlier_threshold,
                                                         min_n_correspondnces=min_n_correspondences,
                                                         n_points=n_points,
                                                         min_movement=min_movement,
                                                         x=x_reshaped)
        x = x_reshaped.flatten()
        
        print(f'Outlier analysis removed {n_corr_removals} correspondences and {len(remove)} images')
        
        i += 1
        stop = (i == n_outlier_removal and not manual_outlier_mode) \
                or (not manual_continue and manual_outlier_mode)

    # convert image-to-map to map-to-image
    GlobalH = convert_image_to_map_to_map_to_image_2d(data, x)
    return GlobalH, x


def global_align_3d_outlier_analysis(IniH, data, mosaic_origin, mosaic_resolution, KMat,
                                     n_outlier_removal=0, n_points=50, min_n_correspondences=14,
                                     manual_outlier_mode=True, parallel_params=None, optim_params=None,
                                     min_movement=None, outlier_threshold=5, init_x=None,
                                     remove_disconnected_threshold=3, **kwargs):
    """
    This function models image-to-map transformations as 3D camera pose
    Rotation is modeled via quaternions. Symmetric transfer error is minimized
    Outlier analysis is carried out with the residuals and 5 times standard deviation
    Args:
        IniH (): Initial estimate of image-to-map 2D planar transformations
        image_size (): Image resolution, e.g., 
            ImageSize = {'Width':1024,'Height':1024,'Depth':3,'Bits':8}
        mosaic_origin (): amount of translation in order to have all images 
        in the positive area of the coordinate frame
        mosaic_resolution (): 1 pixel in mm
        KMat (): camera instrinsics
        data (): Data structure to keep the successfully matched
            correspondences positions (x,y) between overlapping image pairs
        n_points (): Number of correspondences per image pair to be used.

    Returns:
        HGlobal : Final estimate of map-to-image image transformations
        data : Data structure after some correspondences,
            image pairs and/or image removed as a result of outlier analysis
        remove : Images that are removed as a result of outlier analysis
    """

    if parallel_params is None:
        parallel_params = dict(do_parallel=False)
    
    remove = data.update_matches(min_n_correspondences=min_n_correspondences, n_points=n_points,
                                 min_movement=min_movement)

    IniH = np.delete(IniH, remove, axis=0)
    if init_x is not None:
        dels = [np.arange(6*i, 6*(i+1)) for i in remove]
        init_x = np.delete(init_x, dels)

    H = IniH
    
    MosaicSize, mosaic_origin, glob_trfm, HGlobal = get_mosaic_size(H, data, mosaic_origin, mosaic_resolution)
    IniPose = get_pose_from_absolute(H, mosaic_origin, mosaic_resolution, KMat)
    RTs, K = convert_bundle_to_data(IniPose, data, KMat, mosaic_origin)

    optim_params, _optim_params = prepare_optim_params(optim_params)
    print('Running least squares optimization with the followig parameters:', _optim_params)

    if init_x is None:
        _, _, _, wHi, x, residuals, parallel_context = optimize_alignment_3d(K, data,
                                                                             RTs=RTs, optim_params=_optim_params,
                                                                             parallel_params=parallel_params)

    else:
        _, _, _, wHi, x, residuals, parallel_context = optimize_alignment_3d(K, data, x=init_x,
                                                                             optim_params=_optim_params,
                                                                             parallel_params=parallel_params)

    q = 0
    while True:
        if manual_outlier_mode:
            manual_continue = get_user_input('Do you want to continue with outlier removal (0/1): ',
                                             expected_dtype=int)
        else:
            manual_continue = False

        stop = (q == n_outlier_removal and not manual_outlier_mode) \
               or (not manual_continue and manual_outlier_mode)

        if stop:
            break

        x_reshaped = x.reshape(len(data), 6)
        
        data, x_reshaped = handle_disconnected(data, 
                                               remove_disconnected_threshold, 
                                               min_n_correspondences, 
                                               n_points, 
                                               min_movement, 
                                               x=x_reshaped)

        data, x_reshaped, n_corr_removals = outlier_analysis(data, 
                                                             residuals, 
                                                             x=x_reshaped, 
                                                             outlier_threshold=outlier_threshold,
                                                             pair_removal_thr=None, 
                                                             min_n_correspondences=min_n_correspondences, 
                                                             n_points=n_points, 
                                                             min_movement=min_movement)

        x = x_reshaped.flatten()
         
        print(f'Outlier analysis removed {n_corr_removals} correspondences and {len(remove)} images')

        optim_params, _optim_params = prepare_optim_params(optim_params)
        print('Running least squares optimization with the followig parameters:', _optim_params)

        _, _, _, wHi, x, residuals, parallel_context = optimize_alignment_3d(K, data, x=x,
                                                                            parallel_context=parallel_context,
                                                                            optim_params=_optim_params,
                                                                            parallel_params=parallel_params)

        q += 1

    if parallel_context is not None:
        try:
            parallel_context.__exit__(None, None, None)

        except:
            pass

    GlobalH = convert_image_to_map_to_map_to_image_3d(data, x, wHi, mosaic_resolution, mosaic_origin)
    return GlobalH, x


def global_align_3d(IniH, data, mosaic_origin, mosaic_resolution, KMat, n_points=50,
                    min_n_correspondences=14, optim_params=None, parallel_params=None,
                    min_movement=None, init_x=None, **kwargs):
    """
        This function models image-to-map transformations as 3D camera pose
        Rotation is modeled via quaternions. Symmetric transfer error is minimized
    Args:
        image_size (): Image resolution, e.g.,
            ImageSize = {'Width':1024,'Height':1024,'Depth':3,'Bits':8}
        mosaic_origin (): amount of translation in order to have all images
        in the positive area of the coordinate frame
        mosaic_resolution (): 1 pixel in mm
        KMat (): camera instrinsics
        data (): Data structure to keep the successfully matched
            correspondences positions (x,y) between overlapping image pairs
        n_points (): Number of correspondences per image pair to be used.

    Returns:
        HGlobal : Final estimate of map-to-image image transformations
    """

    if parallel_params is None:
        parallel_params = dict(do_parallel=False)

    # use n_points, min_n_correspondences
    remove = data.update_matches(min_n_correspondences=min_n_correspondences, n_points=n_points,
                                 min_movement=min_movement)
    IniH = np.delete(IniH, remove, axis=0)

    MosaicSize, mosaic_origin, glob_trfm, HGlobal = get_mosaic_size(IniH, data, mosaic_origin, mosaic_resolution)
    IniPose = get_pose_from_absolute(IniH, mosaic_origin, mosaic_resolution, KMat)
    RTs, K = convert_bundle_to_data(IniPose, data, KMat, mosaic_origin)

    optim_params, _optim_params = prepare_optim_params(optim_params)
    print('Running least squares optimization with the followig parameters:', _optim_params)

    if init_x is None:
        _, _, _, wHi, x, residuals, parallel_context = optimize_alignment_3d(K, data,
                                                                             RTs=RTs, optim_params=_optim_params,
                                                                             parallel_params=parallel_params)

    else:
        _, _, _, wHi, x, residuals, parallel_context = optimize_alignment_3d(K, data, x=init_x,
                                                                             optim_params=_optim_params,
                                                                             parallel_params=parallel_params)

    GlobalH = convert_image_to_map_to_map_to_image_3d(data, x, wHi, mosaic_resolution, mosaic_origin)
    return GlobalH, x


def prepare_optim_params(optim_params):
    optim_params = dict([k for k in optim_params.items() if k[1] is not None])

    # translate optim_params to scipy.optimize
    if not 'tr_options' in optim_params:
        optim_params['tr_options'] = dict()

    if 'lsmr_maxiter' in optim_params:
        optim_params['tr_options']['maxiter'] = optim_params['lsmr_maxiter']
        del optim_params['lsmr_maxiter']
    
    # default values
    _optim_params = dict(verbose=2, x_scale='jac', ftol=1e-6, xtol=1e-3, method='trf', tr_solver='lsmr')
    if optim_params is not None:
        _optim_params.update(optim_params)
    
    if ('tr_options' in _optim_params and 'maxiter' in _optim_params['tr_options']
            and 'lsmr_dynamic_maxiter_factor' in _optim_params):

        if not 'lsmr_max_maxiter' in _optim_params:
            _optim_params['lsmr_max_maxiter'] = np.inf

        _optim_params['tr_options']['maxiter'] = min(int(_optim_params['tr_options']['maxiter'] \
                                                         * _optim_params['lsmr_dynamic_maxiter_factor']),
                                                     _optim_params['lsmr_max_maxiter'])

    __optim_params = copy.deepcopy(_optim_params)

    # translate optim_params to scipy.optimize
    for var in ('lsmr_maxiter', 'lsmr_dynamic_maxiter_factor', 'lsmr_max_maxiter'):
        if var in __optim_params:
            del __optim_params[var]
    
    return _optim_params, __optim_params


def outlier_analysis(data, residuals, x=None, outlier_threshold=5, 
                     min_n_correspondences=None, n_points=None, min_movement=None, 
                     x_adapter=None, **kwargs):
    """
        This function checks the residuals and tries to remove correspondences
        that are out of 5sigma regions
    Args:
        data (): Data structure to keep the successfully matched
            correspondences positions (x,y) between overlapping image pairs
        residuals (): Symmetric transfer error residuals between correspondences

    Returns:
        data : Data structure after some correspondences,
            image pairs and/or image removed as a result of outlier analysis
        flag : boolean if any removal procedure need to be carried out 
    """
    adapt_x = x is not None
    matches = data.matches
    n_pairs = len(matches)
    pos = np.concatenate((np.array([0]), np.cumsum(matches[:, 2])))

    NumStDev = [outlier_threshold] * 4
    m = [None] * 4
    s = [None] * 4

    n_removals = 0

    for ff in range(4):
        tmpRes = residuals[ff::4]
        a1 = np.argmax(tmpRes)
        a2 = np.argmin(tmpRes)
        tmpRes = np.delete(tmpRes, [a1, a2])
        m[ff] = np.mean(tmpRes)
        s[ff] = np.std(tmpRes)

    for i in range(n_pairs):
        Im1 = matches[i][0]
        Im2 = matches[i][1]
        
        RemID = np.array([], dtype=int)
        pair_removal = False

        rr = residuals[4 * pos[i]: 4 * pos[i+1]]
        for ff in range(4):
            SubRes = rr[ff::4]
            Ix = np.where(np.abs(SubRes) > m[ff] + NumStDev[ff] * s[ff])[0]
            RemID = np.unique(np.concatenate((RemID, Ix)))
              
        data.remove_correspondence(Im1, Im2, correspondence_id=RemID)
        n_removals += len(RemID)
        
    if adapt_x:
        remove = data.update_matches(min_n_correspondences=min_n_correspondences,
                                     n_points=n_points,
                                     min_movement=min_movement)
        if x_adapter is None:
            x = np.delete(x, remove, axis=0)
        else:
            x = x_adapter(x, remove)
    
        return data, x, n_removals

    return data, n_removals


def get_init_H(data, min_n_correspondences, n_points, min_movement, **kwargs):
    """
        This functions computes the initial image-to-map transformations based on 
        Minimum spanning tree algorithm and shortest path between images.
    Args:
        data (): Data structure to keep the successfully matched
            correspondences positions (x,y) between overlapping image pairs

    Returns:
        IniH: Initial estimates for image-to-map transformations
    """
    data.update_matches(min_n_correspondences, n_points, min_movement)

    n_images = len(data)  #data.match_matrix.shape[0]

    matches = np.array(data.matches, dtype='float')
    matches[:, 2] = np.around(1.0 / matches[:, 2], 5)

    A = np.zeros((n_images, n_images))

    for row in matches:
        A[int(row[0]), int(row[1])] = row[2]
        A[int(row[1]), int(row[0])] = row[2]

    G = nx.from_numpy_array(A)
    T = nx.minimum_spanning_tree(G)

    deg_ranks = nx.degree_centrality(T)
    sid = max(deg_ranks, key=deg_ranks.get)

    IniH = [np.eye(3) for _ in range(n_images)]

    for rt in set(range(0, n_images - 1)) - {sid}:
        temp = np.eye(3)
        if nx.has_path(T, sid, rt):
            Pat = nx.shortest_path(T, source=sid, target=rt)

            for nn in range(len(Pat) - 1):
                no2 = Pat[nn]
                no1 = Pat[nn + 1]
                Point = data.match_matrix[no1, no2]
                Match = data.match_matrix[no2, no1]

                tform = ski.transform.estimate_transform('euclidean', Match, Point)
                temp = temp @ tform.params

        IniH[rt] = temp / temp[2, 2]

    return np.asarray(IniH)


def check_connected(data):
    adj_matrix = np.zeros(data.match_matrix.shape)
    for m in data.matches:
        adj_matrix[m[0], m[1]] = m is not None and m[2] > 1
        adj_matrix[m[1], m[0]] = m is not None and m[2] > 1

    G = nx.from_numpy_array(adj_matrix)
    return G, nx.is_connected(G)


def handle_disconnected(data, remove_disconnected_threshold, min_n_correspondences,
                        n_points, min_movement, x=None, x_adapter=None, **kwargs):

    G, connected = check_connected(data)
    components = [np.array(sorted(c), dtype=int) for c in nx.connected_components(G)]
    len_per_component = np.asarray([len(c) for c in components])

    if not connected:
        print(f'WARINING: Found disconnected graph with these component sizes {len_per_component}.')
        stop = False
        n_deleted_images = 0

        while not stop:
            G, connected = check_connected(data)
            components = [np.array(sorted(c), dtype=int) for c in nx.connected_components(G)]
            len_per_component = np.asarray([len(c) for c in components])
            order = np.argsort(len_per_component)

            components = [components[o] for o in order]
            len_per_component = len_per_component[order]
            tbd = np.where(len_per_component <= remove_disconnected_threshold)[0]

            if len(tbd) == 0:
                break
            
            tbd = components[tbd[0]]
            data.remove(tbd)
            if x is not None and x_adapter is None:
                x = np.delete(x, tbd, axis=0)

            elif x is not None and x_apapter is not None:
                x = x_adapter(x, tbd)

            n_deleted_images += len(tbd)

    if x is not None:
        return data, x

    return data


#@timeit
@profile()
def align(data, init_H=None, init_x=None, **kwargs):
    """
        This function is to run the global alignment to obtain image-to-map transformations
    Args:
        data (): Data structure to keep the successfully matched
            correspondences positions (x,y) between overlapping image pairs
        image_size (): Image resolution, e.g.,
            ImageSize = {'Width':1024,'Height':1024,'Depth':3,'Bits':8}
        KMat (): camera instrinsic matrix
        mosaic_origin (): amount of translation in order to have all images
            in the positive area of the coordinate frame
        mosaic_resolution (): 1 pixel in mm
        n_points ():
        case (): Selection of global alignment method

    Returns:
        data : Final Data structure keeping correspondences among image pairs
        HGlobal : Final estimate of map-to-image image transformations
    """

    # compute initial image-to-map transformations based on graph theory
    if init_H is None:
        init_H = get_init_H(data, **kwargs)
    
    # global alignment either in 2D or 3D (accurate camera instrinsics needed)
    handle_disconnected(data, **kwargs) 

    if not kwargs['n_outlier_removal'] > 0:
        HGlobal, x = global_align_3d(init_H, data, init_x=init_x, **kwargs)

    else:
        HGlobal, x = global_align_3d_outlier_analysis(init_H, data, init_x=init_x, **kwargs)

    return data, np.asarray(HGlobal), x

