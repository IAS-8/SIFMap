# SPDX-FileCopyrightText: © 2025 Forschungszentrum Jülich GmbH
# SPDX-FileContributor: Armagan Elibol
# SPDX-FileContributor: Christian Hirt
# SPDX-FileContributor: Jim Buffat <j.buffat@fz-juelich.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import skimage as ski
import networkx as nx
from scipy.optimize import least_squares

from align.calc import vector_to_affine_homography, calc_residual_offsets, get_mosaic_size
from align.jacobians import global_2d_jacobian
from align.optimize import optimize_alignment_3d, point_match_residual_2d
from data.geometry import get_pose_from_absolute, convert_bundle_to_gmml, convert_bundle_to_data, set_resolution
from data.utils import timeit, print_text_histogram


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
        for mm in range(1, len(data)):
            t1 = np.linalg.inv(IniH[0][:]) @ IniH[mm][:]
            tt1 = t1.flatten()
            xini.append(tt1[0:6])

        xini = np.array(xini)
        xini = xini.flatten()
    
    else:
        xini = init_x

    _optim_params = dict(verbose=2,x_scale='jac', ftol=1e-6, xtol=1e-3)
    if optim_params is not None:
        _optim_params.update(optim_params)
    
    orig = [1, 0, 0, 0, 1, 0]
    out = least_squares(fun=point_match_residual_2d, x0=xini, jac=global_2d_jacobian, args=(data, orig), **_optim_params)

    HRes = vector_to_affine_homography(out.x)
    GlobalH = [np.identity(3) for _ in range(len(data))]

    #convert image-to-map to map-to-image
    for i in range(1, len(data)):
        HH = np.linalg.inv(HRes[i - 1])
        GlobalH[i] = HH / HH[2, 2]

    return GlobalH, out.x


def global_align_2d_w_outlier_analysis(IniH, data, min_n_correspondences=14, n_points=50,
                                       n_outlier_removal=0, manual_outlier_mode=True, optim_params=None,
                                       min_movement=None, outlier_threshold=5, init_x=None, **kwargs):
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
        for mm in range(1, len(data)):
            t1 = np.linalg.inv(IniH[0][:]) @ IniH[mm][:]
            tt1 = t1.flatten()
            xini.append(tt1[0:6])

        xini = np.array(xini)
        xini = xini.flatten()

    else:
        xini = init_x

    stop = False
    i = 0
    orig = [1, 0, 0, 0, 1, 0]
    while not stop:

        _optim_params = dict(verbose=2, x_scale='jac', ftol=1e-6, xtol=1e-3)
        if optim_params is not None:
            _optim_params.update(optim_params)

        result = least_squares(fun=point_match_residual_2d, x0=xini, jac=global_2d_jacobian, args=(data, orig),
                               **_optim_params)


        xini = result.x
        x_reshaped = xini.reshape(len(data) - 1, -1)

        if manual_outlier_mode:
            manual_continue = get_user_input('Do you want to continue with outlier removal (0/1): ',
                                             expected_dtype=int)

        else:
            manual_continue = False

        #if n_outlier_removal > 0 or manual_continue:
        data, n_corr_removals = outlier_analysis(data, result.fun, outlier_threshold=outlier_threshold)

        #if flag > 0:
        remove = data.update_matches(min_n_correspondences=min_n_correspondences, n_points=n_points,
                                     min_movement=min_movement)
        print(f'Outlier analysis removed {n_corr_removals} correspondences and {len(remove)} images')
        print('Running with the following correspondence statistics:') # data.correspondence_statistics())
        print_text_histogram(data.matches[:, 2])

        remove_first = 0 in remove
        remove = [r - 1 for r in remove if r > 0] 
        if len(remove) > 0:
            xini = np.delete(x_reshaped, remove, axis=0).flatten()

        if remove_first:
            print('Removing image 0')
            #orig = xini[:x_reshaped.shape[0]]
            xini = xini[x_reshaped.shape[1]:]

        i += 1
        stop = (i == n_outlier_removal and not manual_outlier_mode) \
                    or (not manual_continue and manual_outlier_mode)

    HRes = vector_to_affine_homography(xini)
    orig = vector_to_affine_homography(orig)
    HGlobal = [orig[0] for _ in range(len(data))]
    for i in range(1, len(data)):
        HH = np.linalg.inv(HRes[i - 1])
        HGlobal[i] = HH / HH[2, 2]

    return np.asarray(HGlobal), xini


def global_align_3d_outlier_analysis(IniH, data, mosaic_origin, mosaic_resolution, KMat,
                                     n_outlier_removal=0, n_points=50, min_n_correspondences=14,
                                     manual_outlier_mode=True, parallel_params=None, optim_params=None,
                                     min_movement=None, outlier_threshold=5, init_x=None,
                                     **kwargs):
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
    
    if init_x is None:
        _, _, _, wHi, x, residuals, parallel_context = optimize_alignment_3d(K, data,
                                                                             RTs=RTs, optim_params=optim_params,
                                                                             parallel_params=parallel_params)

    else:
        _, _, _, wHi, x, residuals, parallel_context = optimize_alignment_3d(K, data, x=init_x,
                                                                             optim_params=optim_params,
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

        data, n_corr_removals = outlier_analysis(data, residuals, outlier_threshold=outlier_threshold,
                                                 pair_removal_thr=None)

        x_reshaped = x.reshape(len(data), 6)
        if n_outlier_removal > 0:
            remove = data.update_matches(min_n_correspondences=min_n_correspondences, n_points=n_points,
                                         min_movement=min_movement)
            x = np.delete(x_reshaped, remove, axis=0).flatten()

        print(f'Outlier analysis removed {n_corr_removals} correspondences and {len(remove)} images')
        print('Running with the following correspondence statistics:')   # data.correspondence_statistics())
        print_text_histogram(data.matches[:, 2])

        # Don't update motion parameters, just point and match data
        #_, _, PointMatchesIdx, PointMatchesData = convert_bundle_to_data(IniPose, data,
        #                                                                 KMat, mosaic_origin)
        # optimize function should come here
        _, _, _, wHi, x, residuals, parallel_context = optimize_alignment_3d(K, data, x=x,
                                                                            parallel_context=parallel_context,
                                                                            optim_params=optim_params,
                                                                            parallel_params=parallel_params)

        q += 1

    H, FinalPose = convert_bundle_to_gmml(wHi, x, mosaic_origin, mosaic_resolution)
    H = set_resolution(H, mosaic_resolution)

    HGlobal = [np.identity(3) for _ in range(len(data))]
    for i in range(len(data)):
        HH = np.linalg.inv(H[i][:])
        HGlobal[i][:] = HH / HH[2, 2]

    if parallel_context is not None:
        try:
            parallel_context.__exit__(None, None, None)

        except:
            pass

    return HGlobal, x


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

    if init_x is None:
        _, _, _, wHi, x, residuals, parallel_context = optimize_alignment_3d(K, data,
                                                                             RTs=RTs, optim_params=optim_params,
                                                                             parallel_params=parallel_params)

    else:
        _, _, _, wHi, x, residuals, parallel_context = optimize_alignment_3d(K, data, x=init_x,
                                                                             optim_params=optim_params,
                                                                             parallel_params=parallel_params)

    HRes1, FinalPose = convert_bundle_to_gmml(wHi, x, mosaic_origin, mosaic_resolution)
    HRes1 = set_resolution(HRes1, mosaic_resolution)

    HGlobal = [np.identity(3) for _ in range(len(data))]
    for i in range(len(data)):
        HH = np.linalg.inv(HRes1[i][:])
        HGlobal[i][:] = HH / HH[2, 2]

    return HGlobal, x


def outlier_analysis(data, residuals, pair_removal_thr=None, outlier_threshold=5):
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

            if pair_removal_thr is not None and len(Ix) > pair_removal_thr:
                pair_removal = True

            #if Ix.size > 0:
            #    Ixx = np.where((SubRes[Ix] < -7.5) | (SubRes[Ix] > 7.5))[0]
            #    if Ixx.size > 0:
            #        RemID = np.unique(np.concatenate((RemID, Ix[Ixx])))

            RemID = np.unique(np.concatenate((RemID, Ix)))
              
        data.remove_correspondence(Im1, Im2, correspondence_id=RemID)
        if pair_removal:
            print(f'Remove pair {(Im1, Im2)}')
            data.remove_pairs(np.array([[Im1, Im2]]))

        n_removals += len(RemID)
 
    return data, n_removals


def get_init_H(data):
    """
        This functions computes the initial image-to-map transformations based on 
        Minimum spanning tree algorithm and shortest path between images.
    Args:
        data (): Data structure to keep the successfully matched
            correspondences positions (x,y) between overlapping image pairs

    Returns:
        IniH: Initial estimates for image-to-map transformations
    """
    n_images = data.match_matrix.shape[0]

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

    return IniH


@timeit
def align(data, case, init_H=None, init_x=None, **kwargs):
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
        init_H = get_init_H(data)

    # global alignment either in 2D or 3D (accurate camera instrinsics needed)
    
    if case == 1:
        HGlobal, x = global_align_2d(init_H, data, init_x=init_x, **kwargs)

    elif case == 2:
        HGlobal, x = global_align_2d_w_outlier_analysis(init_H, data, init_x=init_x, **kwargs)

    elif case == 3:
        HGlobal, x = global_align_3d(init_H, data, init_x=init_x, **kwargs)

    elif case == 4:
        HGlobal, x = global_align_3d_outlier_analysis(init_H, data, init_x=init_x, **kwargs)

    else:
        raise ValueError('case must be in [1, 2, 3, 4]')

    return data, np.asarray(HGlobal), x
