# SPDX-FileCopyrightText: © 2025 Forschungszentrum Jülich GmbH
# SPDX-FileContributor: Armagan Elibol
# SPDX-FileContributor: Christian Hirt
# SPDX-FileContributor: Jim Buffat <j.buffat@fz-juelich.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from match.utils import feature_matcher_flann_init, feature_matcher_flann
from data.utils import run_jobs
from functools import partial
from data.utils import timeit, profile
from joblib import Parallel
from data.utils import init_with_valid_kwargs
from contextlib import nullcontext


def pairwise_image_registration(matches, data, flann, min_n_correspondences, parallel_params=None,
                                parallel_context=None, **kwargs):
    """
        This function is to attempt to match given a list of image pairs
        via descriptor matching and RANSAC
    Args:
        matches (): a list of possible overlapping image pairs
        data (): data structure including SIF data obtained from
        registering C757-C760 pairs 
        flann (): matching object
        min_n_correspondences (): total number of correspondences for an overlapping image pair
            to be considered as successful
    Returns:
        data : data structure keeping the correspondences data
    """
    if parallel_params is None:
        parallel_params = dict(do_parallel=False)

    out = []

    jobs = []
    for i in range(len(matches)):
        im1 = int(matches[i][1])
        im2 = int(matches[i][0])

        fea1 = data.images[im1]['features']
        valP1 = data.images[im1]['valid_points']

        fea2 = data.images[im2]['features']
        valP2 = data.images[im2]['valid_points']

        # Preparing parallel execution of
        # tformT, status, matchedPoints1, matchedPoints2 = featureMatcherFLANN(fea1, fea2, valP1, valP2, flann, th=1.5)
        jobs.append(partial(feature_matcher_flann, fea1, fea2, valP1, valP2, 
                            flann=flann,
                            th=flann['ransac_residual_threshold'],
                            alignment_dim=flann['alignment_dim'], 
                            #stop_sample_num=10*min_n_correspondences
                            ))

    # Execute in parallel
    # output is ordered as tformT, status, matchedPoints1, matchedPoints2
    parallel_out = run_jobs(jobs, **parallel_params, parallel_context=parallel_context)

    for i in range(len(matches)):
        tformT, status, matchedPoints1, matchedPoints2 = parallel_out[i]
        im1 = int(matches[i][1])
        im2 = int(matches[i][0])

        delList = []
        enough_matches = len(matchedPoints1) >= min_n_correspondences \
                        if not data.is_close_images(im1, im2) \
                        else True 

        if (tformT is not None) and enough_matches:
            # delete some repeated feature positions
            _, uni1 = np.unique(matchedPoints1, axis=0, return_index=True)
            _, uni2 = np.unique(matchedPoints2, axis=0, return_index=True)

            nbr = len(matchedPoints1)
            vct = np.arange(nbr)

            delList.extend([np.where(np.isin(vct, uni2) == False),
                            np.where(np.isin(vct, uni1) == False)])
            delList = np.unique(np.column_stack((delList[0], delList[1])))

            matchedPoints1 = np.delete(matchedPoints1, delList, axis=0)
            matchedPoints2 = np.delete(matchedPoints2, delList, axis=0)

            data.match_matrix[im2][im1] = np.array(matchedPoints1, dtype=float)
            data.match_matrix[im1][im2] = np.array(matchedPoints2, dtype=float)

            out.append([im2, im1, matchedPoints1.shape[0]])

        else:
            pass

    data.matches = np.array(out).astype(int)
    return data


def coarse_overlapping_image_pair_list_generation(data, flann, parallel_params=None, parallel_context=None,
                                                  n_features_trial=250, **kwargs):
    """
        This function is to create a possible overlapping image pairs 
        to be verified by RANSAC later.
    Args:
        data (): data structure including SIF data obtained from
        registering C757-C760 pairs 
        flannObj (): matching object

    Returns:
        match_save_list : a list of possibly overlapping image pairs.
    """
    if parallel_params is None:
        parallel_params = dict(do_parallel=False)

    match_save_list = []

    images = data.images
    jobs = []
    inds = []

    # compute initial similarity with SIF images all-against-all manner.
    for im1 in range(0, len(images)):
        fea1 = images[im1]['features'][:n_features_trial, :]

        for im2 in range(im1 + 1, len(images)):
            fea2 = images[im2]['features'][:n_features_trial, :]

            # Prepare parallel execution of
            func = partial(feature_matcher_flann_init, fea1, fea2, flann)
            jobs.append(func)
            inds.append([im1, im2])

    # Execute jobs in parallel
    parallel_out = run_jobs(jobs, **parallel_params, parallel_context=parallel_context)
    for n_matches, (im1, im2) in zip(parallel_out, inds):
        # if there is more than 4 matches between image pairs
        if n_matches >= 4 or data.is_close_images(im1, im2):
            match_save_list.append([im2, im1, n_matches])

    return np.array(match_save_list)


@profile()
def register_images(data, min_n_correspondences, flann, ransac_residual_threshold=0.15,
                    parallel_params=None, min_movement=None, **kwargs):
    """
        This function is to carry out pairwise image registration 
        for the entire dataset.
    Args:
        data (): data structure keeping the correspondences data
        min_n_correspondences (): total number of correspondences for an overlapping image pair
            to be considered as successful
        flann (): image matching object
        registering C757-C760 pairs

    Returns:
        data : data structure keeping the correspondences data
        matches : a list of overlapping image pairs
    """
    flann.update(ransac_residual_threshold=ransac_residual_threshold)

    if parallel_params is None:
        parallel_params = dict(do_parallel=False)

    _create_parallel_context = parallel_params['do_parallel'] and not parallel_params['individual_worker_pools']
    if _create_parallel_context:
        context = init_with_valid_kwargs(Parallel, **parallel_params)
    else:
        context = nullcontext()

    with context:
        parallel_context = context if _create_parallel_context else None

        # possible overlapping SIF image pairs generation
        print("Possible overlapping SIF image pairs generation starts now by coarse subset feature matching")
        matches = coarse_overlapping_image_pair_list_generation(data, flann,
                                                                parallel_params=parallel_params,
                                                                parallel_context=parallel_context,
                                                                **kwargs)

        # pairwise SIF image registration
        print("Pairwise SIF image pairs registration starts now using FLANN matching and RANSAC motion estimation.",  
              f"Nr of match possibilities to be checked: {len(matches)}")
        data = pairwise_image_registration(matches, data, flann, min_n_correspondences,
                                           parallel_params=parallel_params,
                                           parallel_context=parallel_context,
                                           **kwargs)

    # clean up
    _ = data.update_matches(min_n_correspondences=min_n_correspondences,
                            min_movement=min_movement)
    return data



