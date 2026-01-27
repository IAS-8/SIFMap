# SPDX-FileCopyrightText: © 2025 Forschungszentrum Jülich GmbH
# SPDX-FileContributor: Armagan Elibol
# SPDX-FileContributor: Christian Hirt
# SPDX-FileContributor: Jim Buffat <j.buffat@fz-juelich.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from skimage.transform import warp, AffineTransform, ProjectiveTransform, PiecewiseAffineTransform, \
        SimilarityTransform, ThinPlateSplineTransform
from skimage.measure import ransac
from skimage.transform import warp
from skimage import io, transform
from scipy.optimize import minimize
import scipy.ndimage as ndimage

import numpy as np
import cv2

from data.utils import get_flann
from scipy.spatial._qhull import QhullError
from match.finetune import finetune_registration
from match.epipolar import epipolar_pose_and_depth_estimation


interpolation_flags_CV2 = dict(linear=cv2.INTER_LINEAR, cubic=cv2.INTER_CUBIC,
                               nearest=cv2.INTER_NEAREST)
interpolation_flags_skimage = dict(linear=1, nearest=0, cubic=3)


def feature_matcher_flann_init(im1, im2, flann=None):
    """
        This function is to match feature descriptors 
    Args:
        im1 (): feature descriptors from first image data
        im2 (): feature descriptors from second image data
        flann (): feature-matching object or dictionary to instantiate object

    Returns:
        counter : a number of matched descriptors
    """
    flann = get_flann(flann)
    matches = flann.knnMatch(im1, im2, k=2)

    # ratio test as per Lowe's paper
    return np.sum([m.distance < 0.8 * n.distance for i, (m, n) in enumerate(matches)])


def feature_matcher_flann(features1, features2, valid_points1, valid_points2,
                          img1=None, img2=None, flann='FLANNBasedMatcher', 
                          th=0.5, alignment_dim=2, min_samples=4, transform_type=None):
    """

    Args:
        features1 (): feature descriptors from first image data
        features2 (): feature descriptors from second image data
        valid_points1 (): feature location data from first image data
        valid_points2 (): feature location data from second image data
        flann : feature matching object
        th (): RANSAC threshold to identify outliers
    Returns:
        tformT : computed transformation between a pair.
        status : control parameter of the registration procedure
        matched_points1 : matched point locations in the first image data
        matched_points2 : matched point locations in the second image data
    """


    flann = get_flann(flann)
    matches = flann.knnMatch(features1, features2, k=2)

    good_matches = []

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.6 * n.distance:
            good_matches.append(m)

    matched_points1 = np.asarray([valid_points1[match.queryIdx] for match in good_matches], dtype=np.float32)
    matched_points2 = np.asarray([valid_points2[match.trainIdx] for match in good_matches], dtype=np.float32)
    
    if transform_type is None:
        transform_type = {2:'affine', 3:'projective'}[alignment_dim]

    min_samples_ransac = 3 * int(alignment_dim <= 2) + 4 * int(alignment_dim > 2)

    if transform_type == 'epipolar':
        #tformT, inliers = ransac(
        #        (matched_points1, matched_points2), ProjectiveTransform,
        #        min_samples=3, residual_threshold=th, max_trials=25000 * 5
        #    )
        
        #matched_points1 = matched_points1[inliers]
        #matched_points2 = matched_points2[inliers]

        tformT, inliers, status = epipolar_pose_and_depth_estimation(matched_points1, matched_points2,
                                                                     camera_matrix=camera_matrix, ransac_thresh=th, 
                                                                     img1=img1, img2=img2)

        if status == 0:
            matched_points1 = matched_points1[inliers]
            matched_points2 = matched_points2[inliers]

        else:
            matched_points1 = []
            matched_points2 = []

        return tformT, status, matched_points1, matched_points2
      
    elif transform_type in ('affine', ):
        model = AffineTransform

    elif transform_type == 'projective':
        model = ProjectiveTransform

    elif transform_type == 'piecewise_affine':
        model = PiecewiseAffineTransform
        min_samples_ransac = 13

    elif transform_type == 'similarity':
        model = SimilarityTransform

    else:
        raise NotImplementedError()
    
    if len(matched_points1) > min_samples_ransac:
        # robustly estimate transform model with RANSAC
        tformT, inliers = ransac(
            (matched_points1, matched_points2), model,
            min_samples=int(min_samples_ransac), residual_threshold=th, max_trials=2500 * 5
        )

        if inliers is not None and np.sum(inliers) > min_samples:
            matched_points1 = matched_points1[inliers]
            matched_points2 = matched_points2[inliers]
            
            model = model()
            success = model.estimate(matched_points1, matched_points2)

            if transform_type in ('affine', 'projective', 'similarity'):
                tformT = np.asarray(model.params)

            else:
                tformT = model

            status = 0 if success is True else 1

        else:
            tformT = None
            status = 2
            matched_points1 = []
            matched_points2 = []

    else:
        tformT = None
        status = 3
        matched_points1 = []
        matched_points2 = []

    return tformT, status, matched_points1, matched_points2


def accurate_image_matcher(Im757, Im760, flann, mask=None, transform_type=None, min_samples=4, 
                           finetune_metric=False, interpolation_method='linear', n_features=50000):

    thr = 0.5
    features757, validPoints757, status = run_SIFT(Im757.copy(), n_features) # mask=Im757 > np.quantile(Im757.flatten(), thr))
    features760, validPoints760, status2 = run_SIFT(Im760.copy(), n_features) # mask=Im760 > np.quantile(Im760.flatten(), thr))

    tformT, status3, points1, points2 = feature_matcher_flann(features757,
                                                              features760,
                                                              validPoints757,
                                                              validPoints760,
                                                              Im757,
                                                              Im760,
                                                              flann,
                                                              th=flann['ransac_residual_threshold'],
                                                              alignment_dim=flann.get('alignment_dim', 3), 
                                                              min_samples=min_samples,
                                                              transform_type=transform_type)
    
    if finetune_metric is not None and tformT is not None:
        #assert transform_type in ('affine', 'projective'), \
        #        f'Finetuning can only run with transform_type "affine" or "projective", ' \
        #        f'but you requested "{transform_type}".'
        new_tformT = finetune_registration(Im757, Im760, tformT, transform_type=transform_type,
                                           interpolation_method=interpolation_method,
                                           metric=finetune_metric, points=(points1, points2))
        status3 = 0  # 2 if np.linalg.det(tformT) < 1e-6 else 0

        tformT = new_tformT
    
    return tformT, not(not status and not status2 and not status3)

    
def run_SIFT(Im, nbr, mask=None):
    status = 0

    sift = cv2.SIFT_create(nbr)

    if np.issubdtype(Im.dtype, np.floating):
        mask_ = np.isnan(Im)
        mask_ = mask_ if np.any(mask_.flatten()) else None

    else:
        mask_ = None

    if mask is not None: 
        mask = mask if mask_ is None else np.logical_or(mask, mask_)

    else:
        mask = mask_

    if mask is not None:
        Im[mask] = 0
        mask = np.logical_not(mask).astype(np.uint8)

    try:
        if not Im.dtype == np.uint8:
            Im = (255 * normalize(Im)).astype('uint8')

        keypoints, features = sift.detectAndCompute(Im, mask)
        points = np.array([keypoint.pt for keypoint in keypoints], dtype=np.float32)

    except Exception:
        features = None
        points = None
        status = 2

    return features, points, status


def normalize(Im, qs=None):
    if qs is None:
        qs = [0, 1]
    min_, max_ = np.nanquantile(Im.flatten(), qs)
    Im = (Im - min_) / (max_ - min_)
    return Im
