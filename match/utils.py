# SPDX-FileCopyrightText: © 2025 Forschungszentrum Jülich GmbH
# SPDX-FileContributor: Armagan Elibol
# SPDX-FileContributor: Christian Hirt
# SPDX-FileContributor: Jim Buffat <j.buffat@fz-juelich.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from skimage.transform import warp, AffineTransform, ProjectiveTransform, PiecewiseAffineTransform, \
        SimilarityTransform, ThinPlateSplineTransform
import skimage
from skimage.measure import ransac
from packaging.version import Version

import numpy as np
import cv2

from data.utils import get_flann
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
    matches = flann.knnMatch(im1.astype(np.float32), im2.astype(np.float32), k=2)

    # ratio test as per Lowe's paper
    return np.sum([m.distance < 0.8 * n.distance for i, (m, n) in enumerate(matches)])


def feature_matcher_flann(features1, features2, valid_points1, valid_points2,
                          img1=None, img2=None, flann='FLANNBasedMatcher', 
                          th=0.5, alignment_dim=2, min_samples=4, transform_type=None, 
                          stop_sample_num=np.inf):
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
    matches = flann.knnMatch(features1.astype(np.float32), 
                             features2.astype(np.float32), k=2)

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
            min_samples=min_samples_ransac, residual_threshold=th, 
            stop_sample_num=stop_sample_num,
            max_trials=2500 * 5
        )

        if inliers is not None and np.sum(inliers) > min_samples and tformT:
            # Order inliers by quality
            residuals = tformT.residuals(matched_points1, matched_points2)
            inlier_residuals = residuals[inliers]
            order = np.argsort(inlier_residuals)

            # Sorted inlier points
            matched_points1 = matched_points1[inliers][order]
            matched_points2 = matched_points2[inliers][order]

            if Version(skimage.__version__) >= Version("0.26"):
                model = ProjectiveTransform.from_estimate(
                    matched_points1,
                    matched_points2,
                )
                success = bool(model)
            else:
                model = ProjectiveTransform()
                success = model.estimate(matched_points1, matched_points2)

            if transform_type in ('affine', 'projective', 'similarity') and success:
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


    features757, validPoints757, status = run_SIFT(Im757.copy(), n_features)
    features760, validPoints760, status2 = run_SIFT(Im760.copy(), n_features)
    
    if features757 is None or features760 is None:
        return None, 1
    
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
        new_tformT = finetune_registration(Im757, Im760, tformT, transform_type=transform_type,
                                           interpolation_method=interpolation_method,
                                           metric=finetune_metric, points=(points1, points2))
        status3 = 0

        tformT = new_tformT
    
    return tformT, not(not status and not status2 and not status3)

    
def run_SIFT(Im, nbr, mask=None, apply_clahe=True, apply_sharpening=True, apply_downscale=True):
    """
    This function attempts to create nbr SIFT features in image Im.

    Args:
        Im: image (numpy array)
        nbr: number of SIFT features to be created
        mask: masking the image for feature generation
        apply_clahe: whether to apply CLAHE before SIFT
        apply_sharpening: whether to apply sharpening before SIFT
        apply_downscale: whether to downscale the image by a factor of 0.5 befire SIFT

    Returns:

    """
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
        
        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
            Im = clahe.apply(Im)
        
        if apply_sharpening:
            blur = cv2.GaussianBlur(Im.astype(float), (0, 0), 1.0)
            Im = cv2.addWeighted(Im.astype(float), 1.5, blur, -0.5, 0).astype(np.uint8)

        if apply_downscale:
            scale = 0.5
            Im = cv2.resize(
                            Im,
                            None,
                            fx=scale,
                            fy=scale,
                            interpolation=cv2.INTER_AREA
                        )

        keypoints, features = sift.detectAndCompute(Im, mask)
        points = np.array([keypoint.pt for keypoint in keypoints], dtype=np.float32)

        responses = np.array([keypoint.response for keypoint in keypoints], dtype=np.float32)
        order = np.argsort(responses)[::-1]

        points = np.asarray(points[order])
        features = np.asarray(features[order])

        if apply_downscale:
            for i, kp in enumerate(points):
                points[i][0] /= scale
                points[i][1] /= scale

    except Exception as e:
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
