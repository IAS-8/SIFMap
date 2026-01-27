
def epipolar_pose_and_depth_estimation(pts1, pts2, camera_matrix, thr, img1, img2):
    """
    Estimate (i) the pose of two cameras with img1 and img2 and (ii) the depth map of img1. 
    The estimated pose should not deviate significantly from the a priori known camera setup 
    with a horizontal baseline of 0.068 m.

    Inputs:
        - pts1, pts2 : coarsly matched SIFT feature point coordinates
        - camera_matrix : 3x3 camera_matrix
        - thr: threshold to control matching of pts1 and pts2, i.e. by RANSAC
        - img1, img2: images to be matched

    Returns:
        (dict(...), inliers, status)
        
        - dict(...) : dictionary with arguments necessary for warp. This dictionary 
                      is passed to epipolar_warp as **kwargs
        - inliers : boolean array, whether a point is an inlier
        - status : {0, 1}, whether a pose could be established and depth_map created, 
                   NOTE: this assumes bash standard, i.e. 0 means True and 1 means False
    """

    raise NotImplementedError()


def epipolar_warp(img1, img2, camera_matrix, **kwargs):
    """
    Warps img2 to img1 given kwargs computed in epipolar_pose_and_depth_estimation.

    Returns:
        (img1_aligned, img2_aligned, status)

        - img1_aligned, img2_aligned: aligned images img1 and img2, these need not have the same shape as img1 and img2
        - status : {0, 1}, whether the images could be aligned,
                   NOTE: this assumes bash standard, i.e. 0 means True and 1 means False
    """

    raise NotImplementedError()

