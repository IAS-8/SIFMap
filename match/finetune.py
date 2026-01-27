import cv2
import numpy as np

interpolation_flags_CV2 = dict(linear=cv2.INTER_LINEAR, cubic=cv2.INTER_CUBIC,
                               nearest=cv2.INTER_NEAREST)
interpolation_flags_skimage = dict(linear=1, nearest=0, cubic=3)

def normalize(Im, qs=None):
    if qs is None:
        qs = [0, 1]
    min_, max_ = np.nanquantile(Im.flatten(), qs)
    Im = (Im - min_) / (max_ - min_)
    return Im

def generate_control_points(img_shape, grid_spacing=20):
    """Generates a grid of control points."""
    h, w = img_shape
    y, x = np.mgrid[0:h:grid_spacing, 0:w:grid_spacing]
    return np.column_stack([x.ravel(), y.ravel()])  # Flatten to Nx2 shape

def apply_piecewise_affine_transform(img, src_points, dst_points, interpolation_method):
    """Applies a piecewise affine transformation."""
    tform = PiecewiseAffineTransform()
    tform.estimate(src_points, dst_points)
    warped = transform.warp(img, tform, output_shape=img.shape,
                            order=interpolation_flags_skimage[interpolation_method])

    # Convert NaNs to zeros (same behavior as cv2.warpAffine)
    #warped[np.isnan(warped)] = 0

    return warped, tformT

def laplacian_highpass(img):
    """Applies a Laplacian filter to emphasize high frequencies."""
    return cv2.Laplacian(img, cv2.CV_64F)

def identity_warp(image):
    """Applies an identity warp to introduce the same resampling artifacts."""
    h, w = image.shape
    identity_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    warped = cv2.warpAffine(image, identity_matrix, (w, h), flags=cv2.INTER_LINEAR)
    return warped

def compute_mi(image1, image2, bins=64):
    """Computes Mutual Information (MI) between two images using a joint histogram."""
    hist_2d, _, _ = np.histogram2d(image1.ravel(), image2.ravel(), bins=bins)

    # Normalize joint histogram
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)  # Marginal probability for image1
    py = np.sum(pxy, axis=0)  # Marginal probability for image2

    px_py = np.outer(px, py)  # Compute product of marginal probabilities
    nz = pxy > 0  # Only consider non-zero probabilities
    mi = np.sum(pxy[nz] * np.log(pxy[nz] / px_py[nz]))  # Compute MI

    return mi

def transform_matrix(params, initial_params, transform_type):
    """Creates either an affine (3x3) or projective (homography) transformation matrix."""
    if transform_type == 'affine':
        a, b, c, d, tx, ty = initial_params * params
        return np.array([[a, b, tx], [c, d, ty], [0, 0, 1]])

    elif transform_type == 'projective':
        a, b, c, d, tx, ty, p, q = initial_params * params
        return np.array([[a, b, tx], [c, d, ty], [p, q, 1]])

    else:
        raise ValueError("transform_type must be either 'affine' or 'projective'")

def apply_transform(image, shape, params, initial_params, transform_type, interpolation_method):
    """Applies an affine or projective transformation."""
    if len(params.shape) == 1:
        matrix = transform_matrix(params, initial_params, transform_type)

    else:
        matrix = params

    #if transform_type == 'affine':
    #    transformed_image = cv2.warpAffine(image, matrix[:2, :], (image.shape[1], image.shape[0]),
    #                                       flags=interpolation_flags_CV2[interpolation_method])
    #else:
    transformed_image = cv2.warpPerspective(image, matrix, (shape[1], shape[0]),
                                            flags=interpolation_flags_CV2[interpolation_method])

    # Identify invalid (empty) pixels and mark them as NaN
    valid_mask = transformed_image != 0
    transformed_image[~valid_mask] = np.nan
    return transformed_image

def ncc_with_exclusion(image1, image2, mask=None):
    """Computes the normalized cross-correlation (NCC) excluding NaN pixels."""
    if mask is None:
        mask = np.ones_like(image1, dtype=bool)

    valid_pixels = mask.ravel()
    image1_valid = image1.ravel()[valid_pixels]
    image2_valid = image2.ravel()[valid_pixels]

    mean1, mean2 = np.mean(image1_valid), np.mean(image2_valid)
    std1, std2 = np.std(image1_valid) + 1e-10, np.std(image2_valid) + 1e-10
    N = len(valid_pixels)

    return np.dot(image1_valid - mean1, image2_valid - mean2) / (N * std1 * std2)

def ssd_with_exclusion(image1, image2, mask=None):
    """Computes the Sum of Squared Differences (SSD) excluding NaN pixels."""
    if mask is None:
        mask = np.ones_like(image1, dtype=bool)

    valid_pixels = mask.ravel()
    image1_valid = image1.ravel()[valid_pixels]
    image2_valid = image2.ravel()[valid_pixels]

    ssd = np.sqrt(np.sum((image1_valid - image2_valid) ** 2)) / (len(image1_valid) + 1e-2)  # Normalize by valid pixels
    return ssd

def mi_with_exclusion(image1, image2, mask=None, bins=256):
    """Computes Mutual Information (MI) while ignoring NaN pixels."""
    if mask is None:
        mask = np.ones_like(image1, dtype=bool)

    valid_pixels = mask.ravel()
    image1_valid = image1.ravel()[valid_pixels]
    image2_valid = image2.ravel()[valid_pixels]

    return compute_mi(image1_valid, image2_valid, bins)

def laplace_diff_with_exclusion(img1, img2, mask=None):
    img1_hf = laplacian_highpass(img1)
    img2_hf = laplacian_highpass(img2)

    if mask is not None:
        img1_hf = img1_hf[mask]
        img2_hf = img2_hf[mask]

    return np.nanmean((img1_hf - img2_hf) ** 2)

def objective_function(params, img1, img2, initial_params, transform_type,
                       interpolation_method='linear', metric='ssd'):

    if transform_type != 'affine_both':
        transformed_img1 = apply_transform(img1, img2.shape, params, initial_params, transform_type,
                                           interpolation_method=interpolation_method)
        transformed_img2 = img2

    else:
        raise NotImplementedError()

        matrix = transform_matrix(params, initial_params, transform_type)

        translation = matrix[:2, 2]
        matrix = matrix[:2, :2]

        U, Sigma, Vt = np.linalg.svd(matrix)
        aff1 = np.linalg.inv((U  - 0.5 * translation) @ np.sqrt(np.diag(Sigma)))
        aff2 = np.diag(np.sqrt(Sigma)) @ Vt + 0.5 * translation

        aff1 = np.concatenate((np.concatenate((aff1, np.zeros((1, 2)))), np.zeros((3, 1))), axis=1)
        aff2 = np.concatenate((np.concatenate((aff2, np.zeros((1, 2)))), np.zeros((3, 1))), axis=1)

        transformed_img1 = apply_transform(img1, img2.shape, aff1, initial_params, transform_type,
                                           interpolation_method=interpolation_method)
        transformed_img2 = apply_transform(img2, img1.shape, aff2, initial_params, transform_type,
                                           interpolation_method=interpolation_method)

    mask = ~np.logical_or(np.isnan(transformed_img1), np.isnan(transformed_img2))
    return get_loss(transformed_img1, transformed_img2, mask, metric)

def get_loss(img1, img2, mask, metric):
    if metric == 'ncc':
        loss = 1 - ncc_with_exclusion(img1, img2, mask)

    elif metric == 'ssd':
        loss = ssd_with_exclusion(img1, img2, mask)

    elif metric == 'mi':
        loss = -mi_with_exclusion(img1, img2, mask)

    elif metric == 'laplace':
        loss = laplace_diff_with_exclusion(img1, img2, mask)

    else:
        raise ValueError("Invalid metric: choose 'ncc' or 'ssd'")

    return loss  #loss + 0.5 * hf_loss

def objective_function_piecewise_affine(params, img1, img2, initial_params, src_points, metric,
                                        interpolation_method):
    """Objective function for piecewise affine optimization."""
    num_points = len(src_points)
    dst_points = src_points + (params + initial_params).reshape(num_points, 2)  # Update points

    transformed_img1, tformT = apply_piecewise_affine_transform(img1, src_points, dst_points,
                                                                interpolation_method=interpolation_method)

    mask = transformed_img1 > 0  # Exclude empty pixels
    return get_loss(transformed_img1, img2, mask, metric)

def optimize_registration(img1, img2, initial_params, bounds, optim_params, transform_type,
                          interpolation_method='linear', metric='ssd',
                          with_std=False):
    """Optimizes the transformation parameters to maximize NCC."""
    params = np.ones(len(initial_params))

    result = minimize(objective_function, params,
                      args=(img1, img2, initial_params, transform_type, interpolation_method,
                            metric),
                      method='Nelder-Mead', bounds=bounds, options=optim_params)
    print(result)
    return result.x, result.success

def optimize_registration_piecewise_affine(img1, img2, tformT, bounds, optim_params,
                                               interpolation_method='linear', metric='ncc',
                                               grid_spacing=100, with_std=False, points=None):
        """Optimizes a piecewise affine transformation."""
        if points is None:
            src_points = generate_control_points(img1.shape, grid_spacing)
            # Apply initial transformation
            ones = np.ones((src_points.shape[0], 1))
            src_homogeneous = np.hstack([src_points, ones])
            dst_points = tformT(src_points)

            mask = dst_points[:, 0] > 0
            dst_points = dst_points[mask]
            src_points = src_points[mask]

        else:
            src_points = points[0]
            dst_points = points[1]

            mask = np.logical_and(np.any(np.isnan(points[0]), axis=1),
                                  np.any(np.isnan(points[1]), axis=1))
            dst_points = dst_points[~mask]
            src_points = src_points[~mask]

        initial_params = (dst_points - src_points).ravel()  # Initial displacement

        # Set reasonable bounds (±10 pixels per control point)
        bounds = [bounds] * len(initial_params)
        params = np.zeros_like(initial_params)

        result = minimize(objective_function_piecewise_affine, params,
                          args=(img1, img2, initial_params, src_points, metric, interpolation_method),
                          method='L-BFGS-B', bounds=bounds, options=optim_params)

        optimized_displacements = result.x.reshape(-1, 2)
        optimized_dst_points = src_points + optimized_displacements + initial_params.reshape(-1, 2)

        _, tformT = apply_piecewise_affine_transform(img2, src_points, optimized_dst_points,
                                                     interpolation_method=interpolation_method)

        print(result)
        return tformT, result.success

def finetune_registration(img1, img2, tformT, transform_type='affine', interpolation_method='linear',
                          metric='ssd', points=None):

    # **Set up the optimization parameters**
    if transform_type in ('affine', 'affine_both'):
        initial_params = np.array([tformT[0, 0], tformT[0, 1], tformT[1, 0],
                                   tformT[1, 1], tformT[0, 2], tformT[1, 2]])
        bounds = [(0.95, 1.05)] * 4 + [(0.95, 1.05)] * 2 # Scaling 10% around initial values
        optim_params = dict(adaptive=True, fatol=1e-10, xatol=1e-10, maxfev=5000)

    elif transform_type == 'projective':
        initial_params = np.array([tformT[0, 0], tformT[0, 1], tformT[1, 0], tformT[1, 1],
                                   tformT[0, 2], tformT[1, 2], tformT[2, 0], tformT[2, 1]])
        bounds = [(0.95, 1.05)] * 4  + [(0.95, 1.05)] * 2  + [(0.95, 1.05)] * 2
        optim_params = dict(adaptive=True, fatol=1e-10, xatol=1e-10, maxfev=5000)

    elif transform_type == 'piecewise_affine':
        bounds = (-3, 3)
        optim_params = dict(gtol=1e-15, ftol=1e-21, maxls=10)

    else:
        raise ValueError("Unsupported transform_type. Choose 'affine' or 'projective'.")

    # Normalize images and optimize
    img1_norm = 256 * normalize(img1)
    img2_norm = 256 * normalize(img2)

    if transform_type in ('affine', 'affine_both', 'projective'):
        params, success = optimize_registration(img1_norm, img2_norm, initial_params.copy(), bounds,
                               optim_params, transform_type, interpolation_method=interpolation_method,
                               metric=metric)

        new_tformT = transform_matrix(params, initial_params, transform_type)

    elif transform_type == 'piecewise_affine':
        try:
            new_tformT, success = optimize_registration_piecewise_affine(img1_norm, img2_norm, tformT,
                                            bounds, optim_params, interpolation_method=interpolation_method,
                                            metric=metric, points=points)

        except (QhullError, ValueError) as e:
            new_tformT, success = None, False

    if success:
        tformT = new_tformT

    else:
        print('Finetuning failed.')

    return tformT
