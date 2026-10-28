import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter

from data.utils import load_image, save

# ==============================================================================
# GLOBAL CACHE FOR FFT KERNELS
# ==============================================================================
KERNEL_FFT_CACHE = {}


def clear_kernel_cache():
    """Clears the FFT kernel cache to free memory."""
    KERNEL_FFT_CACHE.clear()


def get_kernel_fft(length, angle, k_size, patch_shape, quantize=True):
    """
    Retrieves or computes the FFT of a motion kernel.
    Uses discretization to maximize cache hits and avoid regenerating kernels.
    """
    # Discretize inputs to increase cache hit rate (0.5px length, 5deg angle)
    if quantize:
        l_disc = round(float(length) * 2) / 2.0
        a_disc = round(float(angle) / 5.0) * 5.0

    else:
        l_disc = length
        a_disc = angle

    key = (l_disc, a_disc, k_size, patch_shape)

    if key not in KERNEL_FFT_CACHE:
        # 1. Generate spatial kernel
        kernel_spatial = create_motion_kernel_subpixel(l_disc, a_disc, k_size)

        # 2. Pad to patch shape and center for FFT
        ph, pw = patch_shape
        kernel_padded = np.zeros((ph, pw), dtype=np.float32)

        ky, kx = kernel_spatial.shape
        cy, cx = ky // 2, kx // 2
        hy, hx = ph // 2, pw // 2

        y_start = max(0, hy - cy)
        x_start = max(0, hx - cx)
        y_end = min(ph, y_start + ky)
        x_end = min(pw, x_start + kx)

        src_y_start = max(0, cy - hy)
        src_x_start = max(0, cx - hx)

        kernel_padded[y_start:y_end, x_start:x_end] = kernel_spatial[src_y_start:src_y_start + (y_end - y_start),
                                                      src_x_start:src_x_start + (x_end - x_start)]

        # 3. FFT Shift (move center to corner) and Transform
        kernel_fft = np.fft.fft2(np.fft.ifftshift(kernel_padded))
        KERNEL_FFT_CACHE[key] = kernel_fft

    return KERNEL_FFT_CACHE[key]


# ==============================================================================
# CORE ESTIMATION LOGIC (OPTIMIZED)
# ==============================================================================

def normalize_patch(patch, valid_mask=None):
    """
    Normalizes a patch to Mean=0, StdDev=1.
    Ignores NaN/Inf values based on the mask.
    """
    if valid_mask is None:
        valid_mask = np.isfinite(patch)

    valid_data = patch[valid_mask]

    if len(valid_data) < 10:
        return patch

    mean = np.mean(valid_data)
    std = np.std(valid_data)

    if std < 1e-6:
        # No contrast: return zeros to avoid influencing MSE
        return np.zeros_like(patch)

    normalized = (patch - mean) / std
    normalized[~valid_mask] = 0.0

    return normalized.astype(np.float32)


def estimate_motion_params_fft(clean_patch, blurry_patch):
    """
    Estimates motion blur length and angle using a Coarse-to-Fine strategy
    with FFT-based convolution for speed.
    """

    def get_simulated_blurry(clean_norm, length, angle, k_size, patch_shape=None, quantize=True, C_fft=None, fft=False):
        if fft:
            ph, pw = patch_shape
            K_fft = get_kernel_fft(length, angle, k_size, (ph, pw), quantize=False)
            sim_spatial = np.real(np.fft.ifft2(C_fft * K_fft))
        
        else:
            kernel = create_motion_kernel_subpixel(length, angle, k_size)
            sim_spatial = cv2.filter2D(clean_norm, -1, kernel, anchor=(-1, -1), borderType=cv2.BORDER_REFLECT)

        return sim_spatial

    def objective(x, clean_norm, k_size, patch_shape=None, quantize=True, C_fft=None, fft=False):
        length = np.exp(x[0])
        angle = x[1]
        length = np.clip(length, MIN_LEN, MAX_LEN)
        
        sim_spatial = get_simulated_blurry(clean_norm, length, angle, k_size, 
                                           patch_shape=patch_shape, quantize=quantize, C_fft=C_fft, fft=fft)
        mse = np.mean((sim_spatial - blurry_norm) ** 2) #+ 0.00001 * length

        if not np.isfinite(mse): return 1e9
        return mse

    # 1. Validity Check
    if np.any(~np.isfinite(clean_patch)) or np.any(~np.isfinite(blurry_patch)):
        return 0.5, 0.0

    global_valid_mask = np.isfinite(clean_patch) & np.isfinite(blurry_patch)
    if np.sum(global_valid_mask) < (clean_patch.size * 0.1):
        return 0.5, 0.0

    # 2. Normalize
    clean_norm = normalize_patch(clean_patch, global_valid_mask)
    blurry_norm = normalize_patch(blurry_patch, global_valid_mask)

    ph, pw = clean_norm.shape
    MAX_LEN = 32
    MIN_LEN = 0.5

    # Kernel size setup
    k_size = int(np.ceil(MAX_LEN)) * 3
    if k_size % 2 == 0: k_size += 1
    k_size = min(k_size, 41)

    # Pre-compute FFTs of patches (constant for this optimization)
    C_fft = np.fft.fft2(clean_norm)
    B_fft = np.fft.fft2(blurry_norm)

    # --- STEP 1: COARSE GRID SEARCH ---
    lengths = np.linspace(MIN_LEN, MAX_LEN, 5)
    angles = [0.0, 45.0, 90.0, 135.0]

    best_mse = np.inf
    best_params = (MIN_LEN, 0.0)
    
    use_fft = False
    quantize = True
    args = (clean_norm, k_size, (ph, pw), quantize, C_fft, use_fft)
    for l in lengths:
        for a in angles:
            mse = objective((np.log(l), a), *args)
            if mse < best_mse:
                best_mse = mse
                best_params = (l, a)

    
    quantize = False
    args = (clean_norm, k_size, (ph, pw), quantize, C_fft, use_fft)
    x0 = [np.log(best_params[0]), best_params[1]]
    bounds = [(np.log(MIN_LEN), np.log(MAX_LEN)), (0.0, 180.0)]
    options = {'ftol': 1e-8, 'gtol': 1e-9, 'maxiter': 500, 'eps': 0.0001}
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options=options, args=args)

    if not np.all(np.isfinite(res.x)):
        return best_params

    length = np.exp(res.x[0])
    angle = res.x[1]

    return float(np.clip(length, MIN_LEN, MAX_LEN)), float(angle)


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def ensure_float32_and_normalize(img):
    """Converts image to float32 and normalizes to [0.0, 1.0]."""
    original_dtype = img.dtype

    if original_dtype == np.uint8:
        scale = 255.0
        img_float = img.astype(np.float32) / scale
    elif original_dtype == np.uint16:
        scale = 65535.0
        img_float = img.astype(np.float32) / scale
    elif np.issubdtype(original_dtype, np.floating):
        max_val = np.max(img)
        if max_val > 1.0:
            scale = max_val
            img_float = img.astype(np.float32) / scale
        else:
            scale = 1.0
            img_float = img.astype(np.float32)
    else:
        scale = 1.0
        img_float = img.astype(np.float32)

    return img_float, original_dtype, scale


def create_weight_map(height, width):
    """Creates a cosine weight map (bell shape) that is 0 at the edges."""
    yy, xx = np.meshgrid(np.linspace(-1, 1, height), np.linspace(-1, 1, width))
    radius = np.sqrt(xx ** 2 + yy ** 2)
    mask = np.zeros_like(radius, dtype=np.float32)
    valid = radius <= 1.0
    mask[valid] = (1.0 + np.cos(np.pi * radius[valid])) / 2.0
    return mask


def apply_patchwise_blur(clean_img, blurry_img, patch_size=20, overlap=0, mode='blur',
                         outpath_rad=None, outpath_mb=None):
    """
    Full pipeline for motion blur estimation and application.

    Args:
        clean_img: Sharp reference image.
        blurry_img: Blurry input image.
        patch_size: Size of patches for estimation.
        overlap: Overlap between patches for feathering.
        mode: 'blur' (simulate blur) or 'deblur' (Richardson-Lucy).
    """
    # 1. Normalize inputs
    ############ Load data
    if type(clean_img) is str:
        clean_img = load_image(clean_img, divide_by=1000).astype(float)

    if type(blurry_img) is str:
        blurry_img = load_image(blurry_img, divide_by=1000).astype(float)

    clean_f, dtype_clean, scale_clean = ensure_float32_and_normalize(clean_img)
    blurry_f, dtype_blurry, scale_blurry = ensure_float32_and_normalize(blurry_img)

    if mode == 'blur':
        dtype, scale = dtype_clean, scale_clean
    else:
        dtype, scale = dtype_blurry, scale_blurry

    h, w = clean_f.shape[:2]

    # 2. Initialize Maps
    map_length = np.zeros((h, w), dtype=np.float32)
    map_angle = np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)

    # Convert to grayscale for estimation
    if len(clean_f.shape) == 3:
        c_gray = cv2.cvtColor(clean_f, cv2.COLOR_BGR2GRAY)
        b_gray = cv2.cvtColor(blurry_f, cv2.COLOR_BGR2GRAY)
    else:
        c_gray, b_gray = clean_f, blurry_f

    print(f"Step 1: Estimating parameters (Mode: {mode})...")
    clear_kernel_cache()

    steps_y = range(0, h, patch_size - overlap)
    steps_x = range(0, w, patch_size - overlap)

    # 3. Patchwise Estimation
    for y in steps_y:
        for x in steps_x:
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)
            p_clean = c_gray[y:y_end, x:x_end]
            p_blur = b_gray[y:y_end, x:x_end]

            if p_clean.shape[0] < 10 or p_clean.shape[1] < 10:
                continue

            est_len, est_ang = estimate_motion_params_fft(p_clean, p_blur)

            # Feathering
            ph, pw = p_clean.shape
            mask = np.ones((ph, pw), dtype=np.float32)
            feather = overlap // 2
            if feather > 0:
                ramp = (1.0 - np.cos(np.linspace(0, np.pi, feather))) / 2.0
                if y > 0: mask[:feather, :] *= ramp[:, None]
                if y_end < h: mask[-feather:, :] *= ramp[::-1][:, None]
                if x > 0: mask[:, :feather] *= ramp[None, :]
                if x_end < w: mask[:, -feather:] *= ramp[::-1][None, :]

            map_length[y:y_end, x:x_end] += est_len * mask
            map_angle[y:y_end, x:x_end] += est_ang * mask
            weight_map[y:y_end, x:x_end] += mask

    # Normalize and Smooth Maps
    weight_map[weight_map == 0] = 1.0
    map_length /= weight_map
    map_angle /= weight_map

    sin_map = np.sin(np.deg2rad(map_angle))
    cos_map = np.cos(np.deg2rad(map_angle))
    smooth_sigma = (patch_size - overlap) * 3

    map_length = gaussian_filter(map_length, sigma=smooth_sigma)
    sin_map = gaussian_filter(sin_map, sigma=smooth_sigma)
    cos_map = gaussian_filter(cos_map, sigma=smooth_sigma)

    map_angle_smoothed = np.rad2deg(np.arctan2(sin_map, cos_map))
    map_angle_smoothed = (map_angle_smoothed + 360) % 180

    # 4. Application (Weighted Accumulation)
    print(f"Step 2: Applying {mode}...")

    result_accum = np.zeros_like(clean_f, dtype=np.float64)
    weight_accum = np.zeros((h, w), dtype=np.float64)

    max_estimated_len = np.max(map_length)
    if max_estimated_len < 1.0: max_estimated_len = 1.0

    required_radius = int(np.ceil(max_estimated_len * 1.5)) + 4
    block_size = 2 * required_radius + 4
    if block_size < 32: block_size = 32

    half_bs = block_size // 2
    block_weight_mask = create_weight_map(block_size, block_size)
    block_step = 8
    
    #if np.sum(np.where(map_length < 0.6)) / np.prod(map_length.shape) > 0.9:
    #    return clean_f, map_length, map_angle_smoothed

    for y in range(0, h, block_step):
        for x in range(0, w, block_step):
            y_end_param = min(y + block_step, h)
            x_end_param = min(x + block_step, w)

            avg_len = np.mean(map_length[y:y_end_param, x:x_end_param])
            avg_ang = np.mean(map_angle_smoothed[y:y_end_param, x:x_end_param])

            center_y = y + block_step // 2
            center_x = x + block_step // 2

            y_start = max(0, center_y - half_bs)
            y_end = min(h, center_y + half_bs)
            x_start = max(0, center_x - half_bs)
            x_end = min(w, center_x + half_bs)

            actual_h = y_end - y_start
            actual_w = x_end - x_start

            if actual_h <= 0 or actual_w <= 0:
                continue

            if mode == 'deblur':
                context_block = blurry_f[y_start:y_end, x_start:x_end]
            else:
                context_block = clean_f[y_start:y_end, x_start:x_end]

            if avg_len < 0.2:
                processed_block = context_block
            else:
                current_k_size = int(np.ceil(avg_len * 1.5)) + 4
                if current_k_size % 2 == 0: current_k_size += 1

                kernel = create_motion_kernel_subpixel(avg_len, avg_ang, current_k_size)

                if mode == 'blur':
                    processed_block = cv2.filter2D(context_block, -1, kernel, anchor=(-1, -1),
                                                   borderType=cv2.BORDER_REFLECT)
                elif mode == 'deblur':
                    processed_block = apply_richardson_lucy_deblur(context_block, kernel, iterations=3)
                else:
                    processed_block = context_block

            mask_y_start = (block_size - actual_h) // 2
            mask_x_start = (block_size - actual_w) // 2
            current_weight_mask = block_weight_mask[mask_y_start:mask_y_start + actual_h,
                                  mask_x_start:mask_x_start + actual_w]

            result_accum[y_start:y_end, x_start:x_end] += processed_block.astype(np.float64) * current_weight_mask
            weight_accum[y_start:y_end, x_start:x_end] += current_weight_mask

    weight_accum[weight_accum == 0] = 1.0
    final_result_float = (result_accum / weight_accum).astype(np.float32)

    final_result = denormalize_and_convert(final_result_float, dtype, scale)

    if outpath_rad is not None:
        save([final_result], [outpath_rad])
        final_result = outpath_rad

    motion_blur = np.stack((map_length, map_angle_smoothed))
    if outpath_mb is not None:
        save([motion_blur], [outpath_mb])
        motion_blur = outpath_mb

    return final_result, motion_blur


def denormalize_and_convert(img_float, original_dtype, scale):
    """Scales float32 image [0,1] back to original dtype and range."""
    img_clipped = img_float
    if np.issubdtype(original_dtype, np.integer):
        img_scaled = img_clipped * scale
        return np.round(img_scaled).astype(original_dtype)
    else:
        return (img_clipped * scale).astype(original_dtype)


def create_motion_kernel_subpixel(length, angle, k_size):
    """
    Creates a motion blur kernel with sub-pixel accuracy using bilinear interpolation.
    """
    length = max(float(length), 0.0)
    kernel = np.zeros((k_size, k_size), dtype=np.float32)
    center = (k_size - 1) / 2.0

    if length < 1e-6:
        kernel[int(round(center)), int(round(center))] = 1.0
        return kernel

    theta = np.deg2rad(angle)
    n_samples = max(32, int(np.ceil(length * 16)))
    t = np.linspace(-length / 2, length / 2, n_samples)

    xs = center + t * np.cos(theta)
    ys = center + t * np.sin(theta)

    for x, y in zip(xs, ys):
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        dx = x - x0
        dy = y - y0

        for iy, wy in [(y0, 1.0 - dy), (y0 + 1, dy)]:
            for ix, wx in [(x0, 1.0 - dx), (x0 + 1, dx)]:
                if 0 <= iy < k_size and 0 <= ix < k_size:
                    kernel[iy, ix] += wx * wy

    total = kernel.sum()
    if total > 0:
        kernel /= total

    return kernel


def apply_richardson_lucy_deblur(image, kernel, iterations=15):
    """Richardson-Lucy deconvolution using FFT."""
    h, w = image.shape[:2]
    ky, kx = kernel.shape

    kernel_padded = np.zeros((h, w), dtype=np.float32)
    cy, cx = ky // 2, kx // 2
    start_y = (h // 2) - cy
    start_x = (w // 2) - cx

    sy, ey = max(0, start_y), min(h, start_y + ky)
    sx, ex = max(0, start_x), min(w, start_x + kx)
    ky_start = max(0, -start_y)
    kx_start = max(0, -start_x)

    kernel_padded[sy:ey, sx:ex] = kernel[ky_start:ky_start + (ey - sy), kx_start:kx_start + (ex - sx)]
    kernel_for_fft = np.fft.ifftshift(kernel_padded)
    H = np.fft.fft2(kernel_for_fft)

    kernel_flip = np.flip(np.flip(kernel, 0), 1)
    kernel_flip_padded = np.zeros((h, w), dtype=np.float32)
    kernel_flip_padded[sy:ey, sx:ex] = kernel_flip[ky_start:ky_start + (ey - sy), kx_start:kx_start + (ex - sx)]
    kernel_flip_for_fft = np.fft.ifftshift(kernel_flip_padded)
    H_flip = np.fft.fft2(kernel_flip_for_fft)

    if len(image.shape) == 3:
        result = np.zeros_like(image, dtype=np.float32)
        for c in range(3):
            result[:, :, c] = _rl_core(image[:, :, c], H, H_flip, iterations)
        return result
    else:
        return _rl_core(image, H, H_flip, iterations)


def _rl_core(img_channel, H, H_flip, iterations):
    """Core Richardson-Lucy loop for a single channel."""
    estimate = img_channel.astype(np.float64) + 1e-7

    for _ in range(iterations):
        conv = np.real(np.fft.ifft2(np.fft.fft2(estimate) * H))
        conv[conv < 1e-7] = 1e-7

        ratio = img_channel.astype(np.float64) / conv
        correction = np.real(np.fft.ifft2(np.fft.fft2(ratio) * H_flip))

        estimate *= correction

    return np.clip(estimate, 0.0, 1.0).astype(np.float32)


