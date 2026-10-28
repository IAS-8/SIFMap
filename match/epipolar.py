import numpy as np
import cv2
from scipy.interpolate import griddata
from scipy import ndimage

def correct_points_using_residuals(pts1, pts2, camera_matrix, max_iter=10):
    """
    Iteratively corrects points by assuming a horizontal rig and removing
    the residual error (which represents motion blur/drift).
    Returns corrected points.
    """
    pts1_corr = pts1.copy().astype(np.float64)
    pts2_corr = pts2.copy().astype(np.float64)

    K = camera_matrix.astype(np.float64)
    K_inv = np.linalg.inv(K)

    # Pre-calculate K^-T and K^-1 for speed
    K_inv_T = K_inv.T

    for i in range(max_iter):
        # 1. Define the IDEAL Fundamental Matrix for a horizontal rig (R=I, t=[1,0,0])
        # E = [t]_x * R  => [t]_x since R=I
        # t = [1, 0, 0]
        t_x = np.array([
            [0, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ], dtype=np.float64)

        # F = K^-T * E * K^-1
        F_ideal = K_inv_T @ t_x @ K_inv

        # 2. Calculate Epipolar Lines for pts1 in Image 2
        # Line l2 = F * p1 (homogeneous)
        pts1_h = np.hstack([pts1_corr, np.ones((len(pts1_corr), 1))]).T # 3xN
        lines2 = F_ideal @ pts1_h # 3xN

        # 3. Calculate Distance of actual pts2 from these lines
        pts2_h = np.hstack([pts2_corr, np.ones((len(pts2_corr), 1))]).T

        # Distance d = (ax + by + c) / sqrt(a^2 + b^2)
        # lines2 rows: [a, b, c]
        a, b, c = lines2[0], lines2[1], lines2[2]
        denominators = np.sqrt(a**2 + b**2)

        # Avoid division by zero
        denominators[denominators == 0] = 1e-9

        distances = (a * pts2_h[0] + b * pts2_h[1] + c * pts2_h[2]) / denominators

        # 4. Calculate Shift Vector to move pts2 ONTO the line
        # The shortest path to the line is along the normal (a, b)
        # We want to move by -distance
        shift_x = -distances * (a / denominators)
        shift_y = -distances * (b / denominators)

        # Apply shift
        pts2_corr[:, 0] += shift_x
        pts2_corr[:, 1] += shift_y

        # Convergence check
        mean_error = np.mean(np.abs(distances))
        # Optional: Print for debugging
        # print(f"Iter {i+1}: Mean epipolar error = {mean_error:.4f} px")

        if mean_error < 0.5: # Sub-pixel convergence
            break

    return pts1_corr, pts2_corr, mean_error


def filter_points_by_epipolar_error(pts1, pts2, camera_matrix, threshold_px=1):
    """
    Filters points based on their epipolar error relative to a perfect horizontal rig.

    Logic:
    1. Define the ideal Fundamental Matrix for a horizontal rig (R=I, t=[1,0,0]).
    2. Calculate the epipolar line in Image 2 for every point in Image 1.
    3. Measure the perpendicular distance of the matched point in Image 2 from this line.
    4. If distance > threshold, mark point as invalid (False).

    Returns:
        mask (bool array): True for valid points, False for outliers.
        errors (float array): The calculated error for each point (for debugging).
    """
    pts1_h = pts1.astype(np.float64)
    pts2_h = pts2.astype(np.float64)
    K = camera_matrix.astype(np.float64)

    K_inv = np.linalg.inv(K)
    K_inv_T = K_inv.T

    # 1. Define Ideal Fundamental Matrix (Horizontal Rig)
    # t = [1, 0, 0], R = I
    t_x = np.array([
        [0, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ], dtype=np.float64)

    F_ideal = K_inv_T @ t_x @ K_inv

    # 2. Calculate Epipolar Lines in Image 2: l2 = F * p1
    pts1_hom = np.hstack([pts1_h, np.ones((len(pts1_h), 1))]).T # 3xN
    lines2 = F_ideal @ pts1_hom # 3xN

    # 3. Calculate Distances for pts2
    pts2_hom = np.hstack([pts2_h, np.ones((len(pts2_h), 1))]).T

    a, b, c = lines2[0], lines2[1], lines2[2]
    denominators = np.sqrt(a**2 + b**2)
    denominators[denominators == 0] = 1e-9 # Avoid div by zero

    # Distance formula: |ax + by + c| / sqrt(a^2 + b^2)
    errors = np.abs(a * pts2_hom[0] + b * pts2_hom[1] + c * pts2_hom[2]) / denominators

    # 4. Create Mask
    mask = errors <= threshold_px

    return mask, errors


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
    status = 1  # Default: Failure
    inliers = np.zeros(len(pts1), dtype=bool)
    
    pts1_cv = np.array(pts1, dtype=np.float32)
    pts2_cv = np.array(pts2, dtype=np.float32)
    
    camera_matrix = camera_matrix.astype(np.float32)

    if len(pts1_cv) < 8:
        return {}, inliers, status

    # 1. Fundamentalmatrix & Inliers
    # FM_RANSAC returns mask with 0 or 1 for each point
    F, mask = cv2.findFundamentalMat(pts1_cv, pts2_cv, cv2.FM_RANSAC, thr)

    if F is None or mask is None:
        return {}, inliers, status

    inliers = mask.ravel().astype(bool)
    pts1_in = pts1_cv[inliers]
    pts2_in = pts2_cv[inliers]

    if len(pts1_in) < 8:
        return {}, inliers, status

    # 2. Pose Recovery (R, t)
    # E = K^T * F * K
    E = camera_matrix.T @ F @ camera_matrix
    
    # recoverPose returns R (3x3) and t (3x1)
    # t is normalized to unit length.
    _, R, t_unit, _ = cv2.recoverPose(E, pts1_in, pts2_in, camera_matrix)

    # Sicherstellen, dass t die richtige Form (3, 1) hat für die Matrixverkettung
    if t_unit.shape == (3,):
        t_unit = t_unit.reshape(3, 1)

    diffs = pts2_in - pts1_in
    dx = diffs[:, 0]
    dy = diffs[:, 1]

    #R = np.eye(3, dtype=np.float64)
    #t_unit = np.array([1.0, - np.median(dy)/ np.median(dx), 0.0], dtype=np.float64)
    t_unit = np.array([1, 0.0, 0.0], dtype=np.float64)
    t_unit = t_unit / np.linalg.norm(t_unit)
    #R = np.eye(3)

    # Optional: Plausibilitätscheck der Rotation (muss orthogonal sein)
    # R * R.T sollte Identitätsmatrix sein
    ortho_error = np.linalg.norm(R @ R.T - np.eye(3))
    if ortho_error > 1e-6:
        # Falls R nicht orthogonal ist, ist die Schätzung schlecht
        return {}, inliers, status

    # print("Translation vector (unit):", t_unit.ravel())
    kwargs = {
        'R': R,
        't': t_unit.ravel(), # Als 3x1 Vektor übergeben
        'camera_matrix': camera_matrix,
    }

    status = 0  # Success
    return (kwargs, inliers, status)


def epipolar_warp(img1, img2, camera_matrix, switch_order=False, **kwargs):
    """
    Warps img2 to img1, then performs Dense Matching (SGM) to align them pixel-perfectly.

    Returns:
        {
            'img1_rect': Rectified image 1 (float32, NaNs on borders),
            'img2_rect': Rectified image 2 (float32, NaNs on borders),
            'img2_aligned': Image 2 shifted pixel-perfectly to match img1 (float32),
            'disparity': The calculated disparity map (float32, pixels),
            'mask': Boolean mask of valid pixels (True where data is reliable)
        }
    """
    status = 1
    
    if switch_order:
        img1_ = img1.copy()
        img1 = img2
        img2 = img1_
    
    if 'R' not in kwargs or 't' not in kwargs:
        return None

    R = kwargs['R']
    t_vec = kwargs['t']
    
    h, w = img1.shape[:2]
    img_size = (w, h)

    # ---------------------------------------------------------
    # STEP 1: Stereo Rectification (Geometric Alignment)
    # ---------------------------------------------------------
    # alpha=0 creates a tight crop of valid overlapping area, reducing black borders
    # alpha=0.5 is a good compromise if you need more field of view
    alpha_val = 0.0 
    
    R1, R2, P1_new, P2_new, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        cameraMatrix1=camera_matrix,
        distCoeffs1=None,
        cameraMatrix2=camera_matrix,
        distCoeffs2=None,
        imageSize=img_size,
        R=R,
        T=t_vec,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=alpha_val, 
        newImageSize=img_size
    )

    # Create Remapping Maps
    map1_1, map2_1 = cv2.initUndistortRectifyMap(camera_matrix, None, R1, P1_new, img_size, cv2.CV_32FC1)
    map1_2, map2_2 = cv2.initUndistortRectifyMap(camera_matrix, None, R2, P2_new, img_size, cv2.CV_32FC1)

    # Convert to float for processing
    img1_float = img1.astype(np.float32)
    img2_float = img2.astype(np.float32)

    # Apply Rectification Warp
    img1_rect = cv2.remap(img1_float, map1_1, map2_1, cv2.INTER_CUBIC)
    img2_rect = cv2.remap(img2_float, map1_2, map2_2, cv2.INTER_CUBIC)

    # ---------------------------------------------------------
    # STEP 2: Prepare for SGM (NaN handling & Normalization)
    # ---------------------------------------------------------
    # SGM cannot handle NaNs. We replace them with 0 temporarily but keep a mask.
    # We also need 8-bit images for SGM.

    # Clean images for SGM input
    img1_clean = np.nan_to_num(img1_rect.copy(), nan=0.0).astype(np.float32)
    img2_clean = np.nan_to_num(img2_rect.copy(), nan=0.0).astype(np.float32)
    
    min1 = np.nanmin(img1_clean)
    scale1 = np.nanmax(img1_clean) - np.nanmin(img1_clean)
    img1_uint8 = ((img1_clean - min1) / scale1 * 255).astype(np.uint8)

    min2 = np.nanmin(img2_clean)
    scale2 = np.nanmax(img2_clean) - np.nanmin(img2_clean)
    img2_uint8 = ((img2_clean - min2) / scale2 * 255).astype(np.uint8)
    
    # ---------------------------------------------------------
    # STEP 3: Dense Matching (Semi-Global Matching)
    # ---------------------------------------------------------

    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(6, 6)) # Etwas aggressiver (3.0)
    img1_enhanced = clahe.apply(img1_uint8)
    img2_enhanced = clahe.apply(img2_uint8)

    # 2. Parameter-Tuning
    min_disparity = 0
    num_disparities = 5 #16 * 4  # 64 reicht für kleinen Versatz völlig aus (spart Rechenzeit)
    block_size = 1             # KLEINSTMOGLICHER WERT (3). Wichtig für feine Strukturen!
                              # 5 ist oft schon zu grob für sehr feine Details.
    
    continue_ = True
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,       
        P2=32 * 3 * block_size**2,      
        disp12MaxDiff=1,       # Bleibt 1 (Konsistenz-Check)
        uniquenessRatio=0,     # AUF 0 SETZEN! 
        speckleWindowSize=0,   # AUSSCHALTEN zum Testen.
        speckleRange=20,       # Egal wenn WindowSize=0, aber sicherheitshalber niedrig.
        preFilterCap=16,      # Standard
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
     
    disparity_raw = stereo.compute(np.ascontiguousarray(img1_enhanced), 
                                   np.ascontiguousarray(img2_enhanced)).astype(np.float32)
    disparity = disparity_raw / 16.0  
    sgm_valid = np.ones_like(disparity_raw).astype(np.bool)  
    sgm_valid = disparity >= min_disparity

    print(disparity, np.unique(disparity))
    if len(np.unique(disparity)) > 8:
        continue_ = False
    
    h, w = disparity.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    map_x = x_coords.astype(np.float32) - disparity
    map_y = y_coords.astype(np.float32) #- disparity
    
    # Apply the dense shift to img2_rect
    img2_aligned = cv2.remap(
        img2_rect,
        map_x, map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan
    )
    img1_aligned = img1_rect

    # ---------------------------------------------------------
    # STEP 5: Separate Interpolation & Masking
    # ---------------------------------------------------------
    
    # Apply SGM mask to Disparity immediately
    disparity[~sgm_valid] = np.nan
    img1_aligned[~sgm_valid] = np.nan
    img2_aligned[~sgm_valid] = np.nan

    if np.sum(np.isnan(img1_aligned)) < np.prod(img1_aligned.shape) * 0.75 \
             and np.sum(np.isnan(img2_aligned)) < np.prod(img2_aligned.shape) * 0.75:

        img1_aligned = interpolate_nan_pixels(img1_aligned)
        img2_aligned = interpolate_nan_pixels(img2_aligned)
        status = 0

    else:
        status = 1
    
    img1_aligned = mask_border(img1_aligned, 1 * (num_disparities - min_disparity))
    img2_aligned = mask_border(img2_aligned, 1 * (num_disparities - min_disparity))

    if switch_order:
        return img2_aligned, img1_aligned, status
    
    return img1_aligned, img2_aligned, status


def mask_border(arr, k):
    if k <= 0:
        return arr

    masked = arr.copy()
    masked[:k, :] = np.nan      # Top border
    masked[-k:, :] = np.nan     # Bottom border
    masked[:, :k] = np.nan      # Left border
    masked[:, -k:] = np.nan     # Right border
    return masked


def interpolate_nan_pixels(img):
    """
    Interpolates NaN values in an image using surrounding valid pixels.
    Works for both 2D (grayscale) and 3D (multi-channel) arrays.
    Uses a distance-transform based approach for speed and efficiency.
    """
    if img.ndim == 2:
        return _interpolate_single_channel(img)

    elif img.ndim == 3:
        # Process each channel separately
        for c in range(img.shape[2]):
            result[:, :, c] = _interpolate_single_channel(img[:, :, c])
        return result

    else:
        raise ValueError("Image must be 2D or 3D")


def _interpolate_single_channel(img, max_radius_pixels=2):
    """
    Interpoliert NaNs nur, wenn der nächste valide Pixel innerhalb von max_radius_pixels liegt.
    """
    if not np.any(np.isnan(img)):
        return img

    # 1. Maske erstellen: True wo Daten FEHLEN
    nan_mask = np.isnan(img)
    valid_mask = ~nan_mask

    # 2. Distanz-Transform berechnen
    # Gibt für jedes NaN-Pixel die Distanz zum nächsten validen Pixel zurück
    # distance_transform_edt arbeitet in Pixel-Einheiten
    distances = ndimage.distance_transform_edt(nan_mask)

    # 3. Maske für "füllbare" Löcher erstellen
    # Nur NaNs füllen, die nah genug an validen Daten sind
    fillable_mask = nan_mask & (distances <= max_radius_pixels)

    # Wenn keine Löcher den Kriterien entsprechen, zurückgeben
    if not np.any(fillable_mask):
        return img

    # 4. Interpolation NUR für die füllbaren Löcher
    # Wir interpolieren das ganze Bild wie vorher, aber maskieren das Ergebnis danach

    # Koordinaten vorbereiten
    y, x = np.mgrid[0:img.shape[0], 0:img.shape[1]]

    valid_points = np.column_stack((x[valid_mask], y[valid_mask]))
    valid_values = img[valid_mask]

    # Punkte die wir füllen WOLLEN
    fill_points = np.column_stack((x[fillable_mask], y[fillable_mask]))

    if len(fill_points) == 0:
        return img

    # Griddata ausführen
    # Hinweis: linear scheitert immer noch wenn die Punkte konvex nicht verbunden sind,
    # aber durch den Distanz-Filter haben wir sichergestellt, dass Nachbarn DA sind.
    # Falls griddata trotzdem NaN zurückgibt (wegen Triangulierungs-Problemen bei dünnen Strukturen),
    # fallen wir auf 'nearest' zurück oder nutzen die Distanz-gewichtete Methode (siehe unten).

    try:
        interpolated = griddata(
            valid_points,
            valid_values,
            fill_points,
            method='linear'
        )
    except Exception:
        # Fallback falls griddata komplett scheitert
        interpolated = griddata(
            valid_points,
            valid_values,
            fill_points,
            method='nearest'
        )

    # Ergebnis in das Bild zurückschreiben
    result = img.copy()

    # WICHTIG: griddata gibt manchmal NaN zurück, wenn die Triangulierung nicht klappt,
    # selbst wenn Distanz klein ist (z.B. alle Nachbarn auf einer Linie).
    # Wir müssen prüfen, ob interpolated Werte hat.

    # Sicherer Weg: Nur dort schreiben, wo griddata ein Ergebnis lieferte
    valid_interpolation = ~np.isnan(interpolated)

    # Indizes im flachen Array berechnen
    flat_indices = np.ravel_multi_index((y[fillable_mask][valid_interpolation], x[fillable_mask][valid_interpolation]), img.shape)

    # Direkt ins flache Ergebnis schreiben ist oft schneller, aber hier zur Lesbarkeit:
    # Wir müssen die Masken korrekt zuordnen
    # Einfacher: Wir erstellen ein leeres Array und füllen es
    temp_fill = np.full_like(fillable_mask, np.nan, dtype=float)
    temp_fill[fillable_mask] = interpolated

    # Nur die erfolgreichen Interpolationen übernehmen
    success_mask = fillable_mask & ~np.isnan(temp_fill)
    result[success_mask] = temp_fill[success_mask]

    return result

