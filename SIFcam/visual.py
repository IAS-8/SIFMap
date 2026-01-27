# SPDX-FileCopyrightText: © 2025 Forschungszentrum Jülich GmbH
# SPDX-FileContributor: Armagan Elibol
# SPDX-FileContributor: Christian Hirt
# SPDX-FileContributor: Jim Buffat <j.buffat@fz-juelich.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import copy
import os.path
import csv
import psutil

import os, subprocess

import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from scipy.io import savemat

import run_utils
from align.calc import get_mosaic_size

from os.path import join as pjoin
from os.path import dirname as pdirname
from os.path import basename as pbasename

from SIFcam.data import SIFcam
from data.utils import timeit, run_jobs, chunk_list, estimate_array_size, write_tif, safe_cast, \
    is_sparse, densify, numba_in_polygon, bicubic_interpolation, bilinear_interpolation, nearest_neighbor_interpolation,\
    modify_attribute
from match.utils import interpolation_flags_CV2

from matplotlib.colors import hsv_to_rgb, ListedColormap
from functools import partial
import numpy as np

import scipy
from scipy import sparse

from numba import jit

import rasterio as rio
from rasterio.transform import from_origin
import cv2 


@timeit
def visualize(data, HGlobal, **kwargs):
    """

    Args:
        data (): data structure to keep correspondences data
            among overlapping image pairs
        saveData (): data structure to keep SIF data
        HGlobal (): map-to-image transformations
        nbrIm (): number of images
        mosaic_resolution (): pixel size in mm
        **kwargs ():

    Returns:

    """
    # load images and SIF data to memory
    SIF_list, Ref757_list, Ref760_list = load_data(data)
    products = dict(sif=SIF_list, ref757=Ref757_list, ref760=Ref760_list)
    
    #raise Exception()

    # visualize the final mosaic (map), using mean and pixel from closest to the image center
    aggregates, glob_trfm, HGlobal = aggregate_images(HGlobal, data, products, save_to_mat=True, **kwargs)

    #Create VRT files
    if False:
        out_dir = pjoin(data.data.result_path, 'SINGLE_ACQUISITIONS')
        os.makedirs(out_dir, exist_ok=True)
        for i, (h, info) in enumerate(zip(HGlobal, data.images)):
            shifted_transform = np.matmul(glob_trfm, np.linalg.inv(h))

            create_output_tifs(dict(Fluo=SIF_list[i].copy(), 
                                    Refl757=Ref757_list[i].copy(), 
                                    Refl760=Ref760_list[i].copy()),
                                projective_matrix=shifted_transform,
                                output=pjoin(out_dir, 'single_acq_' + info['id'])) 

    quick_look_dir = pjoin(data.data.result_path, 'quick_looks')
    os.makedirs(quick_look_dir, exist_ok=True)
    arr, band_names = plot_and_save(aggregates, quick_look_dir)

    results_tif = pjoin(data.data.result_path, 'results.tif')
    write_tif(results_tif,
              arr, band_names=band_names, dtype=rio.int16)

    #command = ['gdaladdo', '-r', 'cubic', results_tif]
    #subprocess.run(command,
    #               stdout=subprocess.DEVNULL,  # Suppress stdout
    #               stderr=subprocess.PIPE,     # Capture stderr
    #               check=True,
    #               cwd=pdirname(results_tif))

    scipy.io.savemat(pjoin(data.data.result_path, 'results.mat'), aggregates)

    mapping_data_path = pjoin(data.data.result_path, 'mapping_data')
    os.makedirs(mapping_data_path, exist_ok=True)
    np.save(pjoin(mapping_data_path, 'global_trfm.npy'), glob_trfm)


def plot_and_save(aggregates, result_path, tif_dtype=np.int16):
    csv_keys = ['path_ids']
    categorical_keys = ['closest_id']
    discrete_keys = ['n_masked', 'n_covering']

    arr = []
    band_names = []
    for key, val_sparse in aggregates.items():
        if is_sparse(val_sparse):
            val = densify(val_sparse, fill_value=np.nan)
        else:
            val = val_sparse

        if key in csv_keys:
            with open(pjoin(result_path, f'{key}.csv'), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows([[item] for item in val])

            continue

        elif np.any([c in key for c in categorical_keys]):
            arr.append(val.astype(tif_dtype))
            band_names.append(key)

            cmap = get_discrete_colormap(val)

            plt.imshow(val, cmap=cmap, interpolation='bilinear')
            plt.colorbar()

        elif np.any([c in key for c in discrete_keys]):
            arr.append(safe_cast(val, tif_dtype, -9999))
            band_names.append(key)

            vmin, vmax = None, None
            plt.imshow(val, cmap='viridis', interpolation='bilinear', vmin=vmin, vmax=vmax)
            plt.colorbar()

        else:
            if 'closest_dist' in key:
                arr.append(safe_cast(val * 10, tif_dtype, -9999))
                band_names.append(f'{key} * 10')

            else:
                arr.append(safe_cast(val * 1000, tif_dtype, -9999))
                band_names.append(f'{key} * 1000')

            vmin, vmax = np.nanquantile(val.flatten(), [0.01, 0.99])
            plt.imshow(val, cmap='viridis', interpolation='bilinear', vmin=vmin, vmax=vmax)
            plt.colorbar()

        plt.title(key)
        plt.savefig(fname=pjoin(result_path, f'{key}.png'), dpi=300)
        plt.close()

    return np.asarray(arr), band_names


@timeit
def aggregate_images(H, data, products, parallel_params=None, parallel_context=None, **kwargs):

    MosaicOrigin = {'X': 0, 'Y': 0}
    MosaicResolution = 1
    MosaicSize, MosaicOrigin, glob_trfm, H = get_mosaic_size(H, data, MosaicOrigin,
                                                             MosaicResolution)

    h = np.int64(MosaicSize['height'])
    w = np.int64(MosaicSize['width'])
    print(f'Preparing to plot a map of dimensions ({h}, {w})')

    ws, hs = np.meshgrid(np.arange(1, w + 1), np.arange(1, h + 1))
    coords = np.column_stack([ws.flatten(), hs.flatten()])

    H = np.asarray(H)
    is_covered = get_is_covered(H, coords, np.asarray(data.shapes))

    # Get aggregates
    # @TODO: assume height and width are the same for all images
    agg_px = _aggregate_iterate_px(H, is_covered, coords,
                                   list(products.values()),
                                   parallel_params=parallel_params,
                                   parallel_context=parallel_context,
                                   **kwargs)

    n_covering = np.sum(is_covered, axis=1)
    px_mask = np.where(n_covering > 0)[0]
    px_coords = (coords[px_mask, 0].astype(np.int32), coords[px_mask, 1].astype(np.int32))

    n_covering = sparse.csr_matrix((n_covering[px_mask], px_coords), shape=(w, h)).toarray().T
    closest_id = sparse.csr_matrix((agg_px['all'][:, 0, 8], px_coords), shape=(w, h)).toarray().T
    closest_dist = sparse.csr_matrix((np.sqrt(agg_px['all'][:, 0, 9]), px_coords), shape=(w, h)).toarray().T

    n_covering, closest_id, closest_dist = mask(np.asarray([n_covering, closest_id, closest_dist]),
                                                            (px_coords[1], px_coords[0]), fill_value=np.nan)

    result_maps = dict()
    for pid, name in enumerate(products.keys()):
        min_ = sparse.csr_matrix((agg_px['all'][:, pid, 0], px_coords), shape=(w, h)).toarray().T
        max_ = sparse.csr_matrix((agg_px['all'][:, pid, 1], px_coords), shape=(w, h)).toarray().T
        mean_ = sparse.csr_matrix((agg_px['all'][:, pid, 2], px_coords), shape=(w, h)).toarray().T
        #median_ = sparse.csr_matrix((agg_px['all'][:, pid, 3], px_coords), shape=(w, h)).toarray().T

        closest_ = sparse.csr_matrix((agg_px['all'][:, pid, 7], px_coords), shape=(w, h)).toarray().T
        #closest_masked = sparse.csr_matrix((agg_px['masked'][:, pid, 7], px_coords), shape=(w, h)).toarray().T

        #n_masked = sparse.csr_matrix((agg_px['n_masked'][:, pid], px_coords), shape=(w, h)).toarray().T

        # mask non_covered_pixels
        res = mask(#np.asarray([mean_, median_, max_, min_, closest_, closest_masked,n_masked]),
                   np.asarray([mean_, max_, min_, closest_]),
                   (px_coords[1], px_coords[0]),
                   fill_value=np.nan)
        #mean_, median_, max_, min_, closest_, closest_masked, n_masked = res
        mean_, max_, min_, closest_ = res

        result_maps.update({
                f'min_{name}': min_,
                f'max_{name}': max_,
                f'mean_{name}': mean_,
         #       f'median_{name}': median_,
                f'closest_{name}': closest_,
         #       f'closest_masked_{name}': closest_masked,
         #       f'n_masked_{name}': n_masked.astype(float),
              })

    result_maps.update({f'n_covering': n_covering.astype(float),
                        f'closest_id': closest_id.astype(float),
                        f'closest_dist' : closest_dist.astype(float)})

    result_maps.update(path_ids=[os.path.basename(p[0]) for p in data.paths])
    return result_maps, glob_trfm, H


def _aggregate_iterate_px(H, is_covered, coords, products, parallel_params=None, parallel_context=None,
                          interpolation_method='linear', **kwargs):
    if parallel_params is None or not 'n_jobs' in parallel_params or not parallel_params['do_parallel']:
        n_jobs = 1

    else:
        n_jobs = parallel_params['n_jobs']

    covered_px_inds = np.where(np.sum(is_covered, axis=1) > 0)[0]
    coords = coords[covered_px_inds].astype(float)
    jobs = []

    products = [np.asarray(product) for product in products]
    H = np.asarray(H)

    for i, ks in enumerate(chunk_list(range(len(covered_px_inds)), len(covered_px_inds) // n_jobs)):
        ks = np.asarray(ks)
        BigH, k_coords, k_is_covered, products_, included_images = _prep_px(ks, H, coords, covered_px_inds,
                                                                            is_covered, products)

        jobs.append(partial(aggregate_stats_per_px, ks, k_is_covered, BigH, k_coords, products_, included_images,
                            interpolation_method))

        # Estimate RAM footprint of job
        if i == 0:
            mem = estimate_array_size((len(ks) * len(products), 10), 'float64')['gigabytes'] * 3

    # Check max jobs fitting into mem
    max_njobs = max((psutil.virtual_memory().available * 0.9 / 1024 ** 3) // mem, 1)
    if n_jobs > max_njobs:
        print(f'WARNING: available memory would be breached with a memory footprint of {mem:.2f} gb per job. '
              f'Reducing n_jobs to {max_njobs}')
        _parallel_params = copy.deepcopy(parallel_params)
        _parallel_params['n_jobs'] = max_njobs
        _parallel_context = None

    else:
        _parallel_params = parallel_params
        _parallel_context = parallel_context

    parallel_out = run_jobs(jobs, **_parallel_params, parallel_context=_parallel_context)

    out = dict()
    keys = ['all', 'masked', 'n_masked']
    for i, key in enumerate(keys):
        out[key] = np.concatenate([p[i] for p in parallel_out], axis=0)

    out['covering_inds'] = covered_px_inds

    return out


@jit(nopython=True, cache=True, parallel=False)
def aggregate_stats_per_px(ks, px_covers, H, coords, products, image_ids, interpolation_method='linear'):
    n_vars = 10
    n_prods = len(products)
    stat_list_all = np.ones((len(ks), n_prods, n_vars)) * np.nan
    stat_list_masked = np.ones((len(ks), n_prods, n_vars)) * np.nan

    n_masked = np.empty((len(products), len(ks)), dtype=np.uint16)

    for i, (k, px_cover, coord) in enumerate(zip(ks, px_covers, coords)):
        coord = coord[:2]
        cover_image_inds = np.where(px_cover > 0)[0]
        cover_image_ids = image_ids[cover_image_inds]

        prod_vals = np.ones((len(products), len(cover_image_inds))) * np.nan

        dst_to_center = np.ones(len(cover_image_inds)) * np.inf

        tbr = H @ np.append(coord, 1)
        covered_image_coords = np.zeros((3, len(H) // 3))  # Initialize 3ximNbr matrix

        # Fill the rows using slicing
        covered_image_coords[0, :] = tbr[0::3]  # Get every 3rd element starting from index 0 (x coordinates)
        covered_image_coords[1, :] = tbr[1::3]  # Get every 3rd element starting from index 1 (y coordinates)
        covered_image_coords[2, :] = tbr[2::3]  # Get every 3rd element starting from index 2 (homogeneous coordinates)

        # Normalize the coordinates
        covered_image_coords[0, :] /= covered_image_coords[2, :]  # Normalize x by homogeneous coordinate
        covered_image_coords[1, :] /= covered_image_coords[2, :]  # Normalize y by homogeneous coordinate

        for nn, idx in enumerate(cover_image_inds):

            width, height = products[0][idx].shape
            centerX = width / 2
            centerY = height / 2

            _coords = covered_image_coords[:, idx]
            
            # fetch interpolated values
            for pid, product in enumerate(products):
                if interpolation_method == "linear":
                    prod_vals[pid, nn] = bilinear_interpolation(product[idx], _coords[0], _coords[1])

                elif interpolation_method == "cubic":
                    prod_vals[pid, nn] = bicubic_interpolation(product[idx], _coords[0], _coords[1])

                elif interpolation_method == "nearest":
                    prod_vals[pid, nn] = nearest_neighbor_interpolation(product[idx], _coords[0], _coords[1])

                else:
                    print(f'ERROR:Interpolation method {interpolation_method} is not known.')
                    return

            dst_to_center[nn] = (_coords[1] - centerX) ** 2 + (_coords[0] - centerY) ** 2

        dstThres = np.sqrt(2) * width / 2 / 3
        dstThresSQ = dstThres ** 2

        # Process the pixel values
        if prod_vals.shape[1] > 0:
            stat_list_all[i] = get_px_stats(prod_vals, cover_image_ids,
                                            dst_to_center, dstThresSQ)

            for pid in range(len(prod_vals)):
                prod_mask = prod_vals[pid] < 0
                n_masked[pid, i] = np.sum(prod_mask)
                masked_prod_val_pid = prod_vals[pid][~prod_mask]

                if n_masked[pid, i] > 0 and n_masked[pid, i] != prod_vals.shape[1]:
                    masked_stat_pid = get_px_stats(masked_prod_val_pid[np.newaxis, :], cover_image_ids,
                                                   dst_to_center[~prod_mask], dstThresSQ)

                    stat_list_masked[i, pid] = masked_stat_pid[0]

                elif n_masked[pid, i] == prod_vals.shape[1]:
                    stat_list_masked[i, pid] = np.ones(stat_list_all[i, pid].shape) * np.nan

                else:  # n_masked[i] == 0:
                    stat_list_masked[i, pid] = stat_list_all[i, pid]

    return stat_list_all, stat_list_masked, n_masked.transpose()


@jit(nopython=True, cache=True, parallel=False)
def get_px_stats(products, image_ids, dist_to_center, dstThresSQ):
    """
    Compute statistics over all observations from different images in a single pixel.

    Args:
        pix_sif_vals: (N, ) SIF observations from different images.
        pix_refl_vals:  (N, ) reflectance observations from different images.
        dist_to_center:
        dstThresSQ:

    Returns:
        (minimum SIF, maximum  SIF, mean  SIF, median  SIF, std  SIF, distance weighted mean  SIF,\
        mean SIF within dstThresSQ, closest SIF, closest reflectance)
    """
    stats = np.ones((len(products), 10)) * np.nan

    for pid, prod in enumerate(products):
        eps = np.finfo(np.float64).eps
        valid_dist_ids = np.where(np.logical_and(dist_to_center < (dstThresSQ + eps),
                                      ~np.isnan(dist_to_center)))[0]

        dist_to_center[np.where(np.isnan(prod))] = np.inf
        mi1 = np.argmin(dist_to_center)

        # Compute basic statistics
        stats[pid, 0:5] = [np.nanmin(prod), np.nanmax(prod),
                           np.nanmean(prod), np.nanmedian(prod),
                           np.nanstd(prod)]

        tot_weights = np.nansum(1.0 / dist_to_center)
        if tot_weights > 0:
            avg1 = np.nansum(prod * (1.0 / dist_to_center)) / np.nansum(1.0 / dist_to_center)

        else:
            avg1 = np.nan

        # clostest center
        avg3 = prod[mi1]

        if len(valid_dist_ids) > 0:
            avg2 = np.nanmean(prod[valid_dist_ids])
        else:
            avg2 = avg3

        stats[pid, 5:8] = [avg1, avg2, avg3]

        # meta
        is_valid = not np.isnan(avg3)
        stats[pid, 8:10] = [image_ids[mi1] if is_valid else np.nan, dist_to_center[mi1] if is_valid else np.nan]

    return stats


@jit(nopython=True, cache=True)
def mask(list_of_arrays, except_coords, fill_value):
    """
    Mask out
    Args:
        list_of_arrays:
        except_coords:
        fill_value:

    Returns:

    """
    mask = np.ones_like(list_of_arrays[0], dtype=np.int32)

    # Unpack except_coords into row and column indices
    row_indices, col_indices = except_coords

    # Set the mask at the specified coordinates to zero
    # mask[row_indices, col_indices] = 0  #np.zeros_like(row_indices, dtype=np.int32)
    for idx in range(len(row_indices)):
        mask[row_indices[idx], col_indices[idx]] = 0

    #for i in range(len(list_of_arrays)):
    #    list_of_arrays[i][mask == 1] = fill_value

    # Iterate through the list of arrays
    for i in range(len(list_of_arrays)):
        # Get the indices where the mask is zero
        for idx in range(mask.shape[0]):
            for jdx in range(mask.shape[1]):
                if mask[idx, jdx]:
                    list_of_arrays[i][idx, jdx] = fill_value

    return list_of_arrays



def images_to_common_shape(images):
    # Determine the maximum height and width
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)
    
    expanded_images = []
    
    for img in images:
        h, w = img.shape[:2]
        
        # Create a new array filled with np.nan
        expanded_img = np.full((max_height, max_width) + img.shape[2:], np.nan, dtype=img.dtype)
        
        # Copy the original image into the top-left corner
        expanded_img[:h, :w] = img
        
        expanded_images.append(expanded_img)
    
    return np.asarray(expanded_images)
 

def load_data(data):
    """
        This function loads image and SIF data to show them as map
    Args:
        data (): Data structure to keep correspondences data
        among successfully matched image pairs

    Returns:
        list of image and SIF data
    """
    SIF_list = []
    Ref757_list = []
    Ref760_list = []
    images = data.images
    for i, img in enumerate(images):
        #img_list.append(ski.color.rgb2gray((ski.io.imread(img['file_name']))))
        if type(img['Fluo']) is str:
            SIF_list.append(SIFcam._load(img['Fluo']).squeeze())

        else:
            SIF_list.append(img['Fluo'].squeeze())

        if type(img['Refl757']) is str:
            Ref757_list.append(SIFcam._load(img['Refl757']).squeeze())
        else:
            Ref757_list.append(img['Refl757'].squeeze())

        if type(img['Refl760']) is str:
            Ref760_list.append(SIFcam._load(img['Refl760']).squeeze())
        else:
            Ref760_list.append(img['Refl760'].squeeze())

    return (images_to_common_shape(SIF_list), 
		    images_to_common_shape(Ref757_list),
		    images_to_common_shape(Ref760_list))


def get_discrete_colormap(arr, max_colors=1000, hue_offset=0.05, seed=42):
    # Get unique values from the array
    # Get unique values from the array
    unique_values = np.unique(arr)
    n_unique = len(unique_values)

    # Determine the number of colors needed (limited by max_colors)
    n_colors = min(n_unique, max_colors)

    # Generate evenly spaced colors in HSV with offsets
    hues = np.linspace(hue_offset, 1 - hue_offset, n_colors, endpoint=False)

    saturation = 0.75  # High saturation for vivid colors
    value = 0.9  # High brightness for lighter colors

    # Create HSV colors
    hsv_colors = np.array([[h, saturation, value] for h in hues])

    # Convert HSV colors to RGB format
    rgb_colors = hsv_to_rgb(hsv_colors)

    # Set the random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Shuffle the colors to ensure that nearby values have distinctly different colors
    np.random.shuffle(rgb_colors)

    # Cycle the color palette if necessary
    if n_unique > max_colors:
        rgb_colors = np.tile(rgb_colors, (int(np.ceil(n_unique / max_colors)), 1))[:n_unique]

    # Return the colormap
    return ListedColormap(rgb_colors)


@jit(nopython=True, cache=True, parallel=False)
def _get_is_covered(H, coords, shape):
    height, width = shape
    height, width = np.float64(height), np.float64(width)
    Corners = np.array([
        [1.0, 1.0, width, width, 1.0],
        [1.0, height, height, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0]
    ])
    ProjectedCorners = np.dot(np.linalg.inv(H), Corners)
    ProjectedCorners[0, :] /= ProjectedCorners[2, :]
    ProjectedCorners[1, :] /= ProjectedCorners[2, :]
    in_ = numba_in_polygon(coords, ProjectedCorners[:2].transpose())

    return in_


@jit(nopython=True, cache=True, parallel=True)
def get_is_covered(H, coords, shapes):
    is_covered = np.empty((coords.shape[0], H.shape[0]), dtype=np.uint8)
    for i in range(H.shape[0]):
        is_covered[:, i] = _get_is_covered(H[i], coords, shapes[i])

    return is_covered


def _prep_px(ks, H, coords, covered_px_inds, is_covered, products):
    k_covered_px_inds = covered_px_inds[ks]

    # Get k pixels
    k_coords = coords[ks]
    k_is_covered = is_covered[k_covered_px_inds]

    # Exclude unused images in this job
    included_images = np.where(k_is_covered.sum(axis=0) > 0)[0]
    k_is_covered = k_is_covered[:, included_images]

    # Get H for the included images only
    BigH = np.zeros((len(included_images) * 3, 3))

    # BigH / HGlobal map to image
    for i, m in enumerate(included_images):
        tbr = H[m]
        BigH[i * 3: (i + 1) * 3, 0:3] = tbr / tbr[2, 2]
    
    products = [p[included_images] for p in products]

    return BigH, k_coords, k_is_covered, products, included_images


def create_output_tifs(inputs, projective_matrix, output, interpolation_method='linear'):
    """
    Creates a VRT file with three bands from three TIFF files, applying the same projective transformation.
    
    Parameters:
    - file1, file2, file3: paths to the three input TIFF files (each representing a channel)
    - projective_matrix: a 3x3 numpy array representing the projective transformation matrix
    - output_vrt: path to the output VRT file
    
    Example of projective_matrix: np.array([[a, b, c], [d, e, f], [g, h, i]])
    """
    image = np.stack(list(inputs.values()), axis=0) * 1000
    image[np.isnan(image)] = -9999

    write_tif(output + '.tif',
              image, band_names=list(inputs.keys()), dtype=rio.int16)
    
    height, width = image.shape[1], image.shape[2]
    
    #projective_matrix[2, 0] = projective_matrix[2, 1] = 0

    # Extract scaling factors from the projective matrix
    a, b, x_translation = projective_matrix[0]
    d, e, y_translation = projective_matrix[1]

    # The scaling factors in the x and y directions
    scale_x = np.sqrt(a**2 + b**2)
    scale_y = np.sqrt(d**2 + e**2)
    
    corners = np.array([[0, 0], 
                        [width-1, 0], 
                        [width-1, height-1], 
                        [0, height-1]], dtype=np.float32)
    
    # Apply the projective transform to the corners
    corners_transformed = cv2.perspectiveTransform(corners[None, :, :], projective_matrix).squeeze()
    x_min, y_min = corners_transformed[:, 0].min(), corners_transformed[:, 1].min()
    x_max, y_max = corners_transformed[:, 0].max(), corners_transformed[:, 1].max()
    
    # Calculate new width and height of the output image
    new_width = int(np.ceil(x_max - x_min))
    new_height = int(np.ceil(y_max - y_min))
    
    # If you want to keep the image centered, you need to adjust the translation
    # For that, shift the translation components of the projective matrix
    translate = np.eye(3)
    translate[0, 2] = -x_min
    translate[1, 2] = -y_min
    corrected_transform = np.matmul(translate, projective_matrix)

    # projective_matrix[2, 0] = projective_matrix[2, 1] = 0
    
    warped_image = np.zeros((image.shape[0], new_height, new_width), dtype=np.float32)
    for channel in range(image.shape[0]):
        warped_image[channel] = cv2.warpPerspective(image[channel].astype(np.float32), 
                                                    corrected_transform.astype(np.float32), 
                                                    (new_width, new_height), 
                                                    flags=interpolation_flags_CV2[interpolation_method],
                                                    borderMode=cv2.BORDER_CONSTANT, 
                                                    borderValue=np.iinfo(np.int16).min)

    
    transform = from_origin(x_min, y_min, 1, 1) 
    write_tif(pjoin(pdirname(output), 'reprojected_' + pbasename(output) + '.tif'),
              warped_image, band_names=list(inputs.keys()), dtype=rio.int16, transform=transform)


def create_output_tifs_with_gdal(*input_paths, projective_matrix, output, transform, to_tif=False):
    """
    Creates a VRT file with three bands from three TIFF files, applying the same projective transformation.
    
    Parameters:
    - file1, file2, file3: paths to the three input TIFF files (each representing a channel)
    - projective_matrix: a 3x3 numpy array representing the projective transformation matrix
    - output_vrt: path to the output VRT file
    
    Example of projective_matrix: np.array([[a, b, c], [d, e, f], [g, h, i]])
    """
    # Ensure the projective matrix is a 3x3 numpy array
    projective_matrix = np.array(projective_matrix)
    
    input_paths = list(input_paths)
    rel_input_paths = [os.path.relpath(p, pdirname(output)) for p in input_paths]
    
    #_tmp_vrt = pjoin(pdirname(output_vrt), f'_combined_{pbasename(output_vrt)}')[:-4] + '.tif'
    vrt_command = [
            'gdalbuildvrt',
            '-separate', 
            output + '.vrt'
    ] + rel_input_paths
    
    run_utils.run(vrt_command,
                  stdout=subprocess.DEVNULL,  # Suppress stdout
                  stderr=subprocess.PIPE,  # Capture stderr
                  check=True,
                  cwd=pdirname(output))

    tree = ET.parse(output + '.vrt')
    tree = modify_attribute(tree, 'SourceFilename', 'relativeToVRT', '1')
    tree.write(output + '.vrt')

    if to_tif:
        vrt_command = [
                'gdal_translate',
                output + '.vrt',
                output + '.tif'
            ]

        run_utils.run(vrt_command,
                      stdout=subprocess.DEVNULL,  # Suppress stdout
                      stderr=subprocess.PIPE,  # Capture stderr
                      check=True,
                      cwd=pdirname(output))

        os.remove(output + '.vrt')

    with rasterio.open(output + '.vrt') as src:
        image = src.read()         
        height, width = image.shape[1], image.shape[2]
         
        warped_image = cv2.warpPerspective(image, transform, (width, height), 
                                           flags=interpolation_flags_CV2[rescale_interpolation_method],
                                           borderMode=cv2.BORDER_REPLICATE)
        
        # Now, save the transformed image
        with rasterio.open(pjoin(pdirname(output), 'reprojected_' + pbasename(output) + '.tif'), 'w', 
                           driver='GTiff', count=warped_image.shape[0],
                           dtype=warped_image.dtype, width=width, height=height) as dst:
            dst.write(warped_image)
 
