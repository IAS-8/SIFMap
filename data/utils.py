# SPDX-FileCopyrightText: © 2025 Forschungszentrum Jülich GmbH
# SPDX-FileContributor: Armagan Elibol
# SPDX-FileContributor: Christian Hirt
# SPDX-FileContributor: Jim Buffat <j.buffat@fz-juelich.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import inspect
import cv2
import numba
from joblib import Parallel, delayed
import itertools
import time
import pickle as pkl
import hashlib

from numba import jit
from numba.typed import List
import numpy as np

import rasterio as rio
from scipy import sparse
from scipy.optimize import curve_fit

from sklearn.base import BaseEstimator, TransformerMixin


import functools
import os
import threading
import time

import psutil


_PROCESS = psutil.Process(os.getpid())


def _total_rss():
    """Return RSS (bytes) of this process and all living descendants."""
    total = 0

    try:
        total += _PROCESS.memory_info().rss
    except psutil.NoSuchProcess:
        return 0

    for child in _PROCESS.children(recursive=True):
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return total


def profile(interval=0.01):
    """
    Decorator reporting execution time and peak RSS of the entire process tree.

    Parameters
    ----------
    interval : float
        Sampling interval in seconds.
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            baseline = _total_rss()
            peak = baseline

            stop = threading.Event()

            def monitor():
                nonlocal peak
                while not stop.is_set():
                    peak = max(peak, _total_rss())
                    stop.wait(interval)

            monitor_thread = threading.Thread(
                target=monitor,
                daemon=True,
            )

            start = time.perf_counter()
            monitor_thread.start()

            try:
                return func(*args, **kwargs)

            finally:
                stop.set()
                monitor_thread.join()

                elapsed = time.perf_counter() - start

                print('\n*** PROFILING RESULTS FOR '
                    f"{func.__qualname__}: "
                    f"{elapsed:.2f} s | "
                    f"Start: {baseline/1024**2:.1f} MB | "
                    f"Peak: {peak/1024**2:.1f} MB | "
                    f"Peak increase: {(peak-baseline)/1024**2:.1f} MB ***\n"
                )

        return wrapper

    return decorator


def timeit(func):
    """
    Decorator for measuring function's running time.
    """
    def measure_time(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        print("Processing time of %s(): %.2f seconds."
              % (func.__qualname__, time.time() - start_time))
        return result

    return measure_time


def run_jobs(jobs, do_parallel=True, n_jobs=4, n_chunks=1, chunk_callback=None, parallel_context=None,
             *args, **kwargs):
    """
    Manage a parallelzed job execution.
    """
    def _append(obj, out, chunk):
        if chunk_callback is not None:
            ret = chunk_callback(obj, args=[job[0].args for job in chunk],
                                 kwargs=[job[0].keywords for job in chunk])
            out.append((obj, ret))

        else:
            out.append(obj)

    def _run_parallel(chunks, parallel_context, n_jobs, *args, **kwargs):
        if parallel_context is None:
            out = []
            with init_with_valid_kwargs(Parallel, n_jobs=n_jobs, *args, **kwargs) as parallel_context:
                for chunk in chunks:
                    chunk_out = parallel_context(chunk)

                    _append(chunk_out, out, chunk)
            return out

        else:
            out = []
            for chunk in chunks:
                chunk_out = parallel_context(chunk)

                _append(chunk_out, out, chunk)
            return out

    def _run_single(chunks):
        out = []

        for j, chunk in enumerate(chunks):
            chunk_out = []

            for i, job in enumerate(chunk):
                chunk_out.append(job())

            _append(chunk_out, out, chunk)

        return out

    if len(jobs) == 0:
        return None

    chunk_size = max(1, len(jobs) // n_chunks)

    if do_parallel:
        jobs = [delayed(job)() for job in jobs]
        chunks = [jobs[i:i + chunk_size] for i in range(0, len(jobs), chunk_size)]
        out = _run_parallel(chunks, parallel_context=parallel_context, n_jobs=n_jobs, *args, **kwargs)

    else:
        chunks = [jobs[i:i + chunk_size] for i in range(0, len(jobs), chunk_size)]
        out = _run_single(chunks)

    return list(itertools.chain(*out))


def chunk_list(it, size):
    it = iter(it)
    return list(iter(lambda: tuple(itertools.islice(it, size)), ()))


def init_with_valid_kwargs(cls, *args, **kwargs):
    valid_params = list(inspect.signature(cls.__init__).parameters)
    valid_kwargs = dict([(k, v) for k, v in kwargs.items() if k in valid_params])

    return cls(*args, **valid_kwargs)


def convert_to_numba_list_of_lists(jagged_array, default_value=None):
    """
    Converts a jagged 2D array (where each element is either a numpy array of float64 or None)
    into a Numba typed List of Lists. Optionally, None values can be replaced with a default value.

    Parameters:
    - jagged_array: 2D list/array where elements are either float64 arrays or None
    - default_value: Value to replace None entries with (default is None)

    Returns:
    - A Numba typed List of Lists
    """
    result = List()
    default_value = np.array([[-100.0]])

    for row in jagged_array:
        typed_row = List()

        for elem in row:
            if elem is None:
                # If element is None, replace with default_value (or skip it)
                typed_row.append(default_value)
            else:
                # If element is a numpy array, append it directly
                typed_row.append(elem)

        # Append the typed_row (List) to the result (List of Lists)
        result.append(typed_row)

    return result


def get_flann(flann):
    if type(flann) is str:
        flann = getattr(cv2, flann)()

    elif type(flann) is dict:
        if not 'args' in flann:
            flann['args'] = []

        if not 'kwargs' in flann:
            flann['kwargs'] = {}

        flann = init_with_valid_kwargs(getattr(cv2, flann['type']), *flann['args'], **flann['kwargs'])

    return flann


@jit(nopython=True)
def numba_vstack(arrays):
    """
    Numba-compatible version of numpy.vstack that vertically stacks a list of numpy arrays.

    Parameters:
    - arrays: A list of numpy arrays with matching columns.

    Returns:
    - A vertically stacked numpy array.
    """
    # Get the number of rows in the result (sum of rows in each input array)
    num_cols = arrays[0].shape[1]  # Assume all arrays have the same number of columns
    total_rows = sum(arr.shape[0] for arr in arrays)

    # Initialize an empty numpy array for the stacked result
    stacked = np.empty((total_rows, num_cols), dtype=arrays[0].dtype)

    # Copy the rows into the stacked array
    current_row = 0
    for arr in arrays:
        stacked[current_row:current_row + arr.shape[0], :] = arr
        current_row += arr.shape[0]

    return stacked


def get_from_dict(dic, keys, need_all=False):
    ret = dict()

    for key in keys:
        if need_all:
            ret[key] = dic.get(key, None)
        
        elif not need_all and key in dic:
            ret[key] = dic[key]

        else:
            pass

    return ret


def replace_ext(path, ext='tif'):
    parts = path.split('.')
    parts[-1] = ext
    return '.'.join(parts)


def estimate_array_size(shape, dtype):
    """
    Estimate the size of a NumPy array in RAM without creating it.

    """
    # Calculate the total number of elements
    num_elements = np.prod(shape)
    
    # Get the size of one element in bytes
    bytes_per_element = np.dtype(dtype).itemsize
    
    # Calculate total size in bytes
    size_in_bytes = num_elements * bytes_per_element
    
    # Convert to other units
    size_in_kb = size_in_bytes / 1024
    size_in_mb = size_in_kb / 1024
    size_in_gb = size_in_mb / 1024
    
    return {
        "bytes": size_in_bytes,
        "kilobytes": size_in_kb,
        "megabytes": size_in_mb,
        "gigabytes": size_in_gb
    }


def get_checksum(obj, algorithm='sha256'):
    # Serialize the object into a byte format using pickle
    serialized_obj = pkl.dumps(obj)

    # Choose the hashing algorithm (default is sha256)
    hash_function = getattr(hashlib, algorithm)()

    # Update the hash function with the serialized object data
    hash_function.update(serialized_obj)

    # Return the hex digest of the checksum
    return hash_function.hexdigest()


def write_tif(out_path, arr, dtype=rio.uint8, band_names=None, **kwargs):
    if len(arr.shape) == 2:
        arr = arr[None]

    profile = dict(transform=None)
    profile.update({
        'driver': 'GTiff',
        'dtype': dtype,
        'height': arr.shape[1],
        'width': arr.shape[2],
        'count': arr.shape[0],
        'interleave': 'band'})
    profile.update(kwargs)

    with rio.open(out_path, 'w', **profile) as dst:
        for i, band in enumerate(arr, start=1):
            dst.write(band, i)

            if band_names is not None:
                dst.set_band_description(i, band_names[i-1])


def safe_cast(arr, dtype, nan_value=-9999):
    arr_ = arr.astype(dtype)
    arr_[np.isnan(arr)] = nan_value
    return arr_


def print_text_histogram(data, bins=10, char="█"):
    # Adjust bin edges to be integers
    min_edge = np.floor(min(data))
    max_edge = np.ceil(max(data))
    bin_edges = np.linspace(min_edge, max_edge, bins + 1, dtype=int)  # Ensure integer edges

    hist, bin_edges = np.histogram(data, bins=bin_edges)
    max_count = max(hist) if max(hist) > 0 else 1  # Avoid division by zero

    # Format the bin edges for consistent width
    edge_width = max(len(f"{bin_edges[i]:d}") for i in range(len(bin_edges))) + 1

    # Print histogram
    for i in range(len(hist)):
        left_edge = f"{bin_edges[i]:{edge_width}d}"
        right_edge = f"{bin_edges[i + 1]:{edge_width}d}"
        bar = char * int((hist[i] / max_count) * 50)  # Scale bars to max width
        print(f"{left_edge} - {right_edge} | {bar} ({hist[i]})")  # Include count


@jit(nopython=True, cache=True, parallel=False)
def clip_coordinates_numba(arr, min_, max_):
    """
    Numba compatible version of numpy.clip
    Args:
        arr:
        min_:
        max_:

    Returns:

    """
    for i in range(arr.shape[0]):
        arr[i] = max(min_[i], min(arr[i], max_[i]))

    return arr


def is_sparse(matrix):
    return isinstance(matrix, sparse.csr_matrix)


def densify(sparse, fill_value=0):
    val = sparse.toarray()

    if fill_value != 0:
        mask = np.ones_like(val).astype(int)
        mask[np.unravel_index(sparse.indices, val.shape)] = 0
        val[mask] = fill_value

    return val.T


def in_polygon(xq, yq, xv, yv):
    # Dummy function, replace with an actual implementation
    poly_path = Path(np.column_stack([xv, yv]))
    inside = poly_path.contains_points(np.column_stack([xq, yq]))
    return inside, np.zeros_like(inside)


@jit(nopython=True, cache=True, parallel=False)
def _point_in_polygon(x, y, poly):
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x, p1y = poly[0]

    for i in range(n + 1):
        p2x, p2y = poly[i % n]

        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


@jit(nopython=True, cache=True, parallel=True)
def numba_in_polygon(points, polygon):
    # Ensure the result array is of the correct type (boolean)
    D = np.empty(len(points), dtype=np.uint8)

    # Loop over the points with prange for parallelization
    for i in numba.prange(len(D)):
        D[i] = _point_in_polygon(points[i, 0], points[i, 1], polygon)

    return D




@jit(nopython=True, cache=True, parallel=False)
def bilinear_interpolation(img, x, y):
    """Fast bilinear interpolation using Numba."""
    h, w = img.shape[:2]

    # Get integer coordinates (top-left corner)
    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)

    # Compute the differences
    dx, dy = x - x0, y - y0

    # Extract four neighboring pixels
    Q11 = img[y0, x0]
    Q21 = img[y0, x1]
    Q12 = img[y1, x0]
    Q22 = img[y1, x1]

    # Bilinear interpolation formula
    value = (Q11 * (1 - dx) * (1 - dy) +
             Q21 * dx * (1 - dy) +
             Q12 * (1 - dx) * dy +
             Q22 * dx * dy)

    return value


@jit(nopython=True, cache=True, parallel=False)
def cubic_kernel(t):
    """Cubic interpolation kernel (Catmull-Rom spline)."""
    t = abs(t)
    if t <= 1:
        return (1.5 * t**3 - 2.5 * t**2 + 1)
    elif t <= 2:
        return (-0.5 * t**3 + 2.5 * t**2 - 4 * t + 2)
    return 0


@jit(nopython=True, cache=True, parallel=False)
def bicubic_interpolation(img, x, y):
    """Fast bicubic interpolation using Numba."""
    h, w = img.shape[:2]

    x0, y0 = int(x), int(y)
    value = 0.0

    for m in range(-1, 3):  # Iterate over 4x4 neighborhood
        for n in range(-1, 3):
            xn, yn = min(max(x0 + m, 0), w - 1), min(max(y0 + n, 0), h - 1)
            weight = cubic_kernel(x - (x0 + m)) * cubic_kernel(y - (y0 + n))
            value += img[yn, xn] * weight

    return value


@jit(nopython=True, cache=True, parallel=False)
def nearest_neighbor_interpolation(img, x, y):
    """Fast nearest-neighbor interpolation using Numba."""
    h, w = img.shape[:2]

    # Round to the nearest integer coordinate
    x_nn = min(max(int(round(x)), 0), w - 1)
    y_nn = min(max(int(round(y)), 0), h - 1)

    return img[y_nn, x_nn]


def extract_patches(images, p):
    if not images:
        return []

    L, W = images[0].shape[:2]
    
    all_patches = [[] for _ in images]

    for i in range(0, L, p):
        for j in range(0, W, p):
            for idx, img in enumerate(images):
                if img is None:
                    all_patches[idx].append(None)
                else:
                    all_patches[idx].append(img[i:i + p, j:j + p])
    
    return all_patches


def reconstruct_from_patches(patches, img_shape, p):
    L, W = img_shape
    img = np.zeros((L, W), dtype=patches[0].dtype)

    index = 0
    for i in range(0, L, p):
        for j in range(0, W, p):
            patch = patches[index]
            img[i:min(i+patch.shape[0], L), j:min(j+patch.shape[1], W)] = patch
            index += 1

    return img


def modify_attribute(tree, name, attribute_name, new_value):
    for elem in tree.iter(name):
        elem.set(attribute_name, new_value)

    return tree


def cast_to_int_type(value: float):
    #if np.all(np.logical_and(np.iinfo(np.int8).min <= value, value <= np.iinfo(np.int8).max)):
    #    return np.int8(value)
    if np.all(np.logical_and(np.iinfo(np.int16).min <= value, value <= np.iinfo(np.int16).max)):
        return np.int16(value)
    elif np.all(np.logical_and(np.iinfo(np.int32).min <= value, value <= np.iinfo(np.int32).max)):
        return np.int32(value)
    elif np.all(np.logical_and(np.iinfo(np.int64).min <= value, value <= np.iinfo(np.int64).max)):
        return np.int64(value)
    else:
        raise ValueError("Value is too large to fit in an integer type")
