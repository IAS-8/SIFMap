# SPDX-FileCopyrightText: © 2025 Forschungszentrum Jülich GmbH
# SPDX-FileContributor: Armagan Elibol
# SPDX-FileContributor: Christian Hirt
# SPDX-FileContributor: Jim Buffat <j.buffat@fz-juelich.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

from data.data import Datastruct
from data.geometry import quat2rotmat
from data.geometry import get_pose_from_absolute, convert_bundle_to_gmml, convert_bundle_to_data, set_resolution

import numba
from numba import jit

import scipy
from scipy.spatial import ConvexHull

import shapely
from shapely.geometry import MultiPoint


def vector_to_affine_homography(x):
    """
    Reshape x vector to matrix (3x3 affine homography)

    Args:
        x (): vector composed of motion parameters
        n_images (): total number of images

    Returns:
        Hini: 2D image-to-map transformations
    """
    n_params = 6
    n_images = len(x) // n_params

    Hini = []
    for i in range(n_images):
        Hini.append(np.array([[x[i*n_params+0], x[i*n_params+1], x[i*n_params+2]],
                              [x[i*n_params+3], x[i*n_params+4], x[i*n_params+5]],
                              [0, 0, 1]]))

    return np.asarray(Hini)


def convert_image_to_map_to_map_to_image_2d(data, x):
    H = vector_to_affine_homography(x)
    GlobalH = []   #[np.identity(3) for _ in range(len(data))]
    for i in range(len(data)):
        HH = np.linalg.inv(H[i][:])
        GlobalH.append(HH / HH[2, 2])

    return np.asarray(GlobalH)


def convert_image_to_map_to_map_to_image_3d(data, x, wHi, mosaic_resolution, mosaic_origin, rotate_to_minimum_area=True):
    H, FinalPose = convert_bundle_to_gmml(wHi, x, mosaic_origin, mosaic_resolution)
    H = set_resolution(H, mosaic_resolution)
    
    if rotate_to_minimum_area:
        H = optimize_global_rotation(H, data)

    GlobalH = [np.identity(3) for _ in range(len(data))]
    for i in range(len(data)):
        HH = np.linalg.inv(H[i][:])
        GlobalH[i][:] = HH / HH[2, 2]

    return np.asarray(GlobalH)


def optimize_global_rotation(GlobalH, data):
    """
    Rotate image->map homographies so the mosaic has the minimum-area
    enclosing rectangle.

    Parameters
    ----------
    GlobalH : (N,3,3) ndarray
        Image -> map homographies.
    data :

    Returns
    -------
    HGlobal_new : (N,3,3) ndarray
        Rotated/transformed image -> map homographies.
    theta : float
        Rotation angle in radians.
    canvas_size : (width, height)
    """
    # Map-space corners
    pts = []
    for i, H in enumerate(GlobalH):
        h, w = data.shapes[i]

        corners = np.array([
            [0, 0, 1],
            [w, 0, 1],
            [w, h, 1],
            [0, h, 1]
        ]).T

        p = H @ corners
        p /= p[2]
        pts.append(p[:2].T)

    pts = np.vstack(pts)

    # Convex hull
    hull = ConvexHull(pts)
    hull_pts = pts[hull.vertices]

    # Minimum rotated rectangle
    rect = MultiPoint(hull_pts).minimum_rotated_rectangle
    rect = np.asarray(rect.exterior.coords[:-1])

    edge = rect[1] - rect[0]
    theta = np.arctan2(edge[1], edge[0])

    # Rotate the rectangle back to horizontal
    c = np.cos(-theta)
    s = np.sin(-theta)

    R = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

    # Rotate all points to determine bounding box
    pts_h = np.c_[pts, np.ones(len(pts))].T
    pts_rot = (R @ pts_h)[:2].T

    xmin, ymin = pts_rot.min(axis=0)

    # Translate so top-left is (0,0)
    T = np.array([
        [1, 0, -xmin],
        [0, 1, -ymin],
        [0, 0, 1]
    ])

    A = T @ R

    # Update homographies (image -> new map)
    HGlobal_new = A @ GlobalH
    return HGlobal_new


def get_non_connected_close(data, t=1):
    """
    Get the image ids of images that connect to an image without neighbours, where neighbour defines an image with an
    index that has a distance of at most t.
    Args:
        data:
        t:

    Returns:
        [(id, n_correspondences)....]: list of tuples
    """
    idnum_mapping = dict([(data.data.data[k]['idnum'], k) for k in range(len(data.match_matrix))])

    non_connected_close = []
    for idnum in idnum_mapping.keys():
        range_ = np.arange(idnum - t, idnum + t + 1)
        close_connects = []
        for other_idnum in range_:
            if other_idnum == idnum or other_idnum not in idnum_mapping:
                continue

            ms = data.match_matrix[idnum_mapping[idnum], idnum_mapping[other_idnum]]
            close_connects.append((other_idnum, 0 if ms is None else len(ms)))

        if len(close_connects) == 0:
            non_connected_close.append((idnum, None))

        elif np.logical_not(np.array(close_connects)[:, 1]).any():
            non_connected_close.append((idnum, close_connects))

    return non_connected_close


def get_mosaic_size(IniH, data, MosaicOrigin, MosaicResolution):
    """

    Args:
        IniH (): image-to-map 2D planar transformations
        ImageSize (): Image resolution, e.g., 
            ImageSize = {'Width':1024,'Height':1024,'Depth':3,'Bits':8}
        MosaicOrigin (): amount of translation in order to have all images 
        in the positive area of the coordinate frame
        MosaicResolution (): 1 pixel in mm
        nbrIm ():

    Returns:
        MosaicSize: Size of the mosaic image
        MosaicOrigin: computed amount of translation in order to have all images 
        in the positive area of the coordinate frame 
        H: transformation to be applied all images to move positive 
        IniH: image-to-map 2D planar tranformations
    """

    if len(data) > 0:
        minx = np.inf
        miny = np.inf
        maxx = -np.inf
        maxy = -np.inf
        for no in range(len(data)):

            if type(data) is list:
                height, width = data[no].shape

            elif type(data) is Datastruct:
                height, width = data.shapes[no]

            else:
                raise Exception(f'Invalid data type {type(data)} of arg "data".')

            IS = dict(width=width, height=height)
            Corners = {1: np.array([[1, 1, IS["width"], IS["width"]],
                                    [1, IS["height"], IS["height"], 1],
                                    [1, 1, 1, 1]])}
            #if no==0:
            #    cc = np.dot(np.eye(3),Corners[1])
            #else:
            #   cc = np.dot(np.linalg.inv(IniH[no][:]), Corners[1])
            cc = np.dot(np.linalg.inv(IniH[no][:]), Corners[1])
            cc[0, :] = cc[0, :] / cc[2, :]
            cc[1, :] = cc[1, :] / cc[2, :]
            minx = min([minx, np.min(cc[0, :])])
            maxx = max([maxx, np.max(cc[0, :])])
            miny = min([miny, np.min(cc[1, :])])
            maxy = max([maxy, np.max(cc[1, :])])

        H = np.array([[1, 0, -minx + 1],
                      [0, 1, -miny + 1],
                      [0, 0, 1]])

        HH1 = np.linalg.inv(H)
        for no in range(len(data)):
            tff = (IniH[no][:]) @ HH1
            IniH[no][:] = tff
            IniH[no][:] = IniH[no] / IniH[no][2, 2]


        MosaicOrigin["X"] = MosaicOrigin["X"] - (-minx + 1) * MosaicResolution
        MosaicOrigin["Y"] = MosaicOrigin["Y"] + (-miny + 1) * MosaicResolution
        MosaicSize = {"width": np.ceil(maxx - minx) + 2,  # +2 due to the 0 and the round
                      "height": np.ceil(maxy - miny) + 2}
    else:
        MosaicOrigin["X"] = 0
        MosaicOrigin["Y"] = 0
        MosaicSize = {"width": 0, "height": 0}

    #return MosaicSize, MosaicOrigin, IniH, H
    return MosaicSize, MosaicOrigin, H, IniH


@jit(nopython=True, cache=True, parallel=False)
def calculate_homography_3d(x, K):
    """
    From 3d pose to 2d homography.

    Args:
        x (): motion parameter of all images in a single vector format
        K (): camera instrinsics

    Returns:
        iHm : map-to-image transformations
        mHi : image-to-map transformations
    """
    n = len(x) // 6
    iHm = np.zeros((3, 3, n))
    mHi = np.zeros((3, 3, n))

    K_1 = np.zeros((3, 3))
    K_1[2, 2] = 1
    K_1[0, 0] = 1.0 / K[0, 0, 0]
    K_1[1, 1] = 1.0 / K[1, 1, 0]
    K_1[0, 2] = -K[0, 2, 0] / K[0, 0, 0]
    K_1[1, 2] = -K[1, 2, 0] / K[1, 1, 0]

    k = 0
    f = 1

    for i in range(n):
        vxi = x[k]
        vyi = x[k + 1]
        vzi = x[k + 2]
        wi = 1.0 - vxi * vxi - vyi * vyi - vzi * vzi
        if wi < 0:
            # wi = 0

            wi = np.cos(np.arcsin(np.sqrt(vxi * vxi + vyi * vyi + vzi * vzi) - 1.0))
            Norms = np.sqrt(vxi * vxi + vyi * vyi + vzi * vzi + wi * wi);
            wi = wi / Norms
            vxi = vxi / Norms
            vyi = vyi / Norms
            vzi = vzi / Norms
        else:
            wi = np.sqrt(wi)

        cRw = quat2rotmat([wi, vxi, vyi, vzi])

        X = x[k + 3]
        Y = x[k + 4]
        Z = x[k + 5]

        wITc = np.array([[1, 0, -X],
                         [0, 1, -Y],
                         [0, 0, -Z]])

        wITc_1 = np.array([[1, 0, -X / Z],
                           [0, 1, -Y / Z],
                           [0, 0, -1 / Z]])

        Kn = K[:, :, f - 1]

        iHm[:, :, i] = Kn @ cRw @ wITc  # np.dot(np.dot(Kn, cRw), wITc)
        mHi[:, :, i] = wITc_1 @ cRw.T @ K_1  # np.dot(np.dot(wITc_1, cRw.T), K_1)

        k += 6
        f += 1

    return iHm, mHi
