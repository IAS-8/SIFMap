# SPDX-FileCopyrightText: © 2025 Forschungszentrum Jülich GmbH
# SPDX-FileContributor: Armagan Elibol
# SPDX-FileContributor: Christian Hirt
# SPDX-FileContributor: Jim Buffat <j.buffat@fz-juelich.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

from data.data import Datastruct
from data.geometry import quat2rotmat


def vector_to_affine_homography(x):
    """
    Reshape x vector to matrix (3x3 affine homography)

    Args:
        x (): vector composed of motion parameters
        n_images (): total number of images

    Returns:
        Hini: 2D image-to-map transformations
    """
    NbrUnk = 6
    n_images = len(x) // NbrUnk

    Hini = [np.eye(3) for _ in range(n_images)]

    for i in range(n_images):
        Hini[i] = np.array([[x[(i)*NbrUnk], x[(i)*NbrUnk+1], x[(i)*NbrUnk+2]],
                            [x[(i)*NbrUnk+3], x[(i)*NbrUnk+4], x[(i)*NbrUnk+5]],
                            [0, 0, 1]])

    return Hini

def calc_residual_offsets(PointMatchesSize, PointMatchesIdxSize):
    """

    Args:
        PointMatchesSize ():
        PointMatchesIdxSize ():

    Returns:

    """
    if (PointMatchesSize is not None and len(PointMatchesSize) != 2) or \
       (PointMatchesIdxSize is not None and len(PointMatchesIdxSize) != 2):
        raise ValueError('Length of input vectors must be 2!')

    NumPointMatches = PointMatchesSize[0] * PointMatchesSize[1]

    if len(locals()) > 1:
        PointMatchesOffs = 0
        return NumPointMatches, PointMatchesOffs
    else:
        return NumPointMatches

def get_mosaic_size(IniH, data, MosaicOrigin, MosaicResolution):
    """

    Args:
        IniH (): Initial estimate of image-to-map 2D planar transformations
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


def calculate_homography_3d(x, K):
    """
    From 3d pose to 2d homography

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

        # wi = np.sqrt(1.0 - vxi * vxi - vyi * vyi - vzi * vzi)

        # q1 = RotLib.from_quat([vxi, vyi, vzi, wi ])
        # cRw =q1.as_matrix()#
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
