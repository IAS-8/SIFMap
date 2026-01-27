# SPDX-FileCopyrightText: © 2025 Forschungszentrum Jülich GmbH
# SPDX-FileContributor: Armagan Elibol
# SPDX-FileContributor: Christian Hirt
# SPDX-FileContributor: Jim Buffat <j.buffat@fz-juelich.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import scipy.optimize as scopt
from joblib import Parallel

from align.calc import calculate_homography_3d, vector_to_affine_homography
from align.jacobians import global_3d_jacobian
from data.geometry import rotmat2quat
from data.utils import init_with_valid_kwargs, convert_to_numba_list_of_lists

import numba
from numba import jit


def optimize_alignment_3d(K, data,
                          RTs=None, x=None, parallel_context=None, parallel_params=None,
                          optim_params=None, **kwargs):
    """

    Args:
        K (): camera instrinsics
        RTs (): rotation and translations
        NumIter ():
        WPointMatches ():
        PointMatchesIdx (): correspondences data (x,y,x',y')
        PointMatchesData (): successfully matched image pairs list

    Returns:

    """
    if parallel_params is None:
        parallel_params = dict(do_parallel=False)

    NumVariables = 6
    if x is None:
        N = len(RTs)
        x = np.zeros(N * NumVariables)
        o = np.arange(1, NumVariables + 1)

        for i in range(N):
            # q1 = RotLib.from_matrix(RTs[i]['cRw'])
            # q = q1.as_quat()
            q = rotmat2quat(RTs[i]['cRw'])
            x[o - 1] = np.concatenate((q[1:4], RTs[i]['wTc']))
            o += NumVariables

    else:
        N = data.match_matrix.shape[0]  #int(len(x) / NumVariables)
    
    b1 = [-1]
    b2 = [-100000]
    b = b1*3 + b2*3
    lb = b*N
    b1 = [1]
    b2 = [100000]
    b = b1*3 + b2*3
    ub = b*N
    bou =(lb,ub)

    if K.shape == (3, 3, 1):
        K = np.tile(K, (1, 1, N))

    elif K.shape == (3, 3, N):
        pass

    else:
        # Handle other cases
        pass

    TotalVars = len(x)

    iHw0, wHi0 = calculate_homography_3d(x, K)

    QX = np.arange(0, TotalVars, NumVariables)
    QY = QX + 1
    QZ = QY + 1

    def cost3d(x, data, K, parallel_context):
        return cost_alignment_3d(x, QX, QY, QZ, data, K, parallel_context=parallel_context)

    _optim_params = dict(verbose=2, x_scale='jac', bounds=bou, xtol=1e-3, ftol=1e-6)

    if optim_params is not None:
        _optim_params.update(optim_params)

    if parallel_context is None and parallel_params['do_parallel']:
        context = init_with_valid_kwargs(Parallel, **parallel_params)
    
    elif not parallel_params['do_parallel']:
        context = None

    res = scopt.least_squares(fun=cost3d, x0=x, jac=global_3d_jacobian, args=(data, K, context), **_optim_params)

    x = res.x
    iHw, wHi = calculate_homography_3d(x, K)
    return iHw0, wHi0, iHw, wHi, x, res['fun'], parallel_context


def cost_alignment_3d(x, QX, QY, QZ, data, K, parallel_context=None):
    """

    Args:
        x (): motion paramaters of all images in a vector format
        QX (): quaternion parameters indexes in vector x
        QY (): quaternion parameters indexes in vector x
        QZ (): quaternion parameters indexes in vector x
        PointMatchesData (): correspondences data (x,y,x',y') used in global alignment
        PointMatchesIdx (): successfully matched image pairs list
        CompPointMatches (): boolean true
        K (): camera instrinsics
        WPointMatches (): weight for point and matches residuals

    Returns:
        r : residual vector, 4 for each correspondences
    """
    xi = x.copy()
    Vxi = xi[QX]
    Vyi = xi[QY]
    Vzi = xi[QZ]

    Sq = -np.multiply(Vxi, Vxi) - np.multiply(Vyi, Vyi) - np.multiply(Vzi, Vzi)

    Idx = np.where(Sq < -1.0)[0]
    if Idx.size > 0:
        qd = np.zeros((Idx.shape[0], 4))

        qd[:, 1] = Vxi[Idx]
        qd[:, 2] = Vyi[Idx]
        qd[:, 3] = Vzi[Idx]
        val = np.sqrt(np.square(qd[:, 1]) + np.square(qd[:, 2]) + np.square(qd[:, 3])) - 1.0

        qd[:, 0] = np.cos(np.arcsin(val))
        qd = np.where(np.abs(qd) < np.finfo(float).eps, 0, qd)
        Norms = np.sqrt(np.square(qd[:, 0]) + np.square(qd[:, 1]) + np.square(qd[:, 2]) + np.square(qd[:, 3]))

        xi[QX[Idx]] = np.divide(qd[:, 1], Norms)
        xi[QY[Idx]] = np.divide(qd[:, 2], Norms)
        xi[QZ[Idx]] = np.divide(qd[:, 3], Norms)

    iHw, wHi = calculate_homography_3d(xi, K)

    match_matrix = convert_to_numba_list_of_lists(data.match_matrix)
    residuals = point_match_residual_3d(iHw, wHi, match_matrix, data.matches)

    return residuals


@jit(nopython=True, cache=True, parallel=False)
def point_match_residual_3d(iHw, wHi, match_matrix, matches):
    """

    Args:
        iHw (): world-to-image transformations
        wHi (): image-to-world transfomrations
        PointMatchesData (): correspondences data (x, y, x', y')
            used in global alignment
        PointMatchesIdx (): successfully matched image pairs list 

    Returns:
        r : vector composed of residuals 4 for each correspondences
        symmetric transfer error
    """

    n_correspondences = matches[:, 2].sum()
    pos = np.concatenate((np.array([0]), np.cumsum(matches[:, 2])))
    n_pairs = len(matches)

    # Initialize residual vector
    r = np.zeros(4 * n_correspondences, dtype=np.float64)

    # Scan overlapping pairs of images
    for s in numba.prange(n_pairs):

        # Get the Image Indices
        i = matches[s][0]  # Current Image (Node in the Mosaic)
        j = matches[s][1]   # Reference Image (Edge in the Mosaic)

        # Calculate the Relative Homographies from Absolute Homographies
        iHj = np.ascontiguousarray(iHw[:, :, i]) @ np.ascontiguousarray(wHi[:, :, j])   # cHr (Current H Reference)
        jHi = np.ascontiguousarray(iHw[:, :, j]) @ np.ascontiguousarray(wHi[:, :, i])   # rHc (Reference H Current)

        # l = 0
        # Scan correspondences in the overlapping pairs
        n_points = len(match_matrix[i][j])
        rr = np.zeros(4*n_points)
               
        Im1PList = match_matrix[j][i]
        Im2PList = match_matrix[i][j]

        for t in range(n_points):
            
            # t correspondences index
            px = Im1PList[t, 0]   # Current Image: x
            py = Im1PList[t, 1]   # Current Image: y
            mx = Im2PList[t, 0]   # Reference Image: x
            my = Im2PList[t, 1]   # Reference Image: y

            # Calculate the Residuals
            # x_i - iHj * x_j
            P = iHj @ np.array([[mx], [my], [1.0]])  # np.dot(iHj, np.array([[mx], [my], [1]]))
            rr[4*t] = (px - P[0] / P[2])[0]
            rr[4*t + 1] = (py - P[1] / P[2])[0]

            # x_j - jHi * x_i
            P = jHi @ np.array([[px], [py], [1.0]])  # np.dot(jHi, np.array([[px], [py], [1]]))
            rr[4*t + 2] = (mx - P[0] / P[2])[0]
            rr[4*t + 3] = (my - P[1] / P[2])[0]
        
        k = pos[s] * 4
        r[k:k+4*n_points] = rr

    return r


def point_match_residual_2d(x, data, xext=None):
    """

    Args:
        x (): vector composed of motion parameters
        data (): Data structure to keep the successfully matched
            correspondences positions (x,y) between overlapping image pairs

    Returns:
        r : residual vector
    """
    matches = data.matches
    n_images = data.match_matrix.shape[0]

    # include first homography as an identity mapping, first image frame is mosaic frame
    if xext is None :
        xext = [1, 0, 0, 0, 1, 0]

    xext = np.concatenate((xext, x))

    H = vector_to_affine_homography(xext)

    match_matrix = convert_to_numba_list_of_lists(data.match_matrix)
    r = _point_match_residual_2d(H, match_matrix, matches)

    return r


@jit(nopython=True, cache=True, parallel=True)
def _point_match_residual_2d(H, match_matrix, matches):

    # 4 residuals for each point-match (two error term * two coordinates (x,y))
    n_correspondences = matches[:, 2].sum()
    pos = np.concatenate((np.array([0]), np.cumsum(matches[:, 2])))
    n_pairs = len(matches)

    r = np.zeros(4 * n_correspondences)
    for i in numba.prange(n_pairs):
        Im2 = matches[i][0]
        Im1 = matches[i][1]

        NbrPoint = match_matrix[Im1][Im2].shape[0]

        Im1PList = match_matrix[Im2][Im1].T
        Im2PList = match_matrix[Im1][Im2].T

        rr = np.zeros(4 * NbrPoint)

        Hi = H[Im1]  # ['H']
        Hj = H[Im2]  # ['H']
        Hi = Hi / Hi[2, 2]
        Hj = Hj / Hj[2, 2]

        Hij = np.linalg.inv(Hi) @ Hj
        Hji = np.linalg.inv(Hj) @ Hi
        Hij = Hij / Hij[2, 2]
        Hji = Hji / Hji[2, 2]
        #d1 = Hji @ np.vstack([Im1PList, [1] * Im1PList.shape[1]])
        #d2 = Hij @ np.vstack([Im2PList, [1] * Im2PList.shape[1]])
        d1 = Hji @ np.concatenate((Im1PList, np.ones((1, Im1PList.shape[1]))), axis=0)
        d2 = Hij @ np.concatenate((Im2PList, np.ones((1, Im2PList.shape[1]))), axis=0)

        d1 = d1 / d1[2, :]
        d2 = d2 / d2[2, :]
        r11 = d1[0, :] - Im2PList[0, :]
        r12 = d1[1, :] - Im2PList[1, :]
        r13 = d2[0, :] - Im1PList[0, :]
        r14 = d2[1, :] - Im1PList[1, :]

        rr[0::4] = r11
        rr[1::4] = r12
        rr[2::4] = r13
        rr[3::4] = r14

        k1 = 4 * pos[i]
        r[k1:k1 + 4 * NbrPoint] = rr
        #k1 += 4 * NbrPoint
        
    return r
