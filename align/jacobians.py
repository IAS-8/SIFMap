# SPDX-FileCopyrightText: © 2025 Forschungszentrum Jülich GmbH
# SPDX-FileContributor: Armagan Elibol
# SPDX-FileContributor: Christian Hirt
# SPDX-FileContributor: Jim Buffat <j.buffat@fz-juelich.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from align.calc import vector_to_affine_homography
from data.utils import run_jobs, timeit, chunk_list, convert_to_numba_list_of_lists
from functools import partial

import numba
from numba import jit


def global_3d_jacobian(x, data, K, parallel_context=None):
    """
        This function computes the jacobian matrix for the symmetric transfer 
        error with motion modeled in 3D 
    Args:
        x ():image-to-map motion parameters in a vector
        PointMatchesData (): correspondences between images
        PointMatchesIdx (): image pair indexes 
        K (): camera instrinaics
        WPointMatches (): weight for the residuals

    Returns:
        J : jacobian matrix 
    """
    matches = data.matches

    n_pairs = len(matches)
    n_correspondences = matches[:, 2].sum()
    
    tot_n_params = len(x)
    n_params = 6
    n_cost_terms_per_corr = 4   # |y'- Hy| and |y - Hinv y'| in both 2d directions

    n_residuals = n_cost_terms_per_corr * n_correspondences
    n_derivs_per_cost_term = n_params * 2

    jac_size = n_correspondences * n_cost_terms_per_corr  * n_derivs_per_cost_term

    # Prepare indices and value matrices for jacobian matrix
    triplet = {"Row": np.zeros(jac_size), "Col": np.zeros(jac_size), "Val": np.ones(jac_size)}

    row = np.flip(np.arange(1, n_params + 1))

    # Prepare indices
    k = 0
    t = 0
    for n in range(n_pairs):
        Ii = (data.matches[n, 0] + 1) * n_params
        Ij = (data.matches[n, 1] + 1) * n_params
        rows = np.vstack((Ii - row, Ij - row))
        a = np.array(rows.flatten(), dtype=int)

        n_points = data.matches[n, 2]
        for m in range(n_points * 4):
            triplet["Col"][t:t + n_derivs_per_cost_term] = a
            triplet["Row"][t:t + n_derivs_per_cost_term] = k
            k += 1
            t += n_derivs_per_cost_term

    match_matrix = convert_to_numba_list_of_lists(data.match_matrix)
    triplet_val = _global_3d_jacobian(K, n_params, match_matrix, matches, x)
    
    triplet["Val"] = triplet_val
    J = csc_matrix((triplet["Val"], (triplet["Col"], triplet["Row"])),
                   shape=(tot_n_params, n_residuals))
    J = J.transpose()

    return J


@jit(nopython=True, cache=True, parallel=False)
def _global_3d_jacobian(K, n_params, match_matrix, matches, x):
    n_pairs = len(matches)
    n_correspondences = matches[:, 2].sum()
    pos = np.concatenate((np.array([0]), np.cumsum(matches[:, 2])))

    row = np.flip(np.arange(1, n_params + 1))

    triplet_val = np.zeros(n_correspondences * 48, dtype=np.float64)
    for n in numba.prange(n_pairs):

        i = matches[n, 0]
        j = matches[n, 1]

        Ii = (i + 1) * n_params
        Ij = (j + 1) * n_params
        rows = np.vstack((Ii - row, Ij - row)).flatten()

        # Obtain Ks
        Ki = K[:, :, i]
        # Kj = K[:, :, j]

        Vars = np.concatenate(
            (np.array([Ki[0, 0], Ki[1, 1], Ki[0, 2], Ki[1, 2], Ki[0, 0], Ki[1, 1], Ki[0, 2], Ki[1, 2]]), x[rows]))
        
        coords = np.concatenate((match_matrix[j][i], match_matrix[i][j]), axis=1)

        n_points = matches[n, 2]
        for m in np.arange(n_points):
            J1, J2, J3, J4 = _single_pair_jacobian_point_matches_3d(np.concatenate((Vars, coords[m])))
            Js = J1, J2, J3, J4
            _append = np.concatenate(Js).flatten()
            #_append = np.array([J[k] for k in range(len(Js[0])) for J in Js ])

            start = pos[n] * len(_append) + m * len(_append)
            stop = start + len(_append)

            triplet_val[start:stop] = _append

    return triplet_val


def global_2d_jacobian(x, data, xext):
    """
        This function computes the jacobian matrix for the symmetric transfer
        error with affine 2d planar motion models
    Args:
        x (): image-to-map motion parameters in a vector
        DATAstruct : Data structure to keep the successfully matched
            correspondences positions (x,y) between overlapping image pairs
        FinalList (): List of succesfully matched image indices

    Returns:
        J : Jacobian matrix
    """
    # Initialize variables
    n_images = data.match_matrix.shape[0]
    matches = data.matches

    n_correspondences = int(np.sum(matches[:, 2]))  # total number of correspondences
    n_params = 6

    if xext is None:
        xext = [1, 0, 0, 0, 1, 0]

    xext = np.concatenate((xext, x))  # include first homography as an identity mapping, first image frame is mosaic frame
    H = vector_to_affine_homography(xext)

    # Initialize row, column and value arrays for Jacobian Matrix
    match_matrix = convert_to_numba_list_of_lists(data.match_matrix)
    H = np.asarray(H)
    I, J, X = _global_2d_jacobian(H, n_params, match_matrix, matches)

    rows = 4 * n_correspondences
    cols = n_params * n_images

    # Create the sparse Jacobain matrix
    J = csr_matrix((X, (I.flatten(), J.flatten())), shape=(rows, cols))

    # First is considered fixed as global (map) frame
    # Trim the matrix J1 to exclude the first 6 columns
    J = J[:, 6:]
    return J


@jit(nopython=True, cache=True, parallel=True)
def _global_2d_jacobian(H, n_params, match_matrix, matches):
    n_pairs = len(match_matrix)
    n_correspondences = matches[:, 2].sum()

    n_cost_terms_per_corr = 4  # |x'- Hx| and |x - Hinv x'| in x and y direction
    n_derivs_per_cost_term_per_param = 2
    n_derivs_per_cost_term_per_corr = n_params * n_derivs_per_cost_term_per_param

    jac_size = n_correspondences * n_cost_terms_per_corr * n_derivs_per_cost_term_per_param

    I = np.zeros(jac_size)
    J = np.zeros(jac_size)
    X = np.zeros(jac_size)

    pos = np.concatenate((np.array([0]), np.cumsum(matches[:, 2])))

    # Loop through pairs
    for i in numba.prange(n_pairs):
        Im2 = int(matches[i][0])
        Im1 = int(matches[i][1])
        n_points = match_matrix[Im1][Im2].shape[0]
        Im1PList = match_matrix[Im2][Im1]
        Im2PList = match_matrix[Im1][Im2]
        
        r1_6, r1_7, r2_6, r2_7, r3_6, r3_7, r4_6, r4_7 = _single_pair_jacobian_2d(H, Im1, Im1PList, Im2, Im2PList,
                                                                                  n_points)
        # Loop through each point
        # create row, column, and value triplets (I,J,X) for each correspondences
        for k in range(n_points):
            RowInd = pos[i] * n_derivs_per_cost_term_per_corr * 4 + k * n_derivs_per_cost_term_per_corr * 4
            PointInd = pos[i] * 4 + k * 4

            r1 = np.concatenate((r1_6[k, :], r1_7[k, :]))
            r2 = np.concatenate((r2_6[k, :], r2_7[k, :]))
            r3 = np.concatenate((r3_6[k, :], r3_7[k, :]))
            r4 = np.concatenate((r4_6[k, :], r4_7[k, :]))

            id1 = np.arange(Im1 * n_params, (Im1 + 1) * n_params)
            id2 = np.arange(Im2 * n_params, (Im2 + 1) * n_params)

            I[RowInd:RowInd + n_derivs_per_cost_term_per_corr] = np.ones_like(I[RowInd:RowInd + n_derivs_per_cost_term_per_corr]) * PointInd
            J[RowInd:RowInd + n_derivs_per_cost_term_per_corr] = np.concatenate((id1, id2))
            X[RowInd:RowInd + n_derivs_per_cost_term_per_corr] = r1
            RowInd += n_derivs_per_cost_term_per_corr

            I[RowInd:RowInd + n_derivs_per_cost_term_per_corr] = np.ones_like(I[RowInd:RowInd + n_derivs_per_cost_term_per_corr]) * (PointInd + 1)
            J[RowInd:RowInd + n_derivs_per_cost_term_per_corr] = np.concatenate((id1, id2))
            X[RowInd:RowInd + n_derivs_per_cost_term_per_corr] = r2
            RowInd += n_derivs_per_cost_term_per_corr

            I[RowInd:RowInd + n_derivs_per_cost_term_per_corr] = np.ones_like(I[RowInd:RowInd + n_derivs_per_cost_term_per_corr]) * (PointInd + 2)
            J[RowInd:RowInd + n_derivs_per_cost_term_per_corr] = np.concatenate((id1, id2))
            X[RowInd:RowInd + n_derivs_per_cost_term_per_corr] = r3
            RowInd += n_derivs_per_cost_term_per_corr

            I[RowInd:RowInd + n_derivs_per_cost_term_per_corr] = np.ones_like(I[RowInd:RowInd + n_derivs_per_cost_term_per_corr]) * (PointInd + 3)
            J[RowInd:RowInd + n_derivs_per_cost_term_per_corr] = np.concatenate((id1, id2))
            X[RowInd:RowInd + n_derivs_per_cost_term_per_corr] = r4
            RowInd += n_derivs_per_cost_term_per_corr

            PointInd += 4

    return I, J, X


@jit(nopython=True, cache=True, parallel=False)
def _single_pair_jacobian_2d(H, Im1, Im1PList, Im2, Im2PList, NbrPoint):
    Hi = H[Im1]  # ['H']
    Hj = H[Im2]  # ['H']
    Hi = Hi / Hi[2, 2]
    Hj = Hj / Hj[2, 2]
    # Extract elements
    h1i, h2i, h3i = Hi[0, 0], Hi[0, 1], Hi[0, 2]
    h4i, h5i, h6i = Hi[1, 0], Hi[1, 1], Hi[1, 2]
    h1j, h2j, h3j = Hj[0, 0], Hj[0, 1], Hj[0, 2]
    h4j, h5j, h6j = Hj[1, 0], Hj[1, 1], Hj[1, 2]
    dmat = np.ones(NbrPoint)
    px = Im1PList[:, 0]
    py = Im1PList[:, 1]
    mx = Im2PList[:, 0]
    my = Im2PList[:, 1]
    sigma1 = h1j * h5j - h2j * h4j;
    r1_6 = np.column_stack((
        (h5j * px) / sigma1,
        (h5j * py) / sigma1,
        dmat * (h5j / sigma1),
        -(h2j * px) / sigma1,
        -(h2j * py) / sigma1,
        dmat * (-h2j / sigma1)
    ))
    r1_7 = np.column_stack((
        -(h5j * (
                h3i * h5j - h6i * h2j + h2j * h6j - h3j * h5j + h1i * h5j * px - h4i * h2j * px + h2i * h5j * py - h5i * h2j * py)) / (
                sigma1 ** 2),
        (h5j * (
                h3i * h4j - h6i * h1j + h1j * h6j - h3j * h4j + h1i * h4j * px - h4i * h1j * px + h2i * h4j * py - h5i * h1j * py)) / (
                sigma1 ** 2),
        dmat * (-h5j / sigma1),
        (h2j * (
                h3i * h5j - h6i * h2j + h2j * h6j - h3j * h5j + h1i * h5j * px - h4i * h2j * px + h2i * h5j * py - h5i * h2j * py)) / (
                sigma1 ** 2),
        -(h2j * (
                h3i * h4j - h6i * h1j + h1j * h6j - h3j * h4j + h1i * h4j * px - h4i * h1j * px + h2i * h4j * py - h5i * h1j * py)) / (
                sigma1 ** 2),
        dmat * (h2j / sigma1)
    ))
    sigma2 = h1j * h5j - h2j * h4j
    r2_6 = np.column_stack((
        -(h4j * px) / sigma2,
        -(h4j * py) / sigma2,
        dmat * (-h4j / sigma2),
        (h1j * px) / sigma2,
        (h1j * py) / sigma2,
        dmat * (h1j / sigma2)
    ))
    r2_7 = np.column_stack((
        (h4j * (h3i * h5j - h6i * h2j + h2j * h6j - h3j * h5j +
                h1i * h5j * px - h4i * h2j * px + h2i * h5j * py - h5i * h2j * py)) / (sigma2 ** 2),
        -(h4j * (h3i * h4j - h6i * h1j + h1j * h6j - h3j * h4j +
                 h1i * h4j * px - h4i * h1j * px + h2i * h4j * py - h5i * h1j * py)) / (sigma2 ** 2),
        dmat * (h4j / sigma2),
        -(h1j * (h3i * h5j - h6i * h2j + h2j * h6j - h3j * h5j +
                 h1i * h5j * px - h4i * h2j * px + h2i * h5j * py - h5i * h2j * py)) / (sigma2 ** 2),
        (h1j * (h3i * h4j - h6i * h1j + h1j * h6j - h3j * h4j +
                h1i * h4j * px - h4i * h1j * px + h2i * h4j * py - h5i * h1j * py)) / (sigma2 ** 2),
        dmat * (-h1j / sigma2)
    ))
    sigma1 = h1i * h5i - h2i * h4i
    r3_6 = np.column_stack((
        -(h5i * (h2i * h6i - h3i * h5i - h2i * h6j + h5i * h3j -
                 h2i * h4j * mx + h5i * h1j * mx - h2i * h5j * my + h5i * h2j * my)) / (sigma1 ** 2),
        (h5i * (h1i * h6i - h3i * h4i - h1i * h6j + h4i * h3j -
                h1i * h4j * mx + h4i * h1j * mx - h1i * h5j * my + h4i * h2j * my)) / (sigma1 ** 2),
        dmat * (-h5i / sigma1),
        (h2i * (h2i * h6i - h3i * h5i - h2i * h6j + h5i * h3j -
                h2i * h4j * mx + h5i * h1j * mx - h2i * h5j * my + h5i * h2j * my)) / (sigma1 ** 2),
        -(h2i * (h1i * h6i - h3i * h4i - h1i * h6j + h4i * h3j -
                 h1i * h4j * mx + h4i * h1j * mx - h1i * h5j * my + h4i * h2j * my)) / (sigma1 ** 2),
        dmat * (h2i / sigma1)
    ))
    r3_7 = np.column_stack((
        (h5i * mx) / sigma1,
        (h5i * my) / sigma1,
        dmat * (h5i / sigma1),
        -(h2i * mx) / sigma1,
        -(h2i * my) / sigma1,
        dmat * (-h2i / sigma1)
    ))
    sigma2 = h1i * h5i - h2i * h4i
    r4_6 = np.column_stack((
        (h4i * (h2i * h6i - h3i * h5i - h2i * h6j + h5i * h3j -
                h2i * h4j * mx + h5i * h1j * mx - h2i * h5j * my + h5i * h2j * my)) / (sigma2 ** 2),
        -(h4i * (h1i * h6i - h3i * h4i - h1i * h6j + h4i * h3j -
                 h1i * h4j * mx + h4i * h1j * mx - h1i * h5j * my + h4i * h2j * my)) / (sigma2 ** 2),
        dmat * (h4i / sigma2),
        -(h1i * (h2i * h6i - h3i * h5i - h2i * h6j + h5i * h3j -
                 h2i * h4j * mx + h5i * h1j * mx - h2i * h5j * my + h5i * h2j * my)) / (sigma2 ** 2),
        (h1i * (h1i * h6i - h3i * h4i - h1i * h6j + h4i * h3j -
                h1i * h4j * mx + h4i * h1j * mx - h1i * h5j * my + h4i * h2j * my)) / (sigma2 ** 2),
        dmat * (-h1i / sigma2)
    ))
    r4_7 = np.column_stack((
        -(h4i * mx) / sigma2,
        -(h4i * my) / sigma2,
        dmat * (-h4i / sigma2),
        (h1i * mx) / sigma2,
        (h1i * my) / sigma2,
        dmat * (h1i / sigma2)
    ))
    return r1_6, r1_7, r2_6, r2_7, r3_6, r3_7, r4_6, r4_7


@jit(nopython=True, cache=True, parallel=False)
def _single_pair_jacobian_point_matches_3d(vars):
    """
        This function computes jacobian entries for a given image pair and
        a single correspondence between them
    Args:
        vars (): motion parameters of two images with calibration instrinsics

    Returns:
        j1, j2, j3, j4 : Jacobian entries for each residual
    """
    # VARIABLES = "Vars = (aui, avi, cxi, cyi, auj, avj, cxj, cyj, vxi, vyi, " \
    #            "vzi, txi, tyi, tzi, vxi, vyi, vzi, txj, tyj, tzj, px, py, mx, my)"

    AUI, AVI, CXI, CYI, AUJ, AVJ, CXJ, CYJ, VXI, VYI, VZI, TXI, TYI, TZI, \
        VXJ, VYJ, VZJ, TXJ, TYJ, TZJ, PX, PY, MX, MY = range(24)

    # Parameter checks
    # if len(vars) != 24:
    #     raise ValueError(ERROR_HEADER + "Vars must be a length 24 vector\n\n" + VARIABLES)

    # Extracting variables from input
    aui = vars[AUI]
    avi = vars[AVI]
    cxi = vars[CXI]
    cyi = vars[CYI]
    auj = vars[AUJ]
    avj = vars[AVJ]
    cxj = vars[CXJ]
    cyj = vars[CYJ]
    vxi = vars[VXI]
    vyi = vars[VYI]
    vzi = vars[VZI]
    txi = vars[TXI]
    tyi = vars[TYI]
    tzi = vars[TZI]
    vxj = vars[VXJ]
    vyj = vars[VYJ]
    vzj = vars[VZJ]
    txj = vars[TXJ]
    tyj = vars[TYJ]
    tzj = vars[TZJ]
    px = vars[PX]
    py = vars[PY]
    mx = vars[MX]
    my = vars[MY]

    wi = 1.0 - vxi * vxi - vyi * vyi - vzi * vzi
    if wi < 0.:
        if np.abs(np.sqrt(vxi * vxi + vyi * vyi + vzi * vzi) - 1.0) < 1e-6:
            wi = np.cos(np.arcsin(0.))
        else:
            wi = np.cos(np.arcsin(np.sqrt(vxi * vxi + vyi * vyi + vzi * vzi) - 1.0))

        Norms = np.sqrt(vxi * vxi + vyi * vyi + vzi * vzi + wi * wi);
        wi = wi / Norms
        vxi = vxi / Norms
        vyi = vyi / Norms
        vzi = vzi / Norms
    wj = 1.0 - vxj * vxj - vyj * vyj - vzj * vzj
    if wj < 0.:
        if np.abs(np.sqrt(vxj * vxj + vyj * vyj + vzj * vzj) - 1.0) < 1e-6:
            wj = np.cos(np.arcsin(0.))
        else:
            wj = np.cos(np.arcsin(np.sqrt(vxj * vxj + vyj * vyj + vzj * vzj) - 1.0))

        Norms = np.sqrt(vxj * vxj + vyj * vyj + vzj * vzj + wj * wj);
        wj = wj / Norms
        vxj = vxj / Norms
        vyj = vyj / Norms
        vzj = vzj / Norms
    vxi2 = vxi * vxi
    vyi2 = vyi * vyi
    vzi2 = vzi * vzi
    wi2 = -vxi2 - vyi2 - vzi2

    wi = np.sqrt(wi2 + 1.0)
    vxyi = vxi * vyi
    vxzi = vxi * vzi
    vyzi = vyi * vzi
    vxj2 = vxj * vxj
    vyj2 = vyj * vyj
    vzj2 = vzj * vzj
    wj2 = -vxj2 - vyj2 - vzj2
    wj = np.sqrt(wj2 + 1.0)
    vxyj = vxj * vyj
    vxzj = vxj * vzj
    vyzj = vyj * vzj

    # Check whether the sqrt() can be computed or not
    if wi2 < -1.0:
        print(f"Point Matches Jacobian cannot be calculated: (wi2 = {wi2}) < 0 in sqrt (wi2)!")
        wi = 0.0

    if wj2 < -1.0:
        print(f"Point Matches Jacobian cannot be calculated: (wj2 = {wj2}) < 0 in sqrt (wj2)!")
        wj = 0.0
    x0 = vxj ** 2
    x1 = 2.0 * x0
    x2 = vyj ** 2
    x3 = 2.0 * x2
    x4 = x3 - 1.0
    x5 = x1 + x4
    x6 = 2.0 * vyi
    x7 = vzi * x6
    x8 = vxi ** 2
    x9 = vyi ** 2
    x10 = vzi ** 2
    x11 = np.sqrt(-x10 - x8 - x9 + 1.0)
    x12 = 2.0 * x11
    x13 = vxi * x12
    x14 = x13 + x7
    x15 = 1. / tzj
    x16 = tyj * x15
    x17 = 2.0 * vzi
    x18 = vxi * x17
    x19 = -2.0 * vyi * x11 + x18
    x20 = -x19
    x21 = txj * x15
    x22 = 2.0 * x8
    x23 = 2.0 * x9
    x24 = x23 - 1.0
    x25 = x22 + x24
    x26 = txi * x20 - tyi * x14 + tzi * x25
    x27 = x14 * x16 + x15 * x26 - x20 * x21
    x28 = 1. / auj
    x29 = vzj ** 2
    x30 = 2.0 * x29
    x31 = x30 + x4
    x32 = 2.0 * vyj
    x33 = vxj * x32
    x34 = np.sqrt(-x0 - x2 - x29 + 1.0)
    x35 = 2.0 * vzj * x34 - x33
    x36 = 2.0 * vzj
    x37 = vxj * x36
    x38 = 2.0 * x34
    x39 = vyj * x38 + x37
    x40 = x28 * (x14 * x35 + x19 * x31 + x27 * x39)
    x41 = 1. / avj
    x42 = x1 + x30 - 1.0
    x43 = x33 + x34 * x36
    x44 = vyj * x36
    x45 = vxj * x38
    x46 = -x44 + x45
    x47 = x41 * (x14 * x42 + x20 * x43 - x27 * x46)
    x48 = x44 + x45
    x49 = -2.0 * vyj * x34 + x37
    x50 = -x49
    x51 = x14 * x48 + x20 * x50
    x52 = cxj * x40 + cyj * x47 - mx * x40 - my * x47 + x27 * x5 + x51
    x53 = 1. / x52
    x54 = 1. / x11
    x55 = vxi * x6
    x56 = x54 * x55
    x57 = x17 + x56
    x58 = x18 * x54
    x59 = x58 + x6
    x60 = -2.0 * x11
    x61 = x22 * x54 + x60
    x62 = -x61
    x63 = aui * x59 + cxi * x62
    x64 = txi * x57
    x65 = x17 - x56
    x66 = cxi * x21 * x57 + x15 * (-cxi * x64 - tyi * x63 + tzi * (-aui * x65 + 4.0 * cxi * vxi)) + x16 * x63
    x67 = x31 * x57
    x68 = cxi * x67 + x35 * x63 + x39 * x66
    x69 = x43 * x57
    x70 = cxi * x69 - x42 * x63 + x46 * x66
    x71 = x15 * (-tyi * x62 + 4.0 * tzi * vxi - x64) + x16 * x62 + x21 * x57
    x72 = x28 * (x35 * x62 + x39 * x71 + x67)
    x73 = x41 * (-x42 * x62 + x46 * x71 + x69)
    x74 = x48 * x62 - x50 * x57
    x75 = cxj * x72 - cyj * x73 - mx * x72 + my * x73 + x5 * x71 + x74
    x76 = x52 ** (-2.0)
    x77 = 2.0 * x10
    x78 = x24 + x77
    x79 = aui * x78 + cxi * x20
    x80 = 2.0 * vzi * x11 - x55
    x81 = aui * x80 - cxi * x14
    x82 = x11 * x6 + x18
    x83 = -aui * x82 + cxi * x25
    x84 = txi * x79 + tyi * x81 + tzi * x83
    x85 = -x15 * x84 + x16 * x81 + x21 * x79
    x86 = x28 * (x31 * x79 + x35 * x81 + x39 * x85)
    x87 = x41 * (-x42 * x81 + x43 * x79 + x46 * x85)
    x88 = x76 * (cxj * x86 - cyj * x87 - mx * x86 + my * x87 + x48 * x81 + x5 * x85 - x50 * x79)
    x89 = 4.0 * vyi
    x90 = x23 * x54 + x60
    x91 = -x90
    x92 = aui * x89 + cxi * x91
    x93 = 2.0 * vxi
    x94 = x54 * x7
    x95 = x93 + x94
    x96 = aui * x95 + cxi * x65
    x97 = x15 * (txi * x92 - tyi * x96 + tzi * (-aui * x91 + 4.0 * cxi * vyi)) + x16 * x96 - x21 * x92
    x98 = -x31 * x92 + x35 * x96 + x39 * x97
    x99 = x42 * x96 + x43 * x92 - x46 * x97
    x100 = x16 * x65
    x101 = tyi * x65
    x102 = x100 + x15 * (txi * x91 + tzi * x89 - x101) - x21 * x91
    x103 = x35 * x65
    x104 = x28 * (x102 * x39 + x103 - x31 * x91)
    x105 = x42 * x65
    x106 = x41 * (-x102 * x46 + x105 + x43 * x91)
    x107 = x48 * x65
    x108 = x107 + x50 * x91
    x109 = cxj * x104 + cyj * x106 - mx * x104 - my * x106 + x102 * x5 + x108
    x110 = 4.0 * vzi
    x111 = aui * x110 - cxi * x95
    x112 = -2.0 * vyi + x58
    x113 = -x112
    x114 = -x54 * x77 - x60
    x115 = aui * x114 - cxi * x113
    x116 = x93 - x94
    x117 = x111 * x21 + x115 * x16 + x15 * (aui * tzi * x116 - txi * x111 - tyi * x115)
    x118 = x28 * (x111 * x31 + x115 * x35 + x117 * x39)
    x119 = x41 * (x111 * x43 - x115 * x42 + x117 * x46)
    x120 = x113 * x48
    x121 = x113 * x16 + x15 * (-txi * x95 - tyi * x113) + x21 * x95
    x122 = x28 * (x113 * x35 + x121 * x39 + x31 * x95)
    x123 = x41 * (x112 * x42 + x121 * x46 + x43 * x95)
    x124 = cxj * x122 - cyj * x123 - mx * x122 + my * x123 + x120 + x121 * x5 - x50 * x95
    x125 = x15 * x5
    x126 = x28 * x39
    x127 = x15 * x79
    x128 = x41 * x46
    x129 = cxj * x126 * x127 - cyj * x15 * x41 * x46 * x79 - mx * x15 * x28 * x39 * x79 + my * x127 * x128 + x125 * x79
    x130 = x15 * x20
    x131 = x126 * x130
    x132 = x128 * x130
    x133 = cxj * x131 - cyj * x132 - mx * x131 + my * x132 + x125 * x20
    x134 = x133 * x88
    x135 = x15 * x81
    x136 = cxj * x126 * x135 - cyj * x15 * x41 * x46 * x81 - mx * x15 * x28 * x39 * x81 + my * x128 * x135 + x125 * x81
    x137 = x14 * x15
    x138 = x126 * x137
    x139 = x128 * x137
    x140 = cxj * x138 - cyj * x139 - mx * x138 + my * x139 + x125 * x14
    x141 = x140 * x88
    x142 = x15 * x83
    x143 = x15 * x25
    x144 = x126 * x143
    x145 = x128 * x143
    x146 = cxj * x144 - cyj * x145 - mx * x144 + my * x145 + x125 * x25
    x147 = 1. / x34
    x148 = x147 * x33
    x149 = x148 + x36
    x150 = -2.0 * x34
    x151 = x1 * x147 + x150
    x152 = -x151
    x153 = 4.0 * vxj
    x154 = x147 * x37
    x155 = x154 + x32
    x156 = 2.0 * vzj - x148
    x157 = x28 * (x155 * x81 - x156 * x85)
    x158 = -2.0 * vyj + x154
    x159 = -x158
    x160 = x41 * (x152 * x85 - x153 * x81 + x159 * x79)
    x161 = x41 * (x14 * x153 + x151 * x27 + x159 * x20)
    x162 = x28 * (x14 * x155 - x156 * x27)
    x163 = x14 * x152 + x149 * x19
    x164 = -cxj * x162 + cyj * x161 + mx * x162 - my * x161 + x153 * x27 + x163
    x165 = x147 * x3 + x150
    x166 = -x165
    x167 = 4.0 * vyj
    x168 = 2.0 * vxj
    x169 = x147 * x44
    x170 = x168 - x169
    x171 = x41 * (-x149 * x85 + x170 * x79)
    x172 = x168 + x169
    x173 = x28 * (x166 * x85 + x167 * x79 - x172 * x81)
    x174 = x41 * (x149 * x27 + x170 * x20)
    x175 = x28 * (x14 * x172 + x165 * x27 + x167 * x20)
    x176 = x14 * x156
    x177 = x166 * x20 + x176
    x178 = -cxj * x175 + cyj * x174 + mx * x175 - my * x174 + x167 * x27 + x177
    x179 = 4.0 * vzj
    x180 = -x147 * x30 - x150
    x181 = x28 * (x170 * x85 + x179 * x79 + x180 * x81)
    x182 = x41 * (x155 * x85 + x179 * x81 - x180 * x79)
    x183 = x172 * x20
    x184 = x14 * x180 + x170 * x27 - x179 * x20
    x185 = x14 * x179 + x155 * x27 + x180 * x20
    x186 = cxj * x184 * x28 + cyj * x185 * x41 - mx * x184 * x28 - my * x185 * x41 + x14 * x159 - x183
    x187 = tzj ** (-2)
    x188 = txj * x187
    x189 = tyj * x187
    x190 = -x187 * x84 + x188 * x79 + x189 * x81
    x191 = x14 * x189 + x187 * x26 - x188 * x20
    x192 = x126 * x191
    x193 = x128 * x191
    x194 = cxj * x192 - cyj * x193 - mx * x192 + my * x193 + x191 * x5
    x195 = 4.0 * vxi
    x196 = avi * x195 - cyi * x62
    x197 = avi * x113 + cyi * x57
    x198 = x15 * (-txi * x197 + tyi * x196 + tzi * (avi * x62 + cyi * x195)) - x16 * x196 + x197 * x21
    x199 = x28 * (-x196 * x35 + x197 * x31 + x198 * x39)
    x200 = x41 * (x196 * x42 + x197 * x43 + x198 * x46)
    x201 = x22 + x77 - 1.0
    x202 = avi * x201 - cyi * x14
    x203 = vzi * x12 + x55
    x204 = avi * x203 - cyi * x20
    x205 = x13 - x7
    x206 = avi * x205 + cyi * x25
    x207 = -txi * x204 + tyi * x202 + tzi * x206
    x208 = x15 * x207 - x16 * x202 + x204 * x21
    x209 = x28 * (-x202 * x35 + x204 * x31 + x208 * x39)
    x210 = x41 * (x202 * x42 + x204 * x43 + x208 * x46)
    x211 = x76 * (-cxj * x209 + cyj * x210 + mx * x209 - my * x210 + x202 * x48 + x204 * x50 - x208 * x5)
    x212 = avi * x116 - cyi * x91
    x213 = -x212
    x214 = cyi * x100 + x15 * (-cyi * x101 + txi * x213 + tzi * (-avi * x57 + 4.0 * cyi * vyi)) - x21 * x213
    x215 = cyi * x103 + x212 * x31 + x214 * x39
    x216 = cyi * x105 + x213 * x43 - x214 * x46
    x217 = avi * x110 - cyi * x113
    x218 = avi * x114 + cyi * x95
    x219 = x15 * (avi * tzi * x59 + txi * x218 - tyi * x217) + x16 * x217 - x21 * x218
    x220 = x28 * (x217 * x35 - x218 * x31 + x219 * x39)
    x221 = x41 * (x217 * x42 + x218 * x43 - x219 * x46)
    x222 = x15 * x204
    x223 = x126 * x222
    x224 = x128 * x222
    x225 = cxj * x223 - cyj * x224 - mx * x223 + my * x224 + x125 * x204
    x226 = x133 * x211
    x227 = x15 * x202
    x228 = cxj * x126 * x227 - cyj * x15 * x202 * x41 * x46 - mx * x15 * x202 * x28 * x39 + my * x128 * x227 + x125 * x202
    x229 = x140 * x211
    x230 = x15 * x206
    x231 = x155 * x202 + x156 * x208
    x232 = x152 * x208 + x153 * x202 + x159 * x204
    x233 = x41 * (-x149 * x208 + x170 * x204)
    x234 = x28 * (x166 * x208 + x167 * x204 + x172 * x202)
    x235 = x170 * x208 + x179 * x204 - x180 * x202
    x236 = -x155 * x208 + x179 * x202 + x180 * x204
    x237 = x187 * x207 + x188 * x204 - x189 * x202
    x238 = x126 * x237
    x239 = x128 * x237
    x240 = 1. / tzi
    x241 = tyi * x240
    x242 = txi * x240
    x243 = txj * x50 - tyj * x48 + tzj * x5
    x244 = x240 * x243 + x241 * x48 - x242 * x50
    x245 = 1. / aui
    x246 = x245 * (x244 * x82 + x48 * x80 + x49 * x78)
    x247 = 1. / avi
    x248 = x247 * (x201 * x48 + x203 * x50 - x205 * x244)
    x249 = cxi * x246 + cyi * x248 - px * x246 - py * x248 + x244 * x25 + x51
    x250 = 1. / x249
    x251 = auj * x31 + cxj * x50
    x252 = auj * x35 - cxj * x48
    x253 = -auj * x39 + cxj * x5
    x254 = txj * x251 + tyj * x252 + tzj * x253
    x255 = -x240 * x254 + x241 * x252 + x242 * x251
    x256 = x245 * (x252 * x59 - x255 * x65)
    x257 = x247 * (x113 * x251 - x195 * x252 + x255 * x62)
    x258 = x247 * (x113 * x50 + x195 * x48 + x244 * x61)
    x259 = x245 * (-x244 * x65 + x48 * x59)
    x260 = -cxi * x259 + cyi * x258 + px * x259 - py * x258 + x195 * x244 + x74
    x261 = x249 ** (-2.0)
    x262 = x245 * (x251 * x78 + x252 * x80 + x255 * x82)
    x263 = x247 * (-x201 * x252 + x203 * x251 + x205 * x255)
    x264 = x261 * (cxi * x262 - cyi * x263 - px * x262 + py * x263 + x14 * x252 - x20 * x251 + x25 * x255)
    x265 = x247 * (x116 * x251 - x255 * x57)
    x266 = x245 * (x251 * x89 - x252 * x95 + x255 * x91)
    x267 = x247 * (x116 * x50 + x244 * x57)
    x268 = x245 * (x244 * x90 + x48 * x95 + x50 * x89)
    x269 = -cxi * x268 + cyi * x267 + px * x268 - py * x267 + x108 + x244 * x89
    x270 = x245 * (x110 * x251 + x114 * x252 + x116 * x255)
    x271 = x247 * (x110 * x252 - x114 * x251 + x255 * x59)
    x272 = x245 * (-x110 * x50 + x114 * x48 + x116 * x244)
    x273 = x247 * (x110 * x48 + x114 * x50 + x244 * x59)
    x274 = -cxi * x272 - cyi * x273 + px * x272 + py * x273 - x120 + x50 * x95
    x275 = x240 * x25
    x276 = x245 * x82
    x277 = x240 * x251
    x278 = x276 * x277
    x279 = x205 * x247
    x280 = x277 * x279
    x281 = cxi * x278 - cyi * x280 - px * x278 + py * x280 + x251 * x275
    x282 = x240 * x50
    x283 = x276 * x282
    x284 = x279 * x282
    x285 = cxi * x283 - cyi * x284 - px * x283 + py * x284 + x275 * x50
    x286 = x264 * x285
    x287 = x240 * x252
    x288 = x276 * x287
    x289 = x279 * x287
    x290 = cxi * x288 - cyi * x289 - px * x288 + py * x289 + x252 * x275
    x291 = x240 * x48
    x292 = x276 * x291
    x293 = x279 * x291
    x294 = cxi * x292 - cyi * x293 - px * x292 + py * x293 + x275 * x48
    x295 = x264 * x294
    x296 = tzi ** (-2)
    x297 = txi * x296
    x298 = tyi * x296
    x299 = x251 * x297 + x252 * x298 - x254 * x296
    x300 = x243 * x296 - x297 * x50 + x298 * x48
    x301 = x276 * x300
    x302 = x279 * x300
    x303 = cxi * x301 - cyi * x302 - px * x301 + py * x302 + x25 * x300
    x304 = auj * x155 + cxj * x152
    x305 = txj * x149
    x306 = cxj * x149 * x242 + x240 * (-cxj * x305 - tyj * x304 + tzj * (-auj * x156 + 4.0 * cxj * vxj)) + x241 * x304
    x307 = x149 * x78
    x308 = cxj * x307 + x304 * x80 + x306 * x82
    x309 = x149 * x203
    x310 = cxj * x309 - x201 * x304 + x205 * x306
    x311 = x149 * x242 + x152 * x241 + x240 * (-tyj * x152 + 4.0 * tzj * vxj - x305)
    x312 = x245 * (x152 * x80 + x307 + x311 * x82)
    x313 = x247 * (-x152 * x201 + x205 * x311 + x309)
    x314 = cxi * x312 - cyi * x313 - px * x312 + py * x313 + x163 + x25 * x311
    x315 = auj * x167 + cxj * x166
    x316 = auj * x172 + cxj * x156
    x317 = x240 * (txj * x315 - tyj * x316 + tzj * (-auj * x166 + 4.0 * cxj * vyj)) + x241 * x316 - x242 * x315
    x318 = -x315 * x78 + x316 * x80 + x317 * x82
    x319 = x201 * x316 + x203 * x315 - x205 * x317
    x320 = x156 * x241
    x321 = tyj * x156
    x322 = -x166 * x242 + x240 * (txj * x166 + tzj * x167 - x321) + x320
    x323 = x156 * x80
    x324 = x245 * (-x166 * x78 + x322 * x82 + x323)
    x325 = x156 * x201
    x326 = x247 * (x166 * x203 - x205 * x322 + x325)
    x327 = cxi * x324 + cyi * x326 - px * x324 - py * x326 + x177 + x25 * x322
    x328 = auj * x179 - cxj * x172
    x329 = auj * x180 - cxj * x159
    x330 = x240 * (auj * tzj * x170 - txj * x328 - tyj * x329) + x241 * x329 + x242 * x328
    x331 = x245 * (x328 * x78 + x329 * x80 + x330 * x82)
    x332 = x247 * (-x201 * x329 + x203 * x328 + x205 * x330)
    x333 = x159 * x241 + x172 * x242 + x240 * (-txj * x172 - tyj * x159)
    x334 = x245 * (x159 * x80 + x172 * x78 + x333 * x82)
    x335 = x247 * (x158 * x201 + x172 * x203 + x205 * x333)
    x336 = cxi * x334 - cyi * x335 - px * x334 + py * x335 + x14 * x159 - x183 + x25 * x333
    x337 = x240 * x253
    x338 = x240 * x5
    x339 = x276 * x338
    x340 = x279 * x338
    x341 = cxi * x339 - cyi * x340 - px * x339 + py * x340 + x275 * x5
    x342 = avj * x43 - cyj * x50
    x343 = avj * x42 - cyj * x48
    x344 = avj * x46 + cyj * x5
    x345 = -txj * x342 + tyj * x343 + tzj * x344
    x346 = x240 * x345 - x241 * x343 + x242 * x342
    x347 = x343 * x59 + x346 * x65
    x348 = x113 * x342 + x195 * x343 + x346 * x62
    x349 = x245 * (x342 * x78 - x343 * x80 + x346 * x82)
    x350 = x247 * (x201 * x343 + x203 * x342 + x205 * x346)
    x351 = x261 * (-cxi * x349 + cyi * x350 + px * x349 - py * x350 + x14 * x343 + x20 * x342 - x25 * x346)
    x352 = x247 * (x116 * x342 - x346 * x57)
    x353 = x245 * (x342 * x89 + x343 * x95 + x346 * x91)
    x354 = x110 * x342 - x114 * x343 + x116 * x346
    x355 = x110 * x343 + x114 * x342 - x346 * x59
    x356 = x240 * x342
    x357 = cxi * x276 * x356 - cyi * x205 * x240 * x247 * x342 - px * x240 * x245 * x342 * x82 + py * x279 * x356 + x275 * x342
    x358 = x285 * x351
    x359 = x240 * x343
    x360 = x276 * x359
    x361 = x279 * x359
    x362 = cxi * x360 - cyi * x361 - px * x360 + py * x361 + x275 * x343
    x363 = x294 * x351
    x364 = x296 * x345 + x297 * x342 - x298 * x343
    x365 = x276 * x364
    x366 = x279 * x364
    x367 = avj * x153 - cyj * x152
    x368 = avj * x159 + cyj * x149
    x369 = x240 * (-txj * x368 + tyj * x367 + tzj * (avj * x152 + cyj * x153)) - x241 * x367 + x242 * x368
    x370 = x245 * (-x367 * x80 + x368 * x78 + x369 * x82)
    x371 = x247 * (x201 * x367 + x203 * x368 + x205 * x369)
    x372 = avj * x170 - cyj * x166
    x373 = -x372
    x374 = cyj * x320 + x240 * (-cyj * x321 + txj * x373 + tzj * (-avj * x149 + 4.0 * cyj * vyj)) - x242 * x373
    x375 = cyj * x323 + x372 * x78 + x374 * x82
    x376 = cyj * x325 + x203 * x373 - x205 * x374
    x377 = avj * x179 - cyj * x159
    x378 = avj * x180 + cyj * x172
    x379 = x240 * (avj * tzj * x155 + txj * x378 - tyj * x377) + x241 * x377 - x242 * x378
    x380 = x245 * (x377 * x80 - x378 * x78 + x379 * x82)
    x381 = x247 * (x201 * x377 + x203 * x378 - x205 * x379)
    x382 = x240 * x344
    j1 = np.zeros(12)
    j1[0] = x53 * (
                cxi * x50 * x57 - cxj * x28 * x68 + cyj * x41 * x70 + mx * x28 * x68 - my * x41 * x70 - x48 * x63 - x5 * x66) - x75 * x88
    j1[1] = -x109 * x88 + x53 * (
                -cxj * x28 * x98 - cyj * x41 * x99 + mx * x28 * x98 + my * x41 * x99 - x48 * x96 - x5 * x97 - x50 * x92)
    j1[2] = -x124 * x88 + x53 * (cxj * x118 - cyj * x119 - mx * x118 + my * x119 - x111 * x50 + x115 * x48 + x117 * x5)
    j1[3] = -x129 * x53 - x134
    j1[4] = -x136 * x53 + x141
    j1[5] = -x146 * x88 + x53 * (
                -cxj * x126 * x142 + cyj * x15 * x41 * x46 * x83 + mx * x15 * x28 * x39 * x83 - my * x128 * x142 - x125 * x83)
    j1[6] = -x164 * x88 + x53 * (
                -cxj * x157 - cyj * x160 + mx * x157 + my * x160 + x149 * x79 + x152 * x81 + x153 * x85)
    j1[7] = -x178 * x88 + x53 * (cxj * x173 - cyj * x171 - mx * x173 + my * x171 + x156 * x81 - x166 * x79 + x167 * x85)
    j1[8] = -x186 * x88 + x53 * (cxj * x181 + cyj * x182 - mx * x181 - my * x182 + x159 * x81 + x172 * x79)
    j1[9] = x129 * x53 + x134
    j1[10] = x136 * x53 - x141
    j1[11] = x194 * x88 + x53 * (
                -cxj * x126 * x190 + cyj * x190 * x41 * x46 + mx * x190 * x28 * x39 - my * x128 * x190 - x190 * x5)
    j2 = np.zeros(12)
    j2[0] = -x211 * x75 + x53 * (-cxj * x199 + cyj * x200 + mx * x199 - my * x200 + x196 * x48 + x197 * x50 - x198 * x5)
    j2[1] = -x109 * x211 + x53 * (
                -cxj * x215 * x28 - cyi * x107 - cyj * x216 * x41 + mx * x215 * x28 + my * x216 * x41 - x213 * x50 - x214 * x5)
    j2[2] = -x124 * x211 + x53 * (cxj * x220 + cyj * x221 - mx * x220 - my * x221 + x217 * x48 + x218 * x50 + x219 * x5)
    j2[3] = x225 * x53 - x226
    j2[4] = -x228 * x53 + x229
    j2[5] = -x146 * x211 + x53 * (
                -cxj * x126 * x230 + cyj * x15 * x206 * x41 * x46 + mx * x15 * x206 * x28 * x39 - my * x128 * x230 - x125 * x206)
    j2[6] = -x164 * x211 + x53 * (
                -cxj * x231 * x28 + cyj * x232 * x41 + mx * x231 * x28 - my * x232 * x41 - x149 * x204 + x152 * x202 - x153 * x208)
    j2[7] = -x178 * x211 + x53 * (
                -cxj * x234 + cyj * x233 + mx * x234 - my * x233 + x156 * x202 + x166 * x204 - x167 * x208)
    j2[8] = -x186 * x211 + x53 * (
                -cxj * x235 * x28 + cyj * x236 * x41 + mx * x235 * x28 - my * x236 * x41 + x159 * x202 - x172 * x204)
    j2[9] = -x225 * x53 + x226
    j2[10] = x228 * x53 - x229
    j2[11] = x194 * x211 + x53 * (cxj * x238 - cyj * x239 - mx * x238 + my * x239 + x237 * x5)
    j3 = np.zeros(12)
    j3[0] = x250 * (
                -cxi * x256 - cyi * x257 + px * x256 + py * x257 + x195 * x255 + x251 * x57 + x252 * x62) - x260 * x264
    j3[1] = x250 * (
                cxi * x266 - cyi * x265 - px * x266 + py * x265 - x251 * x91 + x252 * x65 + x255 * x89) - x264 * x269
    j3[2] = x250 * (cxi * x270 + cyi * x271 - px * x270 - py * x271 + x113 * x252 + x251 * x95) + x264 * x274
    j3[3] = x250 * x281 + x286
    j3[4] = x250 * x290 - x295
    j3[5] = x250 * (
                -cxi * x276 * x299 + cyi * x205 * x247 * x299 + px * x245 * x299 * x82 - py * x279 * x299 - x25 * x299) + x264 * x303
    j3[6] = x250 * (
                -cxi * x245 * x308 + cxj * x149 * x20 + cyi * x247 * x310 + px * x245 * x308 - py * x247 * x310 - x14 * x304 - x25 * x306) - x264 * x314
    j3[7] = x250 * (
                -cxi * x245 * x318 - cyi * x247 * x319 + px * x245 * x318 + py * x247 * x319 - x14 * x316 - x20 * x315 - x25 * x317) - x264 * x327
    j3[8] = x250 * (
                cxi * x331 - cyi * x332 - px * x331 + py * x332 + x14 * x329 - x20 * x328 + x25 * x330) - x264 * x336
    j3[9] = -x250 * x281 - x286
    j3[10] = -x250 * x290 + x295
    j3[11] = x250 * (
                -cxi * x276 * x337 + cyi * x205 * x240 * x247 * x253 + px * x240 * x245 * x253 * x82 - py * x279 * x337 - x253 * x275) - x264 * x341
    j4 = np.zeros(12)
    j4[0] = x250 * (
                -cxi * x245 * x347 + cyi * x247 * x348 + px * x245 * x347 - py * x247 * x348 - x195 * x346 - x342 * x57 + x343 * x62) - x260 * x351
    j4[1] = x250 * (
                -cxi * x353 + cyi * x352 + px * x353 - py * x352 + x342 * x91 + x343 * x65 - x346 * x89) - x269 * x351
    j4[2] = x250 * (
                -cxi * x245 * x354 + cyi * x247 * x355 + px * x245 * x354 - py * x247 * x355 + x113 * x343 - x342 * x95) + x274 * x351
    j4[3] = -x250 * x357 + x358
    j4[4] = x250 * x362 - x363
    j4[5] = x250 * (cxi * x365 - cyi * x366 - px * x365 + py * x366 + x25 * x364) + x303 * x351
    j4[6] = x250 * (
                -cxi * x370 + cyi * x371 + px * x370 - py * x371 + x14 * x367 + x20 * x368 - x25 * x369) - x314 * x351
    j4[7] = x250 * (
                -cxi * x245 * x375 - cyi * x247 * x376 - cyj * x176 + px * x245 * x375 + py * x247 * x376 - x20 * x373 - x25 * x374) - x327 * x351
    j4[8] = x250 * (
                cxi * x380 + cyi * x381 - px * x380 - py * x381 + x14 * x377 + x20 * x378 + x25 * x379) - x336 * x351
    j4[9] = x250 * x357 - x358
    j4[10] = -x250 * x362 + x363
    j4[11] = x250 * (
                -cxi * x276 * x382 + cyi * x205 * x240 * x247 * x344 + px * x240 * x245 * x344 * x82 - py * x279 * x382 - x275 * x344) - x341 * x351

    return j1, j2, j3, j4
