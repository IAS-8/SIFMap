# SPDX-FileCopyrightText: © 2025 Forschungszentrum Jülich GmbH
# SPDX-FileContributor: Armagan Elibol
# SPDX-FileContributor: Christian Hirt
# SPDX-FileContributor: Jim Buffat <j.buffat@fz-juelich.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import math
import numpy as np
import scipy.optimize as scopt

from data.data import spread_points

import numba
from numba import jit


def RotX(a):
    # Compute a Rotation Matrix around X Axis of a Rads
    R = np.array([[1, 0, 0],
                  [0, np.cos(a), -np.sin(a)],
                  [0, np.sin(a), np.cos(a)]])
    return R


def RotY(a):
    # Compute a Rotation Matrix around X Axis of a Rads
    R = np.array([[np.cos(a), 0, np.sin(a)],
                  [0, 1, 0],
                  [-np.sin(a), 0, np.cos(a)]])
    return R


def RotZ(a):
    # Compute a Rotation Matrix around X Axis of a Rads
    R = np.array([[np.cos(a), -np.sin(a), 0],
                  [np.sin(a), np.cos(a), 0],
                  [0, 0, 1]])
    return R


def compute_angle(H, Center):
    """
        This function maps image corners according to the motion and computes
        the angle between diagonals
    Args:
        H (): motion parameters
        Center (): image corners

    Returns:
        Angle : angle in degrees
    """
    Points = np.array([[Center[0], Center[0], Center[0]],
                       [Center[1], 1, 1],
                       [1, 1, 1]])

    ProjPoints = np.dot(H, Points)
    ProjPoints[0, :] /= ProjPoints[2, :]
    ProjPoints[1, :] /= ProjPoints[2, :]

    Angle = np.degrees(np.arctan2(ProjPoints[1, 0] - ProjPoints[1, 1], ProjPoints[0, 1] - ProjPoints[0, 0]))

    return Angle


def get_pose_from_absolute(IniH, MosaicOrigin, MosaicResolution, K):
    """
        This function approximately computes the 3D pose from 2D planar transformation
    Args:
        IniH (): 2D planar image-to-map transformation
        MosaicOrigin (): 
        MosaicResolution ():
        K (): camera instrinsics

    Returns:
        IniPose : camera poses in 3D
    """
    Origin = np.array([MosaicOrigin['X'], MosaicOrigin['Y'], 0])
    Resolution = MosaicResolution

    W_R_M = RotX(np.pi)
    W_T_M = Origin

    M_R_W = W_R_M.T
    M_T_W = -np.dot(M_R_W, W_T_M)

    C_R_I = RotZ(np.pi / 2)

    Hs = np.array([[Resolution, 0, 0], [0, Resolution, 0], [0, 0, 1]])
    nbrIm = len(IniH)
    IniPose = []
    for no in range(nbrIm):
        H = IniH[no][:]

        Corners = np.array([[1, K[0, 2], K[0, 2], 1],
                            [K[1, 2], K[1, 2], 1, 1],
                            [1, 1, 1, 1]])

        Yaw = np.deg2rad(-compute_angle(H, K[0:2, 2]))

        _, D, _ = np.linalg.svd(H[0:2, 0:2])
        Alt = ((D[0] + D[1]) / 2 * Resolution) / (2 / (K[0, 0] + K[1, 1]))

        ProjCenter = np.dot(H, K[:, 2])
        ProjCenter /= ProjCenter[2]
        utmX = Origin[0] + (ProjCenter[0] * Resolution)
        utmY = Origin[1] - (ProjCenter[1] * Resolution)
        PoseInit = np.array([utmX, utmY, Alt, 0, 0, Yaw])

        Pose = scopt.least_squares(pose_lsq, PoseInit,
                                   args=(K, H, Hs, Corners, Origin, Resolution, C_R_I, M_R_W, M_T_W), verbose=0,
                                   x_scale='jac', xtol=5e-16, ftol=1e-8).x
        IniPose.append(np.array([Pose[0], Pose[1], Pose[2], 0, Pose[3], Pose[4], Pose[5]]))

    return np.array(IniPose)


def pose_lsq(Pose, K, H, Hs, Corners, Origin, Resolution, C_R_I, M_R_W, M_T_W):
    W_R_C = RotX(np.pi) @ RotZ(Pose[5] - np.pi / 2) @ RotY(Pose[4]) @ RotX(
        Pose[3])  # np.dot(np.dot(RotX(np.pi), np.dot(RotZ(Pose[5] - np.pi/2), np.dot(RotY(Pose[4]), RotX(Pose[3])))))
    W_R_I = np.dot(W_R_C, C_R_I)
    W_T_I = np.array([Pose[0], Pose[1], Pose[2]])

    M_R_I = np.dot(M_R_W, W_R_I)
    M_T_I = np.dot(M_R_W, W_T_I) + M_T_W

    I_R_M = M_R_I.T
    I_T_M = -np.dot(I_R_M, M_T_I)

    I_M_M = np.column_stack((I_R_M, I_T_M))
    i_P_M = np.dot(K, I_M_M)

    i_H_m = i_P_M[:, [0, 1, 3]]
    i_H_m = np.dot(i_H_m, Hs)
    i_H_m /= i_H_m[2, 2]

    i_H_i = np.dot(i_H_m, H)
    i_H_i /= i_H_i[2, 2]

    Match2 = np.dot(i_H_i, Corners)

    Match2[0, :] /= Match2[2, :]
    Match2[1, :] /= Match2[2, :]

    r = Corners[:2, :].flatten() - Match2[:2, :].flatten()

    return r


def convert_bundle_to_gmml(wHi, x, MosaicOrigin, MosaicResolution):
    """
        This function converts the globally aligned motion parameters in 3D
        to image-to-map 2D transformations
    Args:
        wHi (): image-to-world transformations
        x (): vector including motion parameters of all images
        MosaicOrigin (): mosaic origin in pixels
        MosaicResolution (): 1 pixel in mm

    Returns:
        HRes : wHi, 
        IniPose : pose parameters for each image
    """
    # Check Number of Nodes
    NumNodes = len(x) // 6
    HRes = [np.eye(3) for _ in range(NumNodes)]
    if NumNodes != wHi.shape[2]:
        raise ValueError("Number of nodes in M and number of homographies must agree!")

    # Setting the altitude to 1 meter (result of optim in meters)
    # MosaicResolution = 1

    # Compute the Transformation 3D Mosaic Frame -> 3D World Frame
    W_R_M = RotX(np.pi)
    W_T_M = np.array([MosaicOrigin['X'], MosaicOrigin['Y'], 0])

    # Compute the Transformation 3D Camera Frame -> 3D Vehicle Frame
    V_R_C = RotZ(np.pi / 2)
    C_R_V = V_R_C.T

    IndexPoseVar = 0
    IniPose = []
    for i in range(NumNodes):
        H = wHi[:, :, i]
        HRes[i][:] = H / H[2, 2]

        vxi = x[IndexPoseVar]
        vyi = x[IndexPoseVar + 1]
        vzi = x[IndexPoseVar + 2]
        wi = (1.0 - vxi * vxi - vyi * vyi - vzi * vzi)
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

        C_R_M = quat2rotmat([wi, vxi, vyi, vzi])
        M_R_C = C_R_M.T
        M_T_C = x[IndexPoseVar + 3:IndexPoseVar + 6]

        M_R_V = np.dot(M_R_C, C_R_V)
        M_T_V = M_T_C

        W_R_V = np.dot(W_R_M, M_R_V)
        W_T_V = np.dot(W_R_M, M_T_V) + W_T_M

        W_R_V = np.dot(RotX(np.pi).T, W_R_V)
        Roll, Pitch, Yawi = RotMat2FixedXYZ(W_R_V)
        Yaw = Yawi + np.pi / 2

        IniPose.append(np.array([*W_T_V, 0, Roll, Pitch, Yaw]))

        IndexPoseVar += 6

    return HRes, IniPose


def convert_bundle_to_data(M, data, KMat, MosaicOrigin):
    """
        This function converts correspondences and initial motion estimate data
        to be run in global alignment 
    Args:
        M (): pose parameters vector for images
        NumPoints (): number of points to be used during global alignment
            per image pair
        data (): Data structure to keep the overlapping image pairs and
            correspondences data
        KMat (): camera instrinsics
        MosaicOrigin (): mosaic origin in pixel

    Returns:
        RTs : rotation and translation in 3D
        K : camera intrinsics
    """
    n_images = data.match_matrix.shape[0]

    K = np.zeros((3, 3, n_images))
    RTs_Struc = {'cRw': np.eye(3), 'wTc': np.zeros((3, 1))}
    RTs = [RTs_Struc.copy() for _ in range(n_images)]

    W_R_M = RotX(np.pi)
    W_T_M = np.array([MosaicOrigin['X'], MosaicOrigin['Y'], 0])
    M_R_W = W_R_M.T
    M_T_W = -M_R_W @ (W_T_M)

    V_R_C = RotZ(np.pi / 2)
    k = 0
    ds = 1
    for img_i in range(n_images):
        Roll = M[img_i][4]
        Pitch = M[img_i][5]
        Yaw = M[img_i][6]

        W_R_V = RotX(np.pi) @ RotZ(Yaw - (np.pi / 2)) @ RotY(Pitch) @ RotX(Roll)

        W_T_V = np.array([M[img_i][0], M[img_i][1], M[img_i][2]])

        W_R_C = W_R_V @ (V_R_C)
        W_T_C = W_T_V

        M_R_C = M_R_W @ (W_R_C)
        M_T_C = M_R_W @ (W_T_C) + M_T_W

        RTs[img_i]['cRw'] = M_R_C.T
        RTs[img_i]['wTc'] = M_T_C

        if not ('KMat' in locals()):
            raise ValueError(f'Dataset n°{ds} Does not have a K matrix!')

        K[:, :, img_i] = KMat

    return RTs, K


def set_resolution(HRes, res):
    HRes = np.array(HRes)
    nbrIm = HRes.shape[0]
    H = np.array([[1 / res, 0, 0], [0, 1 / res, 0], [0, 0, 1]])
    for i in range(nbrIm):
        HRes[i][:] = H @ HRes[i][:]
        # HRes[i][:] = HRes[i][:] @ HH
    return HRes


def RotMat2MobileRPY(R):
    Px = np.sqrt(R[1, 2] ** 2 + R[2, 2] ** 2)

    if Px > 1e-8:
        Pitch = np.arctan2(R[0, 2], Px)

        if (Pitch < np.pi / 2) and (Pitch > -np.pi / 2):
            Yaw = np.arctan2(-R[0, 1], R[0, 0])
            Roll = np.arctan2(-R[1, 2], R[2, 2])
        else:
            Yaw = np.arctan2(R[0, 1], -R[0, 0])
            Roll = np.arctan2(R[1, 2], -R[2, 2])
    else:
        Pitch = np.pi / 2
        Yaw = 0
        Roll = np.arctan2(R[2, 1], R[1, 1])

    return Roll, Pitch, Yaw


def RotMat2FixedXYZ(R):
    Px = np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)

    if Px > 1e-8:
        B = np.arctan2(-R[2, 0], Px)

        if (B < np.pi / 2) and (B > -np.pi / 2):
            A = np.arctan2(R[1, 0], R[0, 0])
            G = np.arctan2(R[2, 1], R[2, 2])
        else:
            A = np.arctan2(-R[1, 0], -R[0, 0])
            G = np.arctan2(-R[2, 1], -R[2, 2])
    else:
        B = np.pi / 2
        A = 0
        G = np.arctan2(-R[1, 2], R[1, 1])

    return G, B, A


@jit(nopython=True, cache=True, parallel=False)
def quat2rotmat(q):
    """

    Args:
        q ():

    Returns:

    """
    R = np.eye(3, dtype=np.float64)
    q0, qx, qy, qz = q
    q02, qx2, qy2, qz2 = q0, qx, qy, qz

    q02 *= q02
    qx2 *= qx2
    qy2 *= qy2
    qz2 *= qz2

    q0x = 2.0 * q0 * qx
    q0y = 2.0 * q0 * qy
    q0z = 2.0 * q0 * qz
    qxy = 2.0 * qx * qy
    qxz = 2.0 * qx * qz
    qyz = 2.0 * qy * qz

    R[0, 0] = q02 + qx2 - qy2 - qz2
    R[0, 1] = -q0z + qxy
    R[0, 2] = q0y + qxz
    R[1, 0] = q0z + qxy
    R[1, 1] = q02 - qx2 + qy2 - qz2
    R[1, 2] = -q0x + qyz
    R[2, 0] = -q0y + qxz
    R[2, 1] = q0x + qyz
    R[2, 2] = q02 - qx2 - qy2 + qz2
    return R


def rotmat2quat(R):
    """

    Args:
        R ():

    Returns:

    """
    q = np.array([0, 0, 0, 0], dtype=float)
    R11, R12, R13, R21, R22, R23, R31, R32, R33 = R[0, 0], R[0, 1], R[0, 2], R[1, 0], R[1, 1], R[1, 2], R[2, 0], R[
        2, 1], R[2, 2]

    q[0] = (1.0 + R11 + R22 + R33) / 4.0
    q[1] = (1.0 + R11 - R22 - R33) / 4.0
    q[2] = (1.0 - R11 + R22 - R33) / 4.0
    q[3] = (1.0 - R11 - R22 + R33) / 4.0

    MaxVal = q[0]
    Index = 0
    i = 1
    while MaxVal < (1.0 / 4.0) and i < 4:
        if q[i] > MaxVal:
            MaxVal = q[i]
            Index = i
        i += 1

    qi = math.sqrt(MaxVal)

    if Index == 0:
        q[0] = qi
        q[1] = ((R32 - R23) / 4.0) / qi
        q[2] = ((R13 - R31) / 4.0) / qi
        q[3] = ((R21 - R12) / 4.0) / qi
    elif Index == 1:
        q[0] = ((R32 - R23) / 4.0) / qi
        q[1] = qi
        q[2] = ((R12 + R21) / 4.0) / qi
        q[3] = ((R13 + R31) / 4.0) / qi
    elif Index == 2:
        q[0] = ((R13 - R31) / 4.0) / qi
        q[1] = ((R12 + R21) / 4.0) / qi
        q[2] = qi
        q[3] = ((R23 + R32) / 4.0) / qi
    elif Index == 3:
        q[0] = ((R21 - R12) / 4.0) / qi
        q[1] = ((R13 + R31) / 4.0) / qi
        q[2] = ((R23 + R32) / 4.0) / qi
        q[3] = qi
    return q
