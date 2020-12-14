import math
import numpy as np
import cv2
import logging
import os
import os.path as osp
import torch

def ndarray2tuple(value):
    return tuple(value.astype(np.int64).tolist())

def get_projection_matrix(focal_length, principal_point, width, height, znear=0.1, zfar=1000):
    """Return the OpenGL projection matrix for this camera.

    Parameters
    ----------
    width : int
        Width of the current viewport, in pixels.
    height : int
        Height of the current viewport, in pixels.
    """
    fx, fy = focal_length[0], focal_length[1]
    cx, cy = principal_point[0], principal_point[1]
    width = float(width)
    height = float(height)

    P = torch.zeros((4,4))
    P[0][0] = 2.0 * fx / width
    P[1][1] = 2.0 * fy / height
    P[0][2] = 1.0 - 2.0 * cx / (width - 1.0)
    P[1][2] = 2.0 * cy / (height - 1.0) - 1.0
    P[3][2] = -1.0

    n = znear
    f = zfar
    if f is None:
        P[2][2] = -1.0
        P[2][3] = -2.0 * n
    else:
        P[2][2] = (f + n) / (n - f)
        P[2][3] = (2 * f * n) / (n - f)

    return P

def projection_torch(vertices, cam_param, width, height):
    extrinsics = cam_param['extrinsics']
    focal = cam_param['focal']
    princpt = cam_param['princpt']

    # ModelView Matrix (Object Coordinate -> Eye Coordinate)
    qw = torch.unsqueeze(torch.ones(vertices.shape[0]), 0).T
    vertices = torch.hstack((vertices, qw))
    eye_coord = torch.einsum('ij,kj->ki', extrinsics, vertices)
    # Projection Matrix (Eye Coordinate -> Clip Coordinate)
    projection_matrix = get_projection_matrix(focal, princpt, width, height)
    clip_coord = torch.einsum('ij,kj->ki', projection_matrix, eye_coord)
    # Divide by w (Clip Coordinate -> Normalized Device Coordinate)
    ndc_coord = clip_coord / torch.unsqueeze(clip_coord[:,-1], 0).reshape(-1, 1)
    # View Transform (Normalized Device Coordinate -> Window Coordinate)
    winndow_coord = (ndc_coord[:,:2] + 1) * torch.tensor([width, height]) / 2 - 1
    # flip for opencv
    window_coord = torch.tensor([0, height]) + winndow_coord*torch.tensor([1, -1])

    return window_coord

def projection(vertices, cam_param, width, height):
    """Return 2D-position calculated with camera matrix like OpenGL
    Object Coordinates -> Eye Coordinate -> Clip Coordinate ->
    Normalized Device Coordinate -> Window Coordinate -> OpenCV Coordinate

    Parameters
    ----------
    vertices : numpy.array
        Vertices 3D-position for projection
        shape : (-1, 3)
    cam_param : dict
        Camera parameter info(focal, princpt, extrinsic matrix)
        - focal : focal length
        - princpt : principal point
    width : int
        Image width (pixel)
    height : int
        Image height (pixel)
    Returns
    -------
    numpy.array
        projected 2D-position

    References
    -------
    http://www.songho.ca/opengl/gl_transform.html
    """
    extrinsics = cam_param['extrinsics']
    focal = cam_param['focal']
    princpt = cam_param['princpt']

    # ModelView Matrix (Object Coordinate -> Eye Coordinate)
    qw = np.expand_dims(np.ones(vertices.shape[0]), axis=0).T
    vertices = np.hstack((vertices, qw))
    eye_coord = np.einsum('ij,kj->ki', extrinsics, vertices)
    # Projection Matrix (Eye Coordinate -> Clip Coordinate)
    projection_matrix = get_projection_matrix(focal, princpt, width, height)
    clip_coord = np.einsum('ij,kj->ki', projection_matrix, eye_coord)
    # Divide by w (Clip Coordinate -> Normalized Device Coordinate)
    ndc_coord = clip_coord / np.expand_dims(clip_coord[:,-1], axis=0).reshape(-1, 1)
    # View Transform (Normalized Device Coordinate -> Window Coordinate)
    winndow_coord = (ndc_coord[:,:2] + 1) * np.array([width, height]) / 2 - 1
    # flip for opencv
    window_coord = np.array([0, height]) + winndow_coord*np.array([1, -1])

    return window_coord

def get_camera_info(intrinsics, extrinsics):
    """Get Camera infomation from intrinsic matrix and extrinsic matrix

    Parameters
    ----------
    intrinsics : torch.tensor
        camera intrinsic matrix
    extrinsics : torch.tensor
        camera extrinsic matrix
    Returns
    -------
    dict
        cam_param : dict
            Camera parameter info(focal, princpt, extrinsic matrix)
            - focal : focal length
            - princpt : principal point
    """
    princpt = intrinsics[:2, 2] * 1920
    focal = torch.diag(intrinsics[:2, :2]) * 1920
    extrinsics = torch.vstack((extrinsics, torch.tensor([0, 0, 0, 1])))
    extrinsics_inv = torch.inverse(extrinsics)
    extrinsics_inv[:3,3] *= 0.001
    extrinsics_inv[:, 1:3] *= -1
    extrinsics = torch.inverse(extrinsics_inv)
    # make camera parameter info
    cam_param = {
        'focal': focal,
        'princpt': princpt,
        'extrinsics': extrinsics
    }
    return cam_param

def rotationMatrixToEulerAngles(R) :
    if R.shape == (9,):
        R = R.reshape(3, 3).transpose(0, 1)
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])


def eulerAnglesToRotationMatrix(theta) :
    """ Change euler-angle to axis-angle (rotation vector)
    """
    theta = [math.radians(theta[0]), math.radians(theta[1]), math.radians(theta[2])]
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    rvec, _ = cv2.Rodrigues(R)

    return rvec.flatten()

