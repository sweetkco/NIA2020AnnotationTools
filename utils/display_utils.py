import numpy as np
import os.path as osp
import os
import cv2
import matplotlib.pyplot as plt
from utils.common_utils import *


def display_model2(visimg, vertices, intrinsics, extrinsics, width, height, colors=None, circle_size=1):
    # matrix preprocessing
    cam_param = get_camera_info(intrinsics, extrinsics)
    projected_vertices = projection_torch(vertices=vertices, cam_param=cam_param, width=width, height=height).numpy()
    if colors is None:
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(vertices))]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    for i, joint in enumerate(projected_vertices):
        x, y = ndarray2tuple(joint)
        visimg = cv2.circle(visimg, (x, y), radius=circle_size, color=colors[i%len(colors)], thickness=-1, lineType=cv2.LINE_AA)
    return visimg

def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()