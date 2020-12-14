from utils import common_utils, display_utils
import torch
import trimesh
import numpy as np
import os
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
import os.path as osp
from torch.autograd import Variable
from django.conf import settings
from tqdm import tqdm
import torch.optim as optim
from utils.common_utils import *
from scipy.spatial.transform import Rotation as R
import cv2


os.environ['DJANGO_SETTINGS_MODULE'] = 'NIAAnnotationTools.settings'
pose_size = 72
beta_size = 10


def get_init_joint_2d(intrinsics, extrinsics, gender):
    pose_params = torch.zeros(pose_size)
    shape_params = torch.rand(beta_size)
    trans_params = torch.tensor([0, 1, 0])

    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender=gender,
        model_root=settings.SMPL_MODEL_PATH)
    smpl_params = torch.Tensor(torch.cat((pose_params, shape_params, trans_params)).type(torch.float32))
    verts, Jtr = smpl_layer(torch.unsqueeze(smpl_params[:pose_size], 0),
                            th_betas=torch.unsqueeze(smpl_params[pose_size:pose_size+beta_size], 0),
                            th_trans=torch.unsqueeze(smpl_params[pose_size+beta_size:], 0))
    cam_param = common_utils.get_camera_info(intrinsics, extrinsics)
    pred_joint2d = common_utils.projection_torch(vertices=Jtr[0], cam_param=cam_param, width=1920, height=1080)
    return pred_joint2d.numpy()

def run_model_single(gender, pose_params, shape_params, trans_params):
    smpl_params = torch.Tensor(torch.cat((pose_params, shape_params, trans_params)).type(torch.float32))

    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender=gender,
        model_root=settings.SMPL_MODEL_PATH)
    verts, Jtr = smpl_layer(torch.unsqueeze(smpl_params[:pose_size], 0),
                            th_betas=torch.unsqueeze(smpl_params[pose_size:pose_size+beta_size], 0),
                            th_trans=torch.unsqueeze(smpl_params[pose_size+beta_size:], 0))
    return verts.numpy()[0], smpl_layer.th_faces.numpy()

def run_model(gt_dataset, gender, epochs=10):
    # parameter setting
    pose_params = torch.zeros(pose_size)
    shape_params = torch.rand(beta_size)
    trans_params = torch.tensor([0, 1, 0])
    # trans_params = torch.zeros(3)
    # trans_params = torch.zeros(3)
    smpl_params = Variable(torch.cat((pose_params, shape_params, trans_params)).type(torch.float32), requires_grad=True)

    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender=gender,
        model_root=settings.SMPL_MODEL_PATH)
    # verts, Jtr = smpl_layer(pose_params, th_betas=shape_params, th_trans=trans_params)
    # mesh = trimesh.Trimesh(vertice, faces)
    # display_utils.save_obj(mesh.vertices, mesh.faces, 'test.obj')
    datasets = []
    for gt_joint2d, intrinsics, extrinsics in gt_dataset:
        datasets.append({'joint2d': torch.Tensor(gt_joint2d), 'intrinsics': torch.Tensor(intrinsics), 'extrinsics': torch.Tensor(extrinsics)})

    # hyper parameter
    learning_rate = 1e-1
    optimizer = optim.Adam([smpl_params], lr=learning_rate)

    for epoch in tqdm(range(epochs)):
        for dataset in datasets:
            verts, Jtr = smpl_layer(torch.unsqueeze(smpl_params[:pose_size], 0),
                                    th_betas=torch.unsqueeze(smpl_params[pose_size:pose_size+beta_size], 0),
                                    th_trans=torch.unsqueeze(smpl_params[pose_size+beta_size:], 0))
            gt_joint2d = dataset['joint2d']
            intrinsics = dataset['intrinsics']
            extrinsics = dataset['extrinsics']
            cam_param = common_utils.get_camera_info(intrinsics, extrinsics)
            pred_joint2d = common_utils.projection_torch(vertices=Jtr[0], cam_param=cam_param, width=1920, height=1080)
            loss = (pred_joint2d - gt_joint2d).pow(2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    pose_t = torch.unsqueeze(smpl_params[:pose_size], 0).reshape(-1, 3).detach().numpy()
    rotation = []
    for pose in pose_t:
        r = R.from_rotvec(pose)
        rotation.append(r.as_euler('xyz', degrees=True).tolist())
    # th_pose_rotmat = th_posemap_axisang(pose_t).reshape(-1, 9)
    # pose_t = th_pose_rotmat.detach().numpy()
    result = {
        'rotation': rotation,
        'shape_params': smpl_params.detach().numpy()[pose_size:pose_size+beta_size].tolist(),
        'trans_params': smpl_params.detach().numpy()[pose_size+beta_size:].tolist(),
        'joint_3d': Jtr[0].detach().numpy().squeeze().tolist()
    }
    return verts, Jtr, result
