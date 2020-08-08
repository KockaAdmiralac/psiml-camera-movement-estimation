#!/usr/bin/env python
from argparse import ArgumentParser
from math import degrees
from os import truncate
from datasets import KITTIDataset
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from models.FlownetSimpleLike import FlowNetS
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.proj3d import proj_transform
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader
from typing import Tuple
import math
import numpy as np
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c
class Arrow3D(FancyArrowPatch):
    def __init__(self, xyz: np.ndarray, dxdydz: np.ndarray, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._xyz = tuple(xyz)
        self._dxdydz = tuple(dxdydz)

    def draw(self, renderer):
        x1,y1,z1 = self._xyz
        dx,dy,dz = self._dxdydz
        x2,y2,z2 = (x1+dx,y1+dy,z1+dz)

        xs, ys, zs = proj_transform((x1,x2),(y1,y2),(z1,z2), renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        super().draw(renderer)

def main():
    parser = ArgumentParser(description='Camera movement estimation visualizer')
    parser.add_argument('-s', '--sequences', nargs='+', type=int, default=[1, 2])
    parser.add_argument('-b', '--batch-size', type=int, default=10)
    parser.add_argument('--kitti-base-dir', type=str, default='../dataset')
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('--validation-size', type=int, default=1000)
    args = parser.parse_args()
    fig = plt.figure()
    use_cuda = torch.cuda.is_available()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('xyz')
    ax.set_proj_type('ortho')
    kitti = [[], [], []]
    our = [[], [], []]
    model = FlowNetS()
    model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
    if use_cuda:
        print('CUDA used.')
        model = model.cuda()
    model.eval()
    starting_points = [np.array([0., 0., 0.]) for _ in args.sequences]
    starting_rotations = [np.array([0., 0., 0.]) for _ in args.sequences]
    starting_points_estimate = [np.array([0., 0., 0.]) for _ in args.sequences]
    starting_rotations_estimate = [np.array([0., 0., 0.]) for _ in args.sequences]
    ax.scatter3D([0], [0], [0])
    dataset = KITTIDataset(args.kitti_base_dir, args.sequences)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True
    )
    i = 0
    validation_step = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if args.validation_size >= 0 and validation_step >= args.validation_size:
                break
            cam0_img, cam1_img, ground_truth = batch
            input_tensor = torch.cat((cam0_img, cam1_img), 1)
            if use_cuda:
                input_tensor = input_tensor.cuda()
            y = model(input_tensor)
            y = y.to("cpu")
            rotation = sum(ground_truth[:, 0:3]).numpy()
            rotation_estimate = sum(y[:, 0:3]).numpy()
            translation = sum(ground_truth[:, 3:6]).numpy()
            translation_estimate = sum(y[:, 3:6]).numpy()
            dataset_idx = dataset.dataset_idx(i)
            p = starting_points[dataset_idx].copy()
            pe = starting_points_estimate[dataset_idx].copy()
            starting_points[dataset_idx] += translation
            starting_points_estimate[dataset_idx] += translation_estimate
            r = Rotation.from_euler('zxy', starting_rotations[dataset_idx], degrees=False)
            re = Rotation.from_euler('zxy', starting_rotations_estimate[dataset_idx], degrees=False)
            camera_point = r.apply(starting_points[dataset_idx])
            camera_point_estimate = re.apply(starting_points_estimate[dataset_idx])
            ax.add_artist(Arrow3D(p, camera_point, arrowstyle='-|>'))
            ax.add_artist(Arrow3D(pe, camera_point_estimate, arrowstyle='-|>'))
            starting_rotations[dataset_idx] += rotation
            starting_rotations_estimate[dataset_idx] += rotation_estimate
            i += len(batch)
            kitti[0].append(starting_points[dataset_idx][0])
            kitti[1].append(starting_points[dataset_idx][1])
            kitti[2].append(starting_points[dataset_idx][2])
            our[0].append(starting_points_estimate[dataset_idx][0])
            our[1].append(starting_points_estimate[dataset_idx][1])
            our[2].append(starting_points_estimate[dataset_idx][2])
            ax.plot3D(*kitti)
            ax.plot3D(*our)
            print(i)
    ax.plot3D(*kitti)
    ax.plot3D(*our)
    plt.show()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        plt.show()
