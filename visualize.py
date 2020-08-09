#!/usr/bin/env python
from argparse import ArgumentParser
from datasets import KITTIDataset
from io import TextIOWrapper
from models.FlownetSimpleLikeV2 import FlowNetS_V2
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader
from typing import Tuple
import numpy as np
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def get_args():
    parser = ArgumentParser(description='Camera movement estimation visualizer')
    parser.add_argument('-s', '--sequence', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=10)
    parser.add_argument('--kitti-base-dir', type=str, default='../dataset')
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('--validation-size', type=int, default=1000)
    parser.add_argument('-p', '--poses-path', type=str, default='.')
    return parser.parse_args()

def get_model(use_cuda: bool, model_path: str) -> FlowNetS_V2:
    model = FlowNetS_V2()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    if use_cuda:
        print('CUDA used.')
        model = model.cuda()
    model.eval()
    return model

def write_poses(poses_file: TextIOWrapper, points: np.ndarray, rotations: np.ndarray,
                use_cuda: bool, model: FlowNetS_V2, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
    cam0_img, cam1_img, _, _ = batch
    input_tensor = torch.cat((cam0_img, cam1_img), 1)
    if use_cuda:
        input_tensor = input_tensor.cuda()
    y = model(input_tensor)
    y = y.to('cpu')
    rotation = sum(y[:, 0:3]).numpy()
    translation = sum(y[:, 3:6]).numpy()
    r = Rotation.from_euler('zxy', rotations, degrees=False)
    points += r.apply(translation)
    rotations += rotation
    rotation_matrix = Rotation.from_euler('zxy', rotations, degrees=False).as_matrix().reshape(9)
    concatenated_list = list(np.concatenate((rotation_matrix, points)))
    poses_file.write(f"{' '.join([str(a) for a in concatenated_list])}\n")
    poses_file.flush()

def main():
    args = get_args()
    use_cuda = torch.cuda.is_available()
    model = get_model(use_cuda, args.model)
    points = np.array([0., 0., 0.])
    rotations = np.array([0., 0., 0.])
    dataset = KITTIDataset(args.kitti_base_dir, [args.sequence])
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True
    )
    with open(os.path.join(args.poses_path, 'poses.txt'), 'w') as poses_file:
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                write_poses(poses_file, points, rotations, use_cuda, model, batch)

if __name__ == '__main__':
    main()
