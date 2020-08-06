from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from pykitti import odometry
from typing import Tuple
from scipy.spatial.transform import Rotation
import numpy as np
import math
import torch

class KITTIDataset(Dataset):
    def __init__(self, base_path: str, sequence: int):
        self._dataset = odometry(base_path, '{:02}'.format(sequence))

    def __len__(self) -> int:
        return len(self._dataset.cam0_files)-1

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        raw_odometry_matrix =self._dataset.poses[index][:3,:4]
        # Rotation angles for x and z are replaced compared to referent code
        # TODO: Interpretability  of axis needs to be investigated
        preprocessed_ground_truth = self.preprocess_odometry_matrix(raw_odometry_matrix)
        preprocessed_ground_truth2 = self.preprocess_odometry_matrix(self._dataset.poses[index+1][:3,:4])

        # Checking if implementation is same as referent
        # R_to_angle(raw_odometry_matrix)
        # print(preprocessed_ground_truth)
        # print("_______________")
        # Return two consecutive images
        return (
            pil_to_tensor(self._dataset.get_cam0(index))/255.0,
            pil_to_tensor(self._dataset.get_cam0(index+1))/255.0,
            preprocessed_ground_truth2-preprocessed_ground_truth
        )
        # return (
        #     pil_to_tensor(self._dataset.get_cam0(index)),
        #     pil_to_tensor(self._dataset.get_cam1(index)),
        #     preprocessed_ground_truth
        # )

    @staticmethod
    def preprocess_odometry_matrix(odometry_matrix):
        """
        Rotation matrix to Euler angles
        Return [float]: Angles and translation
        :return:
        """
        rotation = Rotation.from_matrix(odometry_matrix[0:3,0:3])
        translation = odometry_matrix[:,3]
        # x and z are replaced compared to original
        rotation_angles = rotation.as_euler('zxy', degrees=False)
        preprocessed_output = np.concatenate((rotation_angles.flatten(), translation.flatten()))
        return torch.from_numpy(preprocessed_output)


# Taken from https://github.com/ChiWeiHsiao/DeepVO-pytorch
# To validate that implementation of preprocessing is valid - without having to mess around with coordinate systems etc.

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def R_to_angle(Rt):
    # Ground truth pose is present as [R | t]
    # R: Rotation Matrix, t: translation vector
    # transform matrix to angles
    t = Rt[:, -1]
    R = Rt[:, :3]

    assert (isRotationMatrix(R))

    x, y, z = euler_from_matrix(R)
    print(x, y, z)
    theta = [x, y, z]
    pose_15 = np.concatenate((theta, t, R.flatten()))
    assert (pose_15.shape == (15,))
    return pose_15


def euler_from_matrix(matrix):
    # y-x-z Tait–Bryan angles intrincic
    # the method code is taken from https://github.com/awesomebytes/delta_robot/blob/master/src/transformations.py

    i = 2
    j = 0
    k = 1
    repetition = 0
    frame = 1
    parity = 0

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > _EPS:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > _EPS:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az