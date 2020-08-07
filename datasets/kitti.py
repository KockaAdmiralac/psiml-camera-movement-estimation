from torch.utils.data import Dataset
from torchvision.transforms.functional import center_crop, normalize, to_tensor
from pykitti import odometry
from typing import Tuple
from scipy.spatial.transform import Rotation
import numpy as np
import PIL
import torch

RESIZE_SIZE = [184, 608]
MEAN = (-0.14968217427134656, -0.12941663107068363, -0.1320610301921484)
STD = (1, 1, 1)

class KITTIDataset(Dataset):
    def __init__(self, base_path: str, sequence: int):
        self._dataset = odometry(base_path, '{:02}'.format(sequence))

    def __len__(self) -> int:
        return len(self._dataset.cam2_files) - 1

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_odometry_matrix_1 = self._dataset.poses[index][:3,:4]
        raw_odometry_matrix_2 = self._dataset.poses[index+1][:3,:4]

        # Rotation angles for x and z are replaced compared to referent code
        preprocessed_ground_truth_1 = self.preprocess_odometry_matrix(raw_odometry_matrix_1)
        preprocessed_ground_truth_2 = self.preprocess_odometry_matrix(raw_odometry_matrix_2)

        # Return two consecutive images
        return (
            self.preprocess_image(self._dataset.get_cam2(index)),
            self.preprocess_image(self._dataset.get_cam2(index+1)),
            preprocessed_ground_truth_2 - preprocessed_ground_truth_1
        )

    @staticmethod
    def preprocess_image(image: PIL.Image) -> torch.Tensor:
        image = center_crop(image, RESIZE_SIZE)
        # or resize()
        image = to_tensor(np.array(image))
        image = normalize(image, MEAN, STD)
        return image

    @staticmethod
    def preprocess_odometry_matrix(odometry_matrix: np.ndarray) -> torch.Tensor:
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
