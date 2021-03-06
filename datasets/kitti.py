from math import degrees
from torch.utils.data import Dataset
from torchvision.transforms.functional import center_crop, normalize, to_tensor, resize
from pykitti import odometry
from typing import Dict, List, Tuple
from scipy.spatial.transform import Rotation
import numpy as np
import PIL
import torch

RESIZE_SIZE = [184, 608]
MEAN = (-0.14968217427134656, -0.12941663107068363, -0.1320610301921484)
STD = (1, 1, 1)

class KITTIDataset(Dataset):

    def __init__(self, base_path: str, sequences: List[int], create_stereo=False):
        self.create_stereo = create_stereo
        # Maps index // batch_size to the dataset under that index
        self.idx_to_dataset = []
        # Length of the whole dataset (all sequences summed, rounded by batch_size)
        self.length: int = 0
        # All loaded sequences
        self.sequences: List[int]

        self._datasets = [odometry(base_path, '{:02}'.format(sequence)) for sequence in sequences]
        for dataset_index, dataset in enumerate(self._datasets):

            self.length += len(dataset.cam2_files) - 1
            for i in range(len(dataset.cam2_files) - 1):
                self.idx_to_dataset.append(dataset_index)

        # print(self.length)
    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_dataset_index = self.idx_to_dataset[index]
        current_dataset = self._datasets[current_dataset_index]
        index_in_dataset = index
        for i in range(current_dataset_index):
            index_in_dataset -= len(self._datasets[i].cam2_files)-1

        raw_odometry_matrix_1 = current_dataset.poses[index_in_dataset][:3, :4]
        raw_odometry_matrix_2 = current_dataset.poses[index_in_dataset + 1][:3, :4]

        # Rotation angles for x and z are replaced compared to referent code
        preprocessed_ground_truth_1 = self.preprocess_odometry_matrix(raw_odometry_matrix_1)
        preprocessed_ground_truth_2 = self.preprocess_odometry_matrix(raw_odometry_matrix_2)

        ground_truth = preprocessed_ground_truth_2 - preprocessed_ground_truth_1
        initial_raw_oddometry_matrix = current_dataset.poses[0][:3, :4]
        initial_mat = self.preprocess_odometry_matrix(initial_raw_oddometry_matrix)

        r = Rotation.from_euler('zxy', -(preprocessed_ground_truth_1[:3]-initial_mat[:3]), degrees=False)

        #ground_truth[3:] = torch.from_numpy((raw_odometry_matrix_2[:3,:3]).dot(ground_truth[3:]))
        ground_truth[3:] = torch.from_numpy(r.apply(ground_truth[3:]))
        # TODO: Make configurable

        if self.create_stereo:
            image_stack = torch.cat((self.preprocess_image(current_dataset.get_cam2(index_in_dataset)),
                                     self.preprocess_image(current_dataset.get_cam2(index_in_dataset + 1)),
                                     self.preprocess_image(current_dataset.get_cam3(index_in_dataset)),
                                     self.preprocess_image(current_dataset.get_cam3(index_in_dataset + 1))), 0)
        else:
            image_stack = torch.cat((self.preprocess_image(current_dataset.get_cam2(index_in_dataset)),
                                     self.preprocess_image(current_dataset.get_cam2(index_in_dataset + 1))), 0)
        return (
            image_stack,
            ground_truth, raw_odometry_matrix_2, initial_raw_oddometry_matrix
        )

    def dataset_idx(self, idx: int) -> int:
        return self.idx_to_dataset[idx]

    @staticmethod
    def preprocess_image(image: PIL.Image) -> torch.Tensor:
        # image = center_crop(image, RESIZE_SIZE)
        image = resize(image, RESIZE_SIZE)
        # or resize()
        image = to_tensor(np.array(image))
        image = normalize(image, MEAN, STD)
        return image

    @staticmethod
    def preprocess_odometry_matrix(odometry_matrix: np.ndarray) -> torch.Tensor:
        """
        Rotation matrix to Euler angles
        Return [torch.Tensor]: Angles and translation
        :return:
        """
        rotation = Rotation.from_matrix(odometry_matrix[0:3, 0:3])
        translation = odometry_matrix[:, 3]

        # x and z are replaced compared to original
        rotation_angles = rotation.as_euler('zxy', degrees=False)

        preprocessed_output = np.concatenate((rotation_angles.flatten(), translation.flatten()))
        return torch.from_numpy(preprocessed_output)


