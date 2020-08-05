from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from pykitti import odometry
from typing import Tuple
import torch

class KITTIDataset(Dataset):
    def __init__(self, base_path: str, sequence: int):
        self._dataset = odometry(base_path, '{:02}'.format(sequence))

    def __len__(self) -> int:
        return len(self._dataset.cam0_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            pil_to_tensor(self._dataset.get_cam0(index)),
            pil_to_tensor(self._dataset.get_cam1(index))
        )
