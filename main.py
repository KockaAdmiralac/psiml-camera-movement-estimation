#!/usr/bin/env python
from datasets import KITTIDataset
from torch.utils.data import DataLoader
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

KITTI_BASE_DIR = '../dataset'
BATCH_SIZE = 10

def main():
    dataset = KITTIDataset(KITTI_BASE_DIR, 1)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True
    )
    print("Dataset loaded")
    for batch in dataloader:
        # Shape: 2 10 1 376 1241
        cam0_img, cam1_img, ground_truth = batch
        #print(cam0_img.shape, cam1_img.shape, ground_truth.shape)
        pass

if __name__ == '__main__':
    main()
