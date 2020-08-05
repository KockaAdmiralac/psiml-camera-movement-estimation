#!/usr/bin/env python
from datasets import KITTIDataset
from torch.utils.data import DataLoader

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
    for batch in dataloader:
        # Shape: 2 10 1 376 1241
        print(batch)

if __name__ == '__main__':
    main()
