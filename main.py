#!/usr/bin/env python
from datasets import KITTIDataset
from models.FlownetSimpleLike import FlowNetS, RMSEWeightedLoss
from torch.utils.data import DataLoader
import os
import cv2
import torch
import argparse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def main():
    parser = argparse.ArgumentParser(description='JustDrop video run')
    parser.add_argument('--kitti-base-dir', type=str,
                        default='../dataset',
                        help='json config file, if provided - overrides args with info provided in that file')
    parser.add_argument('--batch-size', type=int,
                        default=10)
    parser.add_argument('--learning-rate', type=float,
                        default=0.0025)
    parser.add_argument('--pretrained-flownet', type=str,
                        default="")

    args = parser.parse_args()

    dataset = KITTIDataset(args.kitti_base_dir, 1)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True
    )
    model = FlowNetS()

    if args.pretrained_flownet:
        pretrained_w = torch.load(args.pretrained_flownet)

        print('Load FlowNet pretrained model')
        # Use only conv-layer-part of FlowNet as CNN for DeepVO
        model_dict = model.state_dict()
        update_dict = {k: v for k, v in pretrained_w.items() if k in model_dict}
        model_dict.update(update_dict)
        model.load_state_dict(model_dict)

    learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss = RMSEWeightedLoss()

    print("Dataset loaded")
    step = 0
    for batch in dataloader:
        # Shape: 2 10 1 376 1241
        cam0_img, cam1_img, ground_truth = batch
        #print(cam0_img.shape, cam1_img.shape, ground_truth.shape)

        print("Forward pass")
        y = model(torch.cat((cam0_img, cam1_img), 1))

        batch_loss = loss(y, ground_truth)
        print(batch_loss)
        # zero the parameter gradients
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        step += 1


if __name__ == '__main__':
    main()
