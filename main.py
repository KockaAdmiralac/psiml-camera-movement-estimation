#!/usr/bin/env python
from datasets import KITTIDataset
from models.FlownetSimpleLike import FlowNetS, RMSEWeightedLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import cv2
import torch
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
cuda = torch.device('cuda')


def main():
    parser = argparse.ArgumentParser(description='JustDrop video run')
    parser.add_argument('--kitti-base-dir', type=str,
                        default='../dataset',
                        help='json config file, if provided - overrides args with info provided in that file')
    parser.add_argument('--batch-size', type=int,
                        default=10)
    parser.add_argument('--validation-size', type=int,
                        default=1000)
    parser.add_argument('--epochs', type=int,
                        default=40)
    parser.add_argument('--learning-rate', type=float,
                        default=0.0025)
    parser.add_argument('--pretrained-flownet', type=str,
                        default="")
    parser.add_argument("--model-tag", type=str, default="initial_model")
    parser.add_argument('--output-path', type=str,
                        default="trained-models/")

    args = parser.parse_args()

    summary_writer = SummaryWriter()

    # Train set
    dataset = KITTIDataset(args.kitti_base_dir, 1)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True
    )
    # Validation set:
    validation_dataset = KITTIDataset(args.kitti_base_dir, 2)
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=1,
        shuffle=True,
        drop_last=True
    )
    model = FlowNetS()

    if args.pretrained_flownet:
        pretrained_w = torch.load(args.pretrained_flownet, map_location='cpu')

        print('Load FlowNet pretrained model')
        # Use only conv-layer-part of FlowNet as CNN for DeepVO
        model_dict = model.state_dict()
        update_dict = {k: v for k, v in pretrained_w.items() if k in model_dict}
        model_dict.update(update_dict)
        model.load_state_dict(model_dict)
        del pretrained_w
        del update_dict

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('CUDA used.')
        model = model.cuda()

    learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss = RMSEWeightedLoss()

    global_step = 0

    for epoch in range(args.epochs):
        print("epoch {}".format(epoch))
        epoch_steps = 0
        total_loss = 0
        for batch in dataloader:
            cam0_img, cam1_img, ground_truth = batch
            # print(cam0_img.shape, cam1_img.shape, ground_truth.shape)

            input_tensor = torch.cat((cam0_img, cam1_img), 1)
            if use_cuda:
                input_tensor = input_tensor.cuda()
            y = model(input_tensor)

            y = y.to("cpu")
            batch_loss = loss(y, ground_truth)
            total_loss += batch_loss

            summary_writer.add_scalar("Batch loss", batch_loss, global_step)
            # zero the parameter gradients
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            global_step += 1
            epoch_steps += 1

        print("Total avg epoch loss: {}".format(total_loss/epoch_steps))
        summary_writer.add_scalar("Epoch average loss", total_loss/epoch_steps, global_step)
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        torch.save(model.state_dict(), os.path.join(args.output_path, args.model_tag+"_e"+str(epoch)+".model"))

        # Validation at the end of epoch
        with torch.no_grad():
            validation_step = 0
            validation_loss = 0
            for batch in validation_dataloader:
                if validation_step >= args.validation_size:
                    break
                cam0_img, cam1_img, ground_truth = batch

                input_tensor = torch.cat((cam0_img, cam1_img), 1)
                if use_cuda:
                    input_tensor = input_tensor.cuda()
                y = model(input_tensor)

                y = y.to("cpu")
                batch_loss = loss(y, ground_truth)

                validation_loss += batch_loss
                validation_step += 1

            summary_writer.add_scalar("Val average loss", validation_loss/validation_step, global_step)
            print("Validation avg loss: {}".format(validation_loss/validation_step))
            # TODO: Add random validation images


if __name__ == '__main__':
    main()
