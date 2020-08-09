#!/usr/bin/env python
from datasets import KITTIDataset
from models.FlownetSimpleLike import FlowNetS, RMSEWeightedLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import numpy as np
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
    parser.add_argument('--tboard-step', type=int,
                        default=50)
    parser.add_argument('--save-model-step', type=int,
                        default=200)
    parser.add_argument('--learning-rate', type=float,
                        default=0.0005) #0.0005
    parser.add_argument('--pretrained-flownet', type=str,
                        default="")
    parser.add_argument("--model-tag", type=str, default="initial_model")
    parser.add_argument('--output-path', type=str,
                        default="trained-models/")

    args = parser.parse_args()

    summary_writer = SummaryWriter()

    # Train set
    dataset = KITTIDataset(args.kitti_base_dir, [0, 2, 5, 6, 7, 8, 9, 10]) #
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )
    # Validation set:
    validation_dataset = KITTIDataset(args.kitti_base_dir, [1,3,4]) #, 3, 4
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=1,
        shuffle=True,
        drop_last=True
    )
    #model = FlowNetS()
    from models.FlownetSimpleLikeV2 import FlowNetS_V2
    model = FlowNetS_V2()

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
    #lr_sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    global_step = 0

    for epoch in range(args.epochs):
        print("Epoch:{}".format(epoch))
        epoch_steps = 0
        total_loss = 0
        raw_output_diff = torch.zeros(6)
        for batch in dataloader:
            model.train()
            cam0_img, cam1_img, ground_truth,_ = batch

            input_tensor = torch.cat((cam0_img, cam1_img), 1)
            if use_cuda:
                input_tensor = input_tensor.cuda()
            y = model(input_tensor)

            y = y.to("cpu")
            batch_loss = loss(y, ground_truth)
            total_loss += batch_loss
            raw_output_diff += torch.mean(torch.abs(y - ground_truth), dim=0)
            if global_step > 0 and epoch_steps > 0 and global_step % args.tboard_step == 0:
                summary_writer.add_scalar("train_loss/Batch loss", batch_loss, global_step)
                print("Step {} avg epoch loss: {}".format(global_step, total_loss / epoch_steps))
                print("Step {} avg raw diff: {}".format(global_step, raw_output_diff / epoch_steps))

                for i in range(6):
                    if i < 3:
                        summary_writer.add_scalar("train_raw/Average Train RotAngle{} degrees diff".format(i),
                                                  raw_output_diff[i] / np.pi*180 / epoch_steps, global_step)
                    else:
                        summary_writer.add_scalar("train_raw/Average Train TransVector{} diff".format(i - 3),
                                                  raw_output_diff[i] / epoch_steps,
                                                  global_step)
                    summary_writer.add_scalar("train_loss/Epoch average loss", total_loss / epoch_steps, global_step)
                val_loss = do_validation(validation_dataloader, model, args.validation_size, loss, use_cuda, global_step, summary_writer)
            # zero the parameter gradients
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            global_step += 1
            epoch_steps += 1

            if global_step % args.save_model_step == 0:
                if not os.path.exists(args.output_path):
                    os.makedirs(args.output_path)
                torch.save(model.state_dict(),
                           os.path.join(args.output_path,
                                        args.model_tag + "_e" + str(epoch) + "_gstep" + str(global_step) + ".model"))

        for i in range(6):
            if i < 3:
                summary_writer.add_scalar("train_raw/Epoch Train RotAngle{} degrees diff".format(i),
                                          raw_output_diff[i] / np.pi * 180 / epoch_steps, global_step)
            else:
                summary_writer.add_scalar("train_raw/Epoch Train TransVector{} diff".format(i - 3),
                                          raw_output_diff[i] / epoch_steps,
                                          global_step)
            summary_writer.add_scalar("train_loss/Epoch average loss", total_loss / epoch_steps, global_step)
        val_loss = do_validation(validation_dataloader, model, -1, loss, use_cuda, global_step, summary_writer, True)

        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        torch.save(model.state_dict(), os.path.join(args.output_path, args.model_tag + "_e" + str(epoch) + ".model"))

        do_validation(validation_dataloader, model, args.validation_size, loss, use_cuda, global_step, summary_writer)


def do_validation(validation_dataloader, model, validation_size, loss, use_cuda, global_step, summary_writer, special=False):
    model.eval()
    with torch.no_grad():
        validation_step = 0
        validation_loss = 0
        raw_output_diff = torch.zeros(6)
        for batch in validation_dataloader:
            if validation_size >= 0 and validation_step >= validation_size:
                break
            cam0_img, cam1_img, ground_truth = batch

            input_tensor = torch.cat((cam0_img, cam1_img), 1)
            if use_cuda:
                input_tensor = input_tensor.cuda()
            y = model(input_tensor)

            y = y.to("cpu")
            batch_loss = loss(y, ground_truth)
            raw_output_diff += torch.mean(torch.abs(y - ground_truth), dim=0)
            validation_loss += batch_loss
            validation_step += 1

        if special:
            print("Global step {} Val avg raw diff: {}".format(global_step, raw_output_diff / validation_step))
            for i in range(6):
                if i < 3:
                    summary_writer.add_scalar("val_raw/Entire Val RotAngle{} degrees diff".format(i),
                                              raw_output_diff[i] / np.pi*180 / validation_step, global_step)
                else:
                    summary_writer.add_scalar("val_raw/Entire Val TransVector{} diff".format(i - 3),
                                              raw_output_diff[i] / validation_step,
                                              global_step)
            summary_writer.add_scalar("val_loss/Entire average loss", validation_loss / validation_step, global_step)
        else:
            print("Global step {} Val avg raw diff: {}".format(global_step, raw_output_diff / validation_step))
            for i in range(6):
                if i < 3:
                    summary_writer.add_scalar("val_raw/Average Val RotAngle{} degrees diff".format(i),
                                              raw_output_diff[i] / np.pi * 180 / validation_step, global_step)
                else:
                    summary_writer.add_scalar("val_raw/Average Val TransVector{} diff".format(i - 3),
                                              raw_output_diff[i] / validation_step,
                                              global_step)
            summary_writer.add_scalar("val_loss/Val average loss", validation_loss / validation_step, global_step)
        print("Global step {} Validation avg loss: {}".format(global_step, validation_loss / validation_step))
        # TODO: Add random validation images
        return validation_loss/validation_step

if __name__ == '__main__':
    main()
