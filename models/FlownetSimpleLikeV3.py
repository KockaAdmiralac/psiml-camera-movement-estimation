import torch
import torch.nn as nn
from torch.nn import init

# Taken and adapted from:
# https://github.com/ClementPinard/FlowNetPytorch/blob/master/models/FlowNetS.py

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )


class FlowNetS_V3(nn.Module):

    def __init__(self, batchNorm=True):
        super(FlowNetS_V3, self).__init__()

        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, 6, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.conv7 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=0,
                  bias=True)
        self.ac_conv7 = nn.LeakyReLU(0.1, inplace=True)

        self.conv8 = nn.Conv2d(512, 512, kernel_size=(1,3), stride=1, padding=0,
                               bias=True)
        self.ac_conv8 = nn.LeakyReLU(0.1, inplace=True)

        # Dummy - rescale input, use RGB, this is just for testing
        self.fc1_branch_anglev3 = nn.Linear(512 * 6, 500)
        self.relu_fc1_branch_anglev3 = nn.ReLU()
        self.fc2_branch_anglev3 = nn.Linear(500, 100)
        self.relu_fc2_branch_anglev3 = nn.ReLU()
        self.output_branch_anglev3 = nn.Linear(100, 3)

        self.fc1_branch_translationv3 = nn.Linear(512 * 6, 500)
        self.relu_fc1_branch_translationv3 = nn.ReLU()
        self.fc2_branch_translationv3 = nn.Linear(500, 100)
        self.relu_fc2_branch_translationv3 = nn.ReLU()

        self.output_branch_translationv3 = nn.Linear(100, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

    def forward(self, x):
        # x is stacked 2 images - t, t+1
        self.conv1(x)
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        out_conv7 = self.ac_conv7(self.conv7(out_conv6))
        out_conv8 = self.ac_conv8(self.conv8(out_conv7))

        feature_extraction_output = out_conv8.reshape(-1, 512*6)
        fc1_branch_angle = self.relu_fc1_branch_anglev3(self.fc1_branch_anglev3(feature_extraction_output))
        fc2_branch_angle = self.relu_fc2_branch_anglev3(self.fc2_branch_anglev3(fc1_branch_angle))
        output_branch_angle = self.output_branch_anglev3(fc2_branch_angle)

        fc1_branch_translation = self.relu_fc1_branch_translationv3(self.fc1_branch_translationv3(feature_extraction_output))
        fc2_branch_translation = self.relu_fc2_branch_translationv3(self.fc2_branch_translationv3(fc1_branch_translation))
        output__branch_translation = self.output_branch_translationv3(fc2_branch_translation)

        return torch.cat((output_branch_angle, output__branch_translation), dim=1)
