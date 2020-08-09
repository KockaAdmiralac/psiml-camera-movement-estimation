import torch
import torch.nn as nn


class RMSEWeightedLoss(nn.Module):
    def __init__(self, eps=1e-6, beta=10):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        self.beta = beta

    def forward(self, yhat, y):
        # Weighted RMSE loss
        loss = self.beta*torch.sqrt(self.mse(y[:, :3], yhat[:, :3]))+torch.sqrt(self.mse(y[:, 3:], yhat[:, 3:]))

        #loss = (self.beta**2) * self.mse(y[:, :3], yhat[:, :3]) + self.mse(y[:, 3:], yhat[:, 3:])
        return loss

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


class FlowNetS(nn.Module):

    def __init__(self, batchNorm=True):
        super(FlowNetS, self).__init__()

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

        # TODO: Add regression for output
        # Dummy - rescale input, use RGB, this is just for testing
        self.fc1 = nn.Linear(1024 * 3 * 10, 1000)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(1000, 100)
        self.relu_fc2 = nn.ReLU()
        self.output = nn.Linear(100, 6)

    def forward(self, x):
        # x is stacked 2 images - t, t+1
        self.conv1(x)
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        fc1 = self.relu_fc1(self.fc1(out_conv6.reshape(-1, 1024*3*10)))
        fc2 = self.relu_fc2(self.fc2(fc1))

        return self.output(fc2)
