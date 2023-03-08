import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super(ResidualBlock, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.depthwise_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.conv2 = nn.Conv2d(out_channels, 30, kernel_size=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.activation(out)
        out = self.depthwise_conv(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.max_pool(out)
        out = self.activation(out)
        out = self.depthwise_conv(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.max_pool(out)
        out = self.activation(out)
        out = self.depthwise_conv(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.max_pool(out)
        out = self.activation(out)
        out += residual
        return out


class Model_torch(nn.Module):
    def __init__(self, width = 227, height = 227, classes = 8, activation='elu', dropout=0.2, dense=256):
        super(Model_torch, self).__init__()
        self.activation = nn.ELU() if activation == 'elu' else nn.ReLU()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.residual_blocks = nn.Sequential(
            ResidualBlock(32, 30, self.activation),
            ResidualBlock(30, 30, self.activation),
            ResidualBlock(30, 30, self.activation),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(30, dense)
        self.fc2 = nn.Linear(dense, classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)
        out = self.residual_blocks(out)
        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = nn.functional.softmax(out, dim=1)
        return out


