import torch
from torch import nn, cat


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
    def __init__(self, width=227, height=227, classes=8, activation='elu', dropout=0.2, dense=256):
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


class SimpleModel(nn.Module):
    def __init__(self, activation='softplus', classes=8):
        super(SimpleModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.activation = nn.ELU()
        self.depthwise_conv1 = nn.Conv2d(32, 32, kernel_size=3, groups=32, padding=1)
        self.conv2 = nn.Conv2d(32, 30, kernel_size=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(30, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.depthwise_conv2 = nn.Conv2d(32, 32, kernel_size=3, groups=32, padding=1)
        self.conv4 = nn.Conv2d(32, 30, kernel_size=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(30, 64, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(64)
        self.depthwise_conv3 = nn.Conv2d(64, 64, kernel_size=3, groups=64, padding=1)
        self.conv6 = nn.Conv2d(64, 30, kernel_size=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # final layers
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # global average pooling layer
        self.flatten = nn.Flatten()  # flatten layer
        self.dense1 = nn.Linear(30, 256)  # fully connected layer
        self.dropout = nn.Dropout(0.1)  # dropout layer
        self.dense2 = nn.Linear(256, classes)  # output layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x0 = x
        x = self.depthwise_conv1(x)
        x = self.activation(x)
        x = x + x0
        x = self.conv2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.bn2(x)
        x = self.activation(x)
        x0 = x
        x = self.depthwise_conv2(x)
        x = self.activation(x)
        x = x + x0
        x = self.conv4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.bn3(x)
        x = self.activation(x)
        x0 = x
        x = self.depthwise_conv3(x)
        x = self.activation(x)
        x = x + x0
        x = self.conv6(x)
        x = self.pool3(x)

        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)

        return x


class DeepModel(nn.Module):
    def __init__(self, activation='softplus', classes=8):
        super(DeepModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.activation = nn.ELU()
        self.depthwise_conv1 = nn.Conv2d(32, 32, kernel_size=3, groups=32, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(64, 32, kernel_size=3)
        self.depthwise_conv2 = nn.Conv2d(32, 32, kernel_size=3, groups=32, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        self.depthwise_conv3 = nn.Conv2d(64, 64, kernel_size=3, groups=64, padding=1)
        self.conv6 = nn.Conv2d(64, 128, kernel_size=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv7 = nn.Conv2d(128, 128, kernel_size=3)
        self.depthwise_conv4 = nn.Conv2d(128, 128, kernel_size=3, groups=128, padding=1)
        self.conv8 = nn.Conv2d(128, 256, kernel_size=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # final layers
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # global average pooling layer
        self.flatten = nn.Flatten()  # flatten layer
        self.dense1 = nn.Linear(256, 256)  # fully connected layer
        self.dropout = nn.Dropout(0.1)  # dropout layer
        self.dense2 = nn.Linear(256, classes)  # output layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x0 = x
        x = self.depthwise_conv1(x)
        x = self.activation(x)
        x = x + x0
        x = self.conv2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.activation(x)
        x0 = x
        x = self.depthwise_conv2(x)
        x = self.activation(x)
        x = x + x0
        x = self.conv4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.activation(x)
        x0 = x
        x = self.depthwise_conv3(x)
        x = self.activation(x)
        x = x + x0
        x = self.conv6(x)
        x = self.pool3(x)

        x = self.conv7(x)
        x = self.activation(x)
        x0 = x
        x = self.depthwise_conv4(x)
        x = self.activation(x)
        x = x + x0
        x = self.conv8(x)
        x = self.pool4(x)

        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)

        return x
