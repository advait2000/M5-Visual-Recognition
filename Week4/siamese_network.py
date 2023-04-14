import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SiameseNet(nn.Module):
    def __init__(self, backbone=models.resnet18(pretrained=True), embedding_size=128):
        super(SiameseNet, self).__init__()
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(backbone.fc.in_features, self.embedding_size)
        self.fc2 = nn.Linear(self.embedding_size, self.embedding_size)

    def forward(self, x1, x2):
        x1 = self.backbone(x1)
        x2 = self.backbone(x2)
        x1 = x1.view(x1.size()[0], -1)
        x2 = x2.view(x2.size()[0], -1)
        x1 = self.fc1(x1)
        x2 = self.fc1(x2)
        x1 = F.relu(x1)
        x2 = F.relu(x2)
        x1 = self.fc2(x1)
        x2 = self.fc2(x2)
        return x1, x2