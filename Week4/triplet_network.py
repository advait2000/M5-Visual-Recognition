#
import torch.nn as nn
import torchvision.models as models


class TripletNetwork(nn.Module):
    def __init__(self, embedding_size):
        super(TripletNetwork, self).__init__()
        self.embedding_size = embedding_size
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, embedding_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, anchor, positive, negative):
        anchor_embedding = self.resnet(anchor)
        positive_embedding = self.resnet(positive)
        negative_embedding = self.resnet(negative)
        return anchor_embedding, positive_embedding, negative_embedding
