import torch.nn as nn

import torchvision.models as models

import fasttext

import numpy as np



# Define the image-to-text retrieval model

class ImageToText(nn.Module):

    def __init__(self):

        super(ImageToText, self).__init__()

        # Define the ResNet-50 model

        resnet = models.resnet50(pretrained=True)

        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        self.flatten = nn.Flatten()

        self.embedding_rest = nn.Linear(2048, 1024)



        self.embedding_text = nn.Linear(300, 1024)



        self.embedding_general = nn.Linear(1024, 1024)





    def forward(self, images, captions_positive, captions_negative):

        x = self.resnet(images)

        x = self.flatten(x)

        x = self.embedding_rest(x)

        image_embeddings = self.embedding_general(x)





        x = self.embedding_text(captions_positive)

        text_embeddings_pos = self.embedding_general(x)



        x = self.embedding_text(captions_negative)

        text_embeddings_neg = self.embedding_general(x)

        

        return image_embeddings, text_embeddings_pos, text_embeddings_neg

