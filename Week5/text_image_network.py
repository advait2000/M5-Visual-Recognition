import torch.nn as nn

import torchvision.models as models

import fasttext

import numpy as np



# Define the image-to-text retrieval model

class TextToImage(nn.Module):

    def __init__(self):

        super(TextToImage, self).__init__()

        # Define the ResNet-50 model

        resnet = models.resnet50(pretrained=True)

        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        self.flatten = nn.Flatten()

        self.embedding_rest = nn.Linear(2048, 1024)



        self.embedding_text = nn.Linear(300, 1024)



        self.embedding_general_1 = nn.Linear(1024, 1024)

        self.embedding_general_2 = nn.Linear(1024, 1024)

        self.embedding_general_3 = nn.Linear(1024, 1024)





    def forward(self, caption, image_positive, image_negative):



        x = self.embedding_text(caption)

        #x = self.embedding_general_1(x)

        #x = self.embedding_general_2(x)

        text_embeddings = self.embedding_general_3(x)



        x = self.resnet(image_positive)

        x = self.flatten(x)

        x = self.embedding_rest(x)

        #x = self.embedding_general_1(x)

        #x = self.embedding_general_2(x)

        image_embeddings_pos = self.embedding_general_3(x)





        x = self.resnet(image_negative)

        x = self.flatten(x)

        x = self.embedding_rest(x)

        #x = self.embedding_general_1(x)

        #x = self.embedding_general_2(x)

        image_embeddings_neg = self.embedding_general_3(x)

        

        return text_embeddings, image_embeddings_pos, image_embeddings_neg

