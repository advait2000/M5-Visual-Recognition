import time



import torch

import torch.nn as nn

from torchvision.datasets import CocoCaptions

import torchvision.models as models

import torchvision.transforms as transforms

from matplotlib import pyplot as plt

from torch.optim import SGD, Adam, Adagrad

from torch.utils.data import DataLoader

from sklearn import preprocessing

import fasttext

import numpy as np

import math

import json
from PIL import Image
from torch.utils.data import Dataset
import cv2


from packages import config 

from image_text_network import ImageToText

from packages.custom_tensor_dataset import CustomTensorDatasetTriplet_Image2Text

import logging



logging.basicConfig(level=logging.INFO)

# Set the device we will be using to train the model

device = torch.device("cuda")

dtype = torch.float



max_it = 39

def read_annotations(annFile, max_it):

    f = open(annFile)
    captionJson = json.load(f)
    f.close()

    ImageNcaption = []
    for i, caption in enumerate(captionJson["annotations"]):
        ImageNcaption.append([str(caption["image_id"]), str(caption["caption"])])
        if i >= max_it:
            break
    return ImageNcaption




ft_model = fasttext.load_model("/home/mcv/m5/fasttext_wiki.en.bin")



# max length in train is 250

seq_len = 300

vocab_size = len(ft_model.get_words())

embedding_dim = ft_model.get_dimension()



# Define data transforms

transform = transforms.Compose([

    transforms.Resize(224),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

])



# https://pytorch.org/vision/main/generated/torchvision.datasets.CocoCaptions.html

# Load the COCO dataset


# Create the model

model = torch.load('/ghome/group06/m5/w5/weights_image_text.pth').to(device)






# measure how long training is going to take

print("[INFO] training the network...")

startTime = time.time()


ImageNcaption = read_annotations('/home/mcv/datasets/COCO/captions_val2014.json', max_it)


n_min_values = 5
m_prec = []

it = 0
correct_pred = 0

# switch off autograd for evaluation

with torch.no_grad():

    # set the model in evaluation mode

    model.eval()



    for gtImage, gtCaption in ImageNcaption:
        logging.info("Iteration: "+str(it))
        img1name = gtImage
        cero = "0"*(12-len(img1name))
        name_file = "/COCO_val2014_"
        root_dir = '/home/mcv/datasets/COCO/val2014'
        gtImagePict = Image.open(root_dir +name_file+cero+ img1name+".jpg").convert('RGB')
        anchor = transform(gtImagePict).unsqueeze(0).to(device)

        distances = []
        for _, caption in ImageNcaption:



            positive_tensors = np.zeros((1, 300))

        

            positive_words = np.array([])

            input_tokens = caption.split()

            for token in input_tokens:

                if token.lower() in ft_model:

                    positive_words = np.append(positive_words, ft_model[token.lower()])

            positive_tensors[0] = np.mean(positive_words, axis=0)



            positive_tensors = torch.from_numpy(positive_tensors)

            

            positive = positive_tensors.to(torch.float32).to(device)



            # Forward + backward + optimize

            image_val_o, text_val, _ = model(anchor, positive, positive)
    
            distances.append(np.linalg.norm(text_val.cpu()-image_val_o.cpu()))

        list_dist = sorted(range(len(distances)), key=lambda k: distances[k])[:n_min_values]
            
        prec = []
        for best in list_dist:
            capt_pred = ImageNcaption[best][1]
            if capt_pred == gtCaption:
                correct_pred += 1

        it += 1
        if it > max_it:
            break


logging.info("Pred: "+str(correct_pred/max_it))

print("Pred at {} with {}".format(n_min_values, correct_pred/max_it))

# finish measuring how long training took

endTime = time.time()

print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))