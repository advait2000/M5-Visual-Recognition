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

import random


logging.basicConfig(level=logging.INFO)

# Set the device we will be using to train the model

device = torch.device("cuda")

dtype = torch.float



max_it = 100

max_search = 40

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

model = torch.load('/ghome/group06/m5/w5/weights_text_image300.pth').to(device)







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



    for i, (gtImage, gtCaption) in enumerate(ImageNcaption):

        logging.info("Iteration: "+str(it))



        anchor_tensors = np.zeros((1, 300))

        positive_words = np.array([])

        input_tokens = gtCaption.split()

        for token in input_tokens:

            if token.lower() in ft_model:

                positive_words = np.append(positive_words, ft_model[token.lower()])

        anchor_tensors[0] = np.mean(positive_words, axis=0)

        anchor_tensors = torch.from_numpy(anchor_tensors)

            

        anchor = anchor_tensors.to(torch.float32).to(device)





        distances = []
        new_list = ImageNcaption[:i] + ImageNcaption[i+1:]  # create new list without position i
        random_list = random.sample(new_list, max_search)  # get n random values from new list
        random_list += [ImageNcaption[i]]

        for image, _ in random_list:



            img1name = image

            cero = "0"*(12-len(img1name))

            name_file = "/COCO_val2014_"

            root_dir = '/home/mcv/datasets/COCO/val2014'

            gtImagePict = Image.open(root_dir +name_file+cero+ img1name+".jpg").convert('RGB')

            positive = transform(gtImagePict).unsqueeze(0).to(device)



            # Forward + backward + optimize

            text_val, image_val_o, _ = model(anchor, positive, positive)

    

            distances.append(np.linalg.norm(text_val.cpu()-image_val_o.cpu()))



        list_dist = sorted(range(len(distances)), key=lambda k: distances[k])[:n_min_values]

            

        prec = []

        for best in list_dist:

            img_pred = ImageNcaption[best][0]

            if img_pred == gtImage:

                correct_pred += 1



        it += 1

        if it > max_it:

            break



logging.info("Pred: "+str(correct_pred/max_it))

print("Pred at {} with {}".format(n_min_values, correct_pred/max_it))

# finish measuring how long training took

endTime = time.time()

print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))