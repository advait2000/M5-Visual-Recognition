# Import required packages

import time

import torch

from matplotlib import pyplot as plt

from torch import nn

from torch.optim import SGD

from torch.utils.data import DataLoader

from torchvision import transforms

from packages import CustomTensorDatasetTripletLoss

from packages.custom_tensor_dataset import TripletCOCOdatabase

from packages import config

from triplet_network import TripletNetwork

from pathlib import Path

from PIL import Image

import os

import random

import numpy as np

import json

import cv2

import logging



def get_name(name):

    name = name.__str__()

    return name.split("/")[-1][-16:].lstrip('0')[:-4]



logging.basicConfig(level=logging.INFO)



# Set the device we will be using to train the model

device = torch.device("cuda")

dtype = torch.float



# Initialize the list of data (images), class labels, target bounding box coordinates, and image paths

print("[INFO] Loading Dataset...")

data = []

labels = []

bboxes = []

imagePaths = []





transform = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])



# Load model

model = torch.load('/ghome/group06/m5/w4/weights.pth')

model.eval()



folder_dir_train = '/home/mcv/datasets/COCO/train2014'

images_train = Path(folder_dir_train).glob('*.jpg')



folder_dir_val = '/home/mcv/datasets/COCO/val2014'

images_val = Path(folder_dir_val).glob('*.jpg')







databaseImageLabels = '/home/mcv/datasets/COCO/mcv_image_retrieval_annotations.json'

f = open(databaseImageLabels)

labelJson = json.load(f)

f.close()

imagesNob = {}



max_it = 1000

logging.info("Reading train")

print("Reading train")

for i,k in enumerate(labelJson['train'].keys()):

    #if i >= max_it:

    #    break

    for img in labelJson['train'][k]:

        if img not in imagesNob:

            imagesNob[img] = [k]

        else:

            list_objs = imagesNob[img]

            imagesNob[img] = list_objs + [k]



logging.info("Reading val")

print("Reading val")



for i,k in enumerate(labelJson['val'].keys()):

    #if i >= max_it:

    #    break

    for img in labelJson['val'][k]:

        if img not in imagesNob:

            imagesNob[img] = [k]

        else:

            list_objs = imagesNob[img]

            imagesNob[img] = list_objs + [k]



images_t = []

train_output = []

for i,image_train_name in enumerate(images_train):

    if i >= max_it:

        break

    images_t.append(image_train_name)

    image_train = Image.open(image_train_name).convert('RGB')

    image_train = transform(image_train).to(device).unsqueeze(0)

    train_output.append(model.resnet(image_train).detach())



n_min_values = 5

m_prec = []

max_it = 39

it = 0

for image_val_name in images_val:

    

    if int(get_name(image_val_name)) in imagesNob.keys():

        logging.info("it "+str(it))

        logging.info(image_val_name)

        it += 1

        image_val = Image.open(image_val_name).convert('RGB')

        image_val = transform(image_val).to(device).unsqueeze(0)

        image_val_o = model.resnet(image_val).detach()



        distances = []

        for train_o in train_output:

            distances.append(np.linalg.norm(train_o.cpu()-image_val_o.cpu()))



        list_dist = sorted(range(len(distances)), key=lambda k: distances[k])[:n_min_values]

        

        objects_val = imagesNob[int(get_name(image_val_name))]

        prec = []

        for best in list_dist:

            name_train_again = images_t[best]

           

            prec.append(any(obj in objects_val for obj in imagesNob[int(get_name(name_train_again))]))

        m_prec.append(prec.count(True)/len(prec))

        if m_prec[-1] == 0.6:

            print("aaaaaa")

            print("image_val_name",image_val_name)

            for best in list_dist:

                print(images_t[best])

           

        if it > max_it:

            break

    else:

        logging.info("nota "+get_name(image_val_name))

print(m_prec)

print("MAP",sum(m_prec)/len(m_prec))







