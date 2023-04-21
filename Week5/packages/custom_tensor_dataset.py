# Import required packages

import os

import random

import torch

from PIL import Image

from torch.utils.data import Dataset

import json

import cv2

import logging

logging.basicConfig(level=logging.INFO)



class CustomTensorDataset(Dataset):

    # Initialize the constructor

    def __init__(self, tensors, transforms=None):

        self.tensors = tensors

        self.transforms = transforms



    def __getitem__(self, index):

        # Grab the image, label

        image = self.tensors[0][index]

        label = self.tensors[1][index]



        # Transpose the image such that its channel dimension becomes the leading one

        image = image.permute(2, 0, 1)



        # Check to see if we have any image transformations to apply and if so, apply them

        if self.transforms:

            image = self.transforms(image)



        # Return a tuple of the images

        return image, label



    def __len__(self):

        # Return the size of the dataset

        return self.tensors[0].size(0)





class CustomTensorDatasetSiamese(Dataset):

    def __init__(self, root_dir, transforms=None):

        self.root_dir = root_dir

        self.transform = transforms

        self.image_paths = []

        self.labels = []



        # Get a list of all image paths and labels

        for class_name in os.listdir(self.root_dir):

            class_dir = os.path.join(self.root_dir, class_name)

            if not os.path.isdir(class_dir):

                continue

            for image_name in os.listdir(class_dir):

                image_path = os.path.join(class_dir, image_name)

                self.image_paths.append(image_path)

                self.labels.append(class_name)



    def __len__(self):

        return len(self.image_paths)



    def __getitem__(self, index):

        # Load the two images and the label

        img1_path = self.image_paths[index]

        label1 = self.labels[index]

        img2_path = self.image_paths[random.choice(self.get_indices(label1))]

        label2 = label1



        img1 = Image.open(img1_path).convert('RGB')

        img2 = Image.open(img2_path).convert('RGB')



        if self.transform is not None:

            img1 = self.transform(img1)

            img2 = self.transform(img2)



        label = torch.tensor(0) if label1 == label2 else torch.tensor(1)



        return (img1, img2), label



    def get_indices(self, label):

        return [i for i, l in enumerate(self.labels) if l == label and i != self.labels.index(label)]





class CustomTensorDatasetTripletLoss(Dataset):

    def __init__(self, root_dir, transforms=None):

        self.root_dir = root_dir

        self.transform = transforms

        self.image_paths = []

        self.labels = []



        # Get a list of all image paths and labels

        for class_name in os.listdir(self.root_dir):

            class_dir = os.path.join(self.root_dir, class_name)

            if not os.path.isdir(class_dir):

                continue

            for image_name in os.listdir(class_dir):

                image_path = os.path.join(class_dir, image_name)

                self.image_paths.append(image_path)

                self.labels.append(class_name)



        # Build a dictionary that maps labels to indices in the dataset

        self.label_to_indices = {}

        for i, label in enumerate(self.labels):

            if label not in self.label_to_indices:

                self.label_to_indices[label] = []

            self.label_to_indices[label].append(i)



    def __len__(self):

        return len(self.image_paths)



    def __getitem__(self, index):

        # Load the anchor image and label

        anchor_path = self.image_paths[index]

        anchor_label = self.labels[index]

        anchor = Image.open(anchor_path).convert('RGB')



        # Apply transform to anchor image

        if self.transform is not None:

            anchor = self.transform(anchor)



        # Sample a positive image with the same label as the anchor

        positive_index = index

        while positive_index == index:

            positive_index = random.choice(self.label_to_indices[anchor_label])

        positive_path = self.image_paths[positive_index]

        positive = Image.open(positive_path).convert('RGB')



        # Apply transform to positive image

        if self.transform is not None:

            positive = self.transform(positive)



        # Sample a negative image with a different label than the anchor

        negative_label = anchor_label

        while negative_label == anchor_label:

            negative_label = random.choice(self.labels)

        negative_index = random.choice(self.label_to_indices[negative_label])

        negative_path = self.image_paths[negative_index]

        negative = Image.open(negative_path).convert('RGB')



        # Apply transform to negative image

        if self.transform is not None:

            negative = self.transform(negative)



        # Return the data for triplet loss

        return (anchor, positive, negative), torch.tensor([0], dtype=torch.float32)





class CustomTensorDatasetTriplet_Image2Text(Dataset):

    def __init__(self, root_dir, annFile, type_data, transforms=None):

        self.root_dir = root_dir

        self.transform = transforms

        self.type_data = type_data



        f = open(annFile)

        captionJson = json.load(f)

        f.close()



        self.ImageNcaption = []

        for caption in captionJson["annotations"]:

            self.ImageNcaption.append([str(caption["image_id"]), str(caption["caption"])])



    def __len__(self):
        #if self.type_data == "val":
        #    logging.info("val: "+str(int(len(self.ImageNcaption)*0.01)))		
        #    return int(len(self.ImageNcaption)*0.01)

        logging.info("train: "+str(int(len(self.ImageNcaption)*0.01)))

        return int(len(self.ImageNcaption)*0.01)



    def __getitem__(self, index):

        # Load the two images and the label

        img1name = self.ImageNcaption[index][0]

        cero = "0"*(12-len(img1name))



        name_file = "/COCO_train2014_"

        if self.type_data == "val":

            name_file = "/COCO_val2014_"



        anchor = Image.open(self.root_dir +name_file+cero+ img1name+".jpg").convert('RGB')



        # Transform

        if self.transform is not None:

            anchor = self.transform(anchor)





        positive = self.ImageNcaption[index][1]

        

        index_neg = index

        while index_neg == index:

            index_neg = random.choice(list(range(len(self.ImageNcaption))))

        

        negative = self.ImageNcaption[index_neg][1]



        return (anchor, positive, negative), torch.tensor([0], dtype=torch.float32)







































































