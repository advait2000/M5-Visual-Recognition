# Import required packages
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset


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


class CustomTensorDatasetSiamese(torch.utils.data.Dataset):
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
