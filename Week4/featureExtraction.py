# Import required packages
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import models
import os
import progressbar

# Initialize model
model = models.resnet18(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])

# Initialize data path and feature path
BASE_PATH = "/Users/advaitdixit/Documents/Masters/dataset/MIT_split/"
DATASET_PATH = os.path.sep.join([BASE_PATH, "train"])
QUERY_PATH = os.path.sep.join([BASE_PATH, "test"])
feature_path = Path("features")
os.makedirs(feature_path, exist_ok=True)

# Initialize the list of data (images), class labels, target bounding box coordinates, and image paths
print("[INFO] loading dataset...")

# Initialize the transforms
transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])

# Initialize the dataset and query set(https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html)
dataset = ImageFolder(DATASET_PATH, transform=transforms)
query_set = ImageFolder(QUERY_PATH, transform=transforms)

# Extract the labels
data_labels = [(data[0].split('/')[-1], data[1]) for data in dataset.imgs]
query_labels = [(data[0].split('/')[-1], data[1]) for data in query_set.imgs]

# Save the labels
with (feature_path / "data_labels.pkl").open('wb') as f_meta:
    pickle.dump(data_labels, f_meta)
with (feature_path / "query_labels.pkl").open('wb') as f_meta:
    pickle.dump(query_labels, f_meta)

# Initialize arrays for feature extraction
dataset_features = np.empty((len(dataset), 512))
query_data = np.empty((len(query_set), 512))

# Initialize the progress bar
widgets = ["Extracting Features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(dataset), widgets=widgets).start()

# Evaluation mode
print("[INFO] Extracting features from dataset...")
with torch.no_grad():
    model.eval()
    # Loop over data
    for ii, (img, _) in enumerate(dataset):
        dataset_features[ii, :] = model(img.unsqueeze(0)).squeeze().numpy()
        pbar.update(ii)

# Finish
pbar.finish()

# Initialize the progress bar
widgets = ["Extracting Features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(query_set), widgets=widgets).start()

# Save the features
with open(feature_path / "dataset.npy", "wb") as f:
    np.save(f, dataset_features)

# Evaluation mode
print("[INFO] Extracting features from query set...")
with torch.no_grad():
    model.eval()
    # Loop over query data
    for ii, (img, _) in enumerate(query_set):
        query_data[ii, :] = model(img.unsqueeze(0)).squeeze().numpy()
        pbar.update(ii)

# Save the features
with open(feature_path / "queries.npy", "wb") as f:
    np.save(f, query_data)

# Finish
pbar.finish()