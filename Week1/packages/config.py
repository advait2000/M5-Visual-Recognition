# Import required packages

import torch

import os



# specify path to the flowers and mnist dataset

BASE_PATH = "/home/mcv/datasets/MIT_split/"

DATASET_PATH = os.path.sep.join([BASE_PATH, "train"])



# specify the paths to our training and validation set

TRAIN = "train"

VAL = "val"



# set the input height and width

INPUT_HEIGHT = 224

INPUT_WIDTH = 224



# set the batch size and validation data split

INIT_LR = 1e-3

BATCH_SIZE = 8

EPOCHS = 100

VAL_SPLIT = 0.3



# Determine the current device and based on that set the pin memory flag

DEVICE = torch.device("mps")

PIN_MEMORY = True if DEVICE == "cuda" else False