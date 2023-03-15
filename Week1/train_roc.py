# Import required packages

import os
import time

import scikitplot as skplt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from imutils import paths

from model_pytorch import SimpleModel, DeepModel
from packages import CustomTensorDataset
from packages import config
from torchsummary import summary

# Initialize the list of data (images), class labels, target bounding box coordinates, and image paths
print("[INFO] loading dataset...")
data = []
labels = []
bboxes = []
imagePaths = []

# Grab the image paths
pathToImages = "/Users/advaitdixit/Documents/Masters/dataset/MIT_split/test"
print(pathToImages)

# Loop over the folders
for trainPath in paths.list_images(pathToImages):
    image = cv2.imread(trainPath)
    label = trainPath.split(os.path.sep)[-2]
    (height, width) = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    # Update our list of data, class labels,
    data.append(image)
    labels.append(label)
    imagePaths.append(trainPath)

# perform label encoding on the labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# Convert the data, class labels, bounding boxes, and image paths to NumPy arrays
testImages = np.array(data, dtype="float32")
testLabels = np.array(labels)
imagePaths = np.array(imagePaths)

# convert NumPy arrays to PyTorch tensors
testImages = torch.tensor(testImages)
# testLabels = torch.tensor(testLabels)

# define normalization transforms
transforms = transforms.Compose(
    [transforms.ToPILImage(), transforms.ToTensor()])

# convert NumPy arrays to PyTorch datasets
testDS = CustomTensorDataset((testImages, testLabels), transforms=transforms)
print("[INFO] total test samples: {}...".format(len(testDS)))

# Calculate steps per epoch for training and validation set
valSteps = len(testDS) // config.BATCH_SIZE

# Create data loaders
testLoader = DataLoader(testDS, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY)

model = DeepModel().to(device=config.DEVICE)
model.load_state_dict(torch.load('output/model_deep_batch_normv2.pth', map_location=torch.device(config.DEVICE)))

summary(model, input_size=(3, config.INPUT_WIDTH, config.INPUT_HEIGHT))

# initialize our optimizer and loss function
opt = Adam(model.parameters(), lr=config.INIT_LR)
lossFn = CrossEntropyLoss()

# initialize a dictionary to store training history
H = {
    "val_loss": [],
    "val_acc": []
}
# measure how long training is going to take
print("[INFO] Evaluating the network...")
startTime = time.time()

# switch off autograd for evaluation
with torch.no_grad():
    # set the model in evaluation mode
    model.eval()

    # Initialize the total training and validation loss
    totalTrainLoss = 0
    totalValLoss = 0

    # Initialize the number of correct predictions in the training and validation step
    trainCorrect = 0
    valCorrect = 0
    true = []
    predicted = []
    # loop over the validation set
    for (images, labels) in testLoader:
        # send the input to the device
        (images, labels) = (images.to(device=config.DEVICE), labels.to(device=config.DEVICE))

        # make the predictions and calculate the validation loss
        predictions = model(images)
        y_pred = torch.argmax(predictions, dim=1).numpy()
        totalValLoss += lossFn(predictions, labels)

        # calculate the number of correct predictions
        valCorrect += (predictions.argmax(1) == labels).type(torch.float).sum().item()
        true.append(labels.cpu().numpy())
        predicted.append(predictions.cpu().numpy())

# convert true and predicted lists to numpy arrays
true = np.concatenate(true)
predicted = np.concatenate(predicted)

predicted_indices = np.argmax(predicted, axis=1)
predicted_labels = le.inverse_transform(predicted_indices)

print(testLabels)
print(predicted_indices)

skplt.metrics.plot_roc(true, predicted)
plt.show()
testLabels = le.inverse_transform(testLabels)
predicted_indices = predicted_indices.astype(int)
skplt.metrics.plot_confusion_matrix(testLabels, predicted_labels, normalize=True)
plt.show()
