# Import required packages
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms

from model_pytorch import DeepModel_he
from packages import CustomTensorDataset
from packages import config

"""if not torch.cuda.is_available():
    print("Cuda not available")
    exit()"""

# set the device we will be using to train the model
device = torch.device("cpu")
dtype = torch.float

# Initialize the list of data (images), class labels, target bounding box coordinates, and image paths
print("[INFO] loading dataset...")
data = []
labels = []
bboxes = []
imagePaths = []

# Grab the image paths
pathToImages = config.DATASET_PATH
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

# Convert the data, class labels and image paths to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)
imagePaths = np.array(imagePaths)

# perform label encoding on the labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# Partition the data into training and testing splits using
# 70% of the data for training and the remaining 30% for testing
(trainImages, testImages, trainLabels, testLabels) = train_test_split(data, labels, test_size=0.30, random_state=42)

# convert NumPy arrays to PyTorch tensors
(trainImages, testImages) = torch.tensor(trainImages), torch.tensor(testImages)
(trainLabels, testLabels) = torch.tensor(trainLabels), torch.tensor(testLabels)

# initialize our data augmentation functions
hFlip = transforms.RandomHorizontalFlip(p=0.25)
vFlip = transforms.RandomVerticalFlip(p=0.25)
rotate = transforms.RandomRotation(degrees=15)

# define normalization transforms
transforms = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

# convert NumPy arrays to PyTorch datasets
trainDS = CustomTensorDataset((trainImages, trainLabels), transforms=transforms)
testDS = CustomTensorDataset((testImages, testLabels), transforms=transforms)

print("[INFO] total training samples: {}...".format(len(trainDS)))
print("[INFO] total test samples: {}...".format(len(testDS)))

# Calculate steps per epoch for training and validation set
trainSteps = len(trainDS) // config.BATCH_SIZE
valSteps = len(testDS) // config.BATCH_SIZE

# Create data loaders
trainLoader = DataLoader(trainDS, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=config.PIN_MEMORY)
testLoader = DataLoader(testDS, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY)
model = DeepModel_he().to(device)
summary(model, input_size=(3, config.INPUT_WIDTH, config.INPUT_HEIGHT))

# initialize our optimizer and loss function
opt = SGD(model.parameters(), lr=config.INIT_LR, momentum=0.9)
lossFn = CrossEntropyLoss()

# initialize a dictionary to store training history
H = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

# measure how long training is going to take
print("[INFO] training the network...")
startTime = time.time()

# loop over our epochs
for e in range(0, config.EPOCHS):
    # set the model in training mode
    model.train()

    # Initialize the total training and validation loss
    totalTrainLoss = 0
    totalValLoss = 0

    # Initialize the number of correct predictions in the training and validation step
    trainCorrect = 0
    valCorrect = 0

    # loop over the training set
    for (images, labels) in trainLoader:
        # send the input to the device
        (images, labels) = (images.to(device), labels.to(device))

        # export the PyTorch model to the ONNX format
        # perform a forward pass and calculate the training loss
        predictions = model(images)
        classLoss = lossFn(predictions, labels)

        # zero out the gradients, perform the backpropagation step, and update the weights
        opt.zero_grad()
        classLoss.backward()
        opt.step()

        # add the loss to the total training loss so far and
        # calculate the number of correct predictions
        totalTrainLoss += classLoss
        trainCorrect += (predictions.argmax(1) == labels).type(torch.float).sum().item()

    # switch off autograd for evaluation

    with torch.no_grad():

        # set the model in evaluation mode
        model.eval()

        # loop over the validation set
        for (images, labels) in testLoader:
            # send the input to the device
            (images, labels) = (images.to(device), labels.to(device))

            # make the predictions and calculate the validation loss
            predictions = model(images)
            totalValLoss += lossFn(predictions, labels)

            # calculate the number of correct predictions
            valCorrect += (predictions.argmax(1) == labels).type(torch.float).sum().item()

    # calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps

    # calculate the training and validation accuracy
    trainCorrect = trainCorrect / len(trainDS)
    valCorrect = valCorrect / len(testDS)

    # update our training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["train_acc"].append(trainCorrect)
    H["val_loss"].append(avgValLoss.cpu().detach().numpy())
    H["val_acc"].append(valCorrect)

    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, config.EPOCHS))
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
    print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(avgValLoss, valCorrect))

# finish measuring how long training took
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.title("Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("output/plotlossdeepxavier.png")

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_acc"], label="train_acc")
plt.plot(H["val_acc"], label="val_acc")
plt.title("Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig("output/plotaccdeepxavier.png")

# serialize the model to disk
torch.save(model, "output/modeldeepxavire.pth")
