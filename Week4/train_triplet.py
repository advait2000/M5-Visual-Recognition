# Import required packages
import time
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import transforms
from packages import CustomTensorDatasetTripletLoss
from packages import config
from triplet_network import TripletNetwork

# Set the device we will be using to train the model
device = torch.device("cpu")
dtype = torch.float

# Initialize the list of data (images), class labels, target bounding box coordinates, and image paths
print("[INFO] Loading Dataset...")
data = []
labels = []
bboxes = []
imagePaths = []

# Grab the image paths
pathToImages = config.DATASET_PATH

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

traindataset = CustomTensorDatasetTripletLoss('/Users/advaitdixit/Documents/Masters/dataset/MIT_split/train',
                                              transforms=transform)
testdataset = CustomTensorDatasetTripletLoss('/Users/advaitdixit/Documents/Masters/dataset/MIT_split/test',
                                             transforms=transform)

# Create data loaders
trainLoader = DataLoader(traindataset, batch_size=32, shuffle=True)
testLoader = DataLoader(testdataset, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY)

# Calculate steps per epoch for training and validation set
trainSteps = len(traindataset) // config.BATCH_SIZE
valSteps = len(testdataset) // config.BATCH_SIZE

# Load model
model = TripletNetwork(embedding_size=128).to(device)

# initialize our optimizer and loss function
opt = SGD(model.parameters(), lr=config.INIT_LR, momentum=0.9)

# Define the loss function
criterion = nn.TripletMarginLoss(margin=1.0)

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

# Loop over our epochs
for e in range(0, config.EPOCHS):
    # set the model in training mode
    model.train()

    # Initialize the total training and validation loss
    totalTrainLoss = 0
    totalValLoss = 0

    for (triplet, _) in trainLoader:
        # send the input to the device
        anchor, positive, negative = triplet
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        # Forward + backward + optimize
        anchor_embedding, positive_embedding, negative_embedding = model(anchor, positive, negative)
        loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
        opt.zero_grad()
        loss.backward(retain_graph=True)
        opt.step()
        totalTrainLoss += loss.item()

    # switch off autograd for evaluation
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()

        for (triplet, _) in testLoader:
            # send the input to the device
            anchor, positive, negative = triplet
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            # Forward + backward + optimize
            anchor_embedding, positive_embedding, negative_embedding = model(anchor, positive, negative)
            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            totalValLoss += loss.item()

    # calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps

    # update our training history
    H["train_loss"].append(avgTrainLoss)
    H["val_loss"].append(avgValLoss)

    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, config.EPOCHS))
    print("Train loss: {:.6f}, Val Loss: {:.4f}".format(avgTrainLoss, avgValLoss))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.title("Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("losstriplet.png")

# finish measuring how long training took
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
