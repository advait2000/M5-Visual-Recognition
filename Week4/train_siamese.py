# Import required packages
import os
import pickle
import time
import torch
import torch.nn.functional as F
import umap
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from average_precision import *
from packages import CustomTensorDatasetSiamese
from packages import config
from siamese_network import SiameseNet


class ContrastiveLoss(nn.Module):
    """
    Takes embeddings of two samples and a target label == 1 if
    samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1.0 - target).float() * F.relu(self.margin
                                                        - (distances + self.eps).sqrt()).pow(2))
        # sqrt() of a tiny number may be negative!
        return losses.mean() if size_average else losses.sum()


# Set the device we will be using to train the model
device = torch.device("cpu")
dtype = torch.float

# Initialize the list of data (images), class labels, target bounding box coordinates, and image paths
print("[INFO] Loading Dataset...")
data = []
labels = []
imagePaths = []

# Grab the image paths
pathToImages = config.DATASET_PATH

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

traindataset = CustomTensorDatasetSiamese('/Users/advaitdixit/Documents/Masters/dataset/MIT_split/train',
                                          transforms=transform)
testdataset = CustomTensorDatasetSiamese('/Users/advaitdixit/Documents/Masters/dataset/MIT_split/test',
                                         transforms=transform)

# Create data loaders
trainLoader = DataLoader(traindataset, batch_size=32, shuffle=True)
testLoader = DataLoader(testdataset, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY)

# Calculate steps per epoch for training and validation set
trainSteps = len(traindataset) // config.BATCH_SIZE
valSteps = len(testdataset) // config.BATCH_SIZE

# Load model
model = SiameseNet().to(device)

# initialize our optimizer and loss function
opt = SGD(model.parameters(), lr=config.INIT_LR, momentum=0.9)

# Define the loss function
criterion = ContrastiveLoss().to(device)

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

    # loop over the training set
    for (inputs1, inputs2), labels in trainLoader:
        # send the input to the device
        (inputs1, inputs2), labels = ((inputs1.to(device), inputs2.to(device)), labels.to(device))
        outputs1, outputs2 = model(inputs1, inputs2)
        loss = criterion(outputs1, outputs2, labels)
        opt.zero_grad()
        loss.backward(retain_graph=True)
        opt.step()
        totalTrainLoss += loss.item()

    # switch off autograd for evaluation
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        # loop over the validation set
        for (inputs1, inputs2), labels in testLoader:
            # send the input to the device
            (inputs1, inputs2), labels = ((inputs1.to(device), inputs2.to(device)), labels.to(device))
            outputs1, outputs2 = model(inputs1, inputs2)
            loss = criterion(outputs1, outputs2, labels)
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
plt.savefig("loss_slidesSiamese.png")

# finish measuring how long training took
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))

# Initialize data path and feature path
BASE_PATH = "/Users/advaitdixit/Documents/Masters/dataset/MIT_split/"
DATASET_PATH = os.path.sep.join([BASE_PATH, "train"])
QUERY_PATH = os.path.sep.join([BASE_PATH, "test"])

# Initialize the dataset and query set(https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html)
dataset = ImageFolder(DATASET_PATH, transform=transform)
query_set = ImageFolder(QUERY_PATH, transform=transform)

model.eval()
# Extract features for visualization train
featuresData = []
labels = []

# Loop over train dataset
for inputs, label in dataset:
    # Pass image through network
    inputs = inputs.unsqueeze(0)
    feature = model.backbone(inputs).detach().numpy()[0]

    # Append features and labels
    featuresData.append(feature)
    labels.append(label)

# Convert to numpy arrays
features = np.array(featuresData).reshape(len(featuresData), -1)
dataFeatures = features
labels = np.array(labels)

# Visualize the features using UMAP
reducer = umap.UMAP()
embedding = reducer.fit_transform(features)
plt.figure()
plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral')
plt.savefig("train_siamese.png")
plt.show()

model.eval()
# Extract features for visualization test
features = []
labels = []
for inputs, label in query_set:
    # Pass image through network
    inputs = inputs.unsqueeze(0)
    feature = model.backbone(inputs).detach().numpy()[0]

    # Append features and labels
    features.append(feature)
    labels.append(label)

# Convert to numpy arrays
features = np.array(features).reshape(len(features), -1)
queryFeatures = features
labels = np.array(labels)

# Visualize the features using UMAP
reducer = umap.UMAP()
embedding = reducer.fit_transform(features)
plt.figure()
plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral')
plt.savefig("test_siamese.png")
plt.show()

# Load the dataset and query labels
with open("features/query_labels.pkl", "rb") as f:
    query_labels = pickle.load(f)
with open("features/data_labels.pkl", "rb") as f:
    data_labels = pickle.load(f)

# Train the classifier
print("[INFO] Training with best K")
knn = KNeighborsClassifier(n_neighbors=5, metric="manhattan")
knn = knn.fit(dataFeatures, data_labels)
neighbors = knn.kneighbors(queryFeatures, return_distance=False)

# Get the labels obtained by classifier
predictions = []
for i in range(len(neighbors)):
    neighbors_class = [data_labels[j][1] for j in neighbors[i]]
    predictions.append(neighbors_class)

# Evaluate the model
print("[INFO] Evaluating..")
ground_truth = [x[1] for x in query_labels]
p_1 = mpk(ground_truth, predictions, 1)
p_5 = mpk(ground_truth, predictions, 5)

print('P@1=', p_1)
print('P@5=', p_5)
map = mAP(ground_truth, predictions)
print('mAP=', map)

# Convert ground truth and predictions to binary format
num_classes = len(set(labels))
binary_ground_truth = label_binarize(ground_truth, classes=range(num_classes))
binary_predictions = []

for pred in predictions:
    binary_pred = label_binarize(pred, classes=range(num_classes))
    binary_predictions.append(binary_pred.mean(axis=0))

binary_predictions = np.array(binary_predictions)

# Calculate precision-recall curve
precision, recall, _ = precision_recall_curve(binary_ground_truth.ravel(), binary_predictions.ravel())

# Plot precision-recall curve
plt.figure()
plt.plot(recall, precision, marker='.', label='KNN Classifier')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.grid()
plt.savefig("Siameseroc.png")
