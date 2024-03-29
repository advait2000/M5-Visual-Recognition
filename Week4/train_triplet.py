# Import required packages
import os
import pickle
import time
import torch
import umap
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from packages import CustomTensorDatasetTripletLoss
from packages import config
from triplet_network import TripletNetwork
from average_precision import *

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


def hard_negative_mining(anchor, positive, negative, negative_ratio=0.5):
    # Calculate the distance between anchor and all negative examples
    all_neg_dist = (anchor - negative).pow(2).sum(-1)

    # Find the hardest negative examples
    num_negative = int(negative_ratio * positive.size(0))
    _, top_idx = torch.topk(all_neg_dist, k=num_negative, largest=False)

    # Select the hardest negative examples
    hard_negative = negative[top_idx, :]

    # Calculate the triplet loss using the hardest negative examples
    loss = criterion(anchor, positive, hard_negative)

    return loss


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

    # Initialize the total training and validation accuracy
    totalTrainAccuracy = 0
    totalValAccuracy = 0

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
        pos_dist = (anchor_embedding - positive_embedding).pow(2).sum(-1).sqrt()
        neg_dist = (anchor_embedding - negative_embedding).pow(2).sum(-1).sqrt()
        distances = pos_dist - neg_dist

        totalTrainAccuracy += torch.sum(distances < 1.0).item()

    # Switch off autograd for evaluation
    with torch.no_grad():
        # Set the model in evaluation mode
        model.eval()

        for (triplet, _) in testLoader:
            # Send the input to the device
            anchor, positive, negative = triplet
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            # Forward + backward + optimize
            anchor_embedding, positive_embedding, negative_embedding = model(anchor, positive, negative)
            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            totalValLoss += loss.item()
            pos_dist = (anchor_embedding - positive_embedding).pow(2).sum(-1).sqrt()
            neg_dist = (anchor_embedding - negative_embedding).pow(2).sum(-1).sqrt()
            distances = pos_dist - neg_dist

            totalValAccuracy += torch.sum(distances < 1.0).item()

    # Calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps
    avgTrainAccuracy = totalTrainAccuracy / trainSteps
    avgValAccuracy = totalValAccuracy / valSteps

    # Update our training history
    H["train_loss"].append(avgTrainLoss)
    H["val_loss"].append(avgValLoss)
    H["train_acc"].append(avgTrainAccuracy)
    H["val_acc"].append(avgValAccuracy)

    # Print the model training and validation information
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

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_acc"], label="train_acc")
plt.plot(H["val_acc"], label="val_acc")
plt.title("Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("acctriplet.png")

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
    feature = model.resnet(inputs).detach().numpy()[0]

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
plt.savefig("train_triplet.png")
plt.show()

model.eval()
# Extract features for visualization test
features = []
labels = []
for inputs, label in query_set:
    # Pass image through network
    inputs = inputs.unsqueeze(0)
    feature = model.resnet(inputs).detach().numpy()[0]

    # Append features and labels
    features.append(feature)
    labels.append(label)

# Convert to numpy arrays
features = np.array(features).reshape(len(features), -1)
queryFeatures = features
labels = np.array(labels)

# Visualize the features using UMAP
umap_obj = umap.UMAP()
umap_embedding = umap_obj.fit_transform(queryFeatures)
unique_labels = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'opencountry', 'street', 'tallbuilding']
colors = [plt.cm.tab10(i) for i in np.linspace(0, 1, len(unique_labels))]
plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=labels)
plt.title('UMAP')
plt.savefig("Tripletumap2d.png")

tsne_results = TSNE().fit_transform(queryFeatures)
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels)
unique_labels = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'opencountry', 'street', 'tallbuilding']
colors = [plt.cm.tab10(i) for i in np.linspace(0, 1, len(unique_labels))]
plt.title('TSNE')
plt.savefig("Triplettsne2d.png")

umap_obj = umap.UMAP(n_components=3)
umap_embedding = umap_obj.fit_transform(queryFeatures)
ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], umap_embedding[:, 2], c=labels)
unique_labels = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'opencountry', 'street', 'tallbuilding']
colors = [plt.cm.tab10(i) for i in np.linspace(0, 1, len(unique_labels))]
plt.title('UMAP')
plt.savefig("Tripletumap3d.png")

tsne_results = TSNE(n_components=3).fit_transform(queryFeatures)
ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], c=labels)
unique_labels = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'opencountry', 'street', 'tallbuilding']
colors = [plt.cm.tab10(i) for i in np.linspace(0, 1, len(unique_labels))]
plt.title('TSNE')
plt.savefig("Triplettsne3d.png")

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
plt.savefig("TripletRoc.png")

# Calculate the ROC curve and the AUC score for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(binary_ground_truth[:, i], binary_predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot the ROC curve for each class in the same figure
plt.figure()

for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for All Classes')
plt.legend(loc='best')
plt.grid()
plt.savefig("Triplet_roc_curve_all_classes.png")
