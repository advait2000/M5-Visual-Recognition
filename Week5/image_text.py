import time

import torch
import torch.nn as nn
from torchvision.datasets import CocoCaptions
import torchvision.models as models
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.optim import SGD

from packages import config

# Set the device we will be using to train the model
device = torch.device("cpu")
dtype = torch.float

# Define data transforms
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# https://pytorch.org/vision/main/generated/torchvision.datasets.CocoCaptions.html
# Load the COCO dataset
traindataset = CocoCaptions(root='coco/train',
                                 annFile='coco/annotations/captions_train2014.json', transform=transform)
testdataset = CocoCaptions(root='coco/val',
                                annFile='coco/annotations/captions_val2014.json', transform=transform)

# Calculate steps per epoch for training and validation set
trainSteps = len(traindataset) // config.BATCH_SIZE
valSteps = len(testdataset) // config.BATCH_SIZE

# Define the ResNet-50 model
resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])


# Define the image-to-text retrieval model
class ImageToText(nn.Module):
    def __init__(self, resnet):
        super(ImageToText, self).__init__()
        self.resnet = resnet
        self.embedding = nn.Linear(2048, 512)

    def forward(self, images, captions):
        image_embeddings = self.resnet(images)
        image_embeddings = image_embeddings.view(image_embeddings.size(0), -1)
        image_embeddings = self.embedding(image_embeddings)

        caption_embeddings = self.embedding(captions)

        return image_embeddings, caption_embeddings


# Create the model
model = ImageToText(resnet).to(device)

# Define the optimizer
optimizer = SGD(model.parameters(), lr=config.INIT_LR, momentum=0.9)

# Define the triplet loss function
triplet_loss_fn = nn.TripletMarginLoss(margin=1.0)

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


# Train the model
for epoch in range(config.EPOCHS):
    # set the model in training mode
    model.train()

    # Initialize the total training and validation loss
    totalTrainLoss = 0
    totalValLoss = 0

    # Initialize batches
    train_batches = 0

    for i in range(0, len(traindataset), config.BATCH_SIZE):
        # Divide into batches
        batch = traindataset[i:i + config.BATCH_SIZE]
        images = torch.stack([x[0] for x in batch]).to(device)
        captions = torch.stack([x[1] for x in batch]).to(device)

        # send the input to the device
        anchor_images, positive_images, negative_images = images.chunk(3, dim=0)
        anchor_captions, positive_captions, negative_captions = captions.chunk(3, dim=0)

        # Forward + backward + optimize
        optimizer.zero_grad()
        anchor_image_embeddings, anchor_caption_embeddings = model(anchor_images, anchor_captions)
        positive_image_embeddings, positive_caption_embeddings = model(positive_images, positive_captions)
        negative_image_embeddings, negative_caption_embeddings = model(negative_images, negative_captions)

        image_loss = triplet_loss_fn(anchor_image_embeddings, positive_image_embeddings, negative_image_embeddings)
        caption_loss = triplet_loss_fn(anchor_caption_embeddings, positive_caption_embeddings, negative_caption_embeddings)
        loss = image_loss + caption_loss
        loss.backward(retain_graph=True)
        optimizer.step()

        totalTrainLoss += loss.item()
        train_batches += 1

    # Switch off autograd for evaluation
    with torch.no_grad():
        # Set the model in evaluation mode
        model.eval()
        test_batches = 0
        for i in range(0, len(testdataset), config.BATCH_SIZE):
            batch = testdataset[i:i + config.BATCH_SIZE]
            images = torch.stack([x[0] for x in batch]).to(device)
            captions = torch.stack([x[1] for x in batch]).to(dev)
            anchor_images, positive_images, negative_images = images.chunk(3, dim=0)
            anchor_captions, positive_captions, negative_captions = captions.chunk(3, dim=0)

            anchor_image_embeddings, anchor_caption_embeddings = model(anchor_images, anchor_captions)
            positive_image_embeddings, positive_caption_embeddings = model(positive_images, positive_captions)
            negative_image_embeddings, negative_caption_embeddings = model(negative_images, negative_captions)

            # calculate the triplet loss for the images
            image_loss = triplet_loss_fn(anchor_image_embeddings, positive_image_embeddings, negative_image_embeddings)
            # calculate the triplet loss for the captions
            caption_loss = triplet_loss_fn(anchor_caption_embeddings, positive_caption_embeddings,
                                           negative_caption_embeddings)

            loss = image_loss + caption_loss

            totalValLoss += loss.item()
            test_batches += 1

    # Calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps

    # Update our training history
    H["train_loss"].append(avgTrainLoss)
    H["val_loss"].append(avgValLoss)

    # Print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(epoch + 1, config.EPOCHS))
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
plt.savefig("loss.png")

# finish measuring how long training took
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
