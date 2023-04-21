import time



import torch

import torch.nn as nn

from torchvision.datasets import CocoCaptions

import torchvision.models as models

import torchvision.transforms as transforms

from matplotlib import pyplot as plt

from torch.optim import SGD, Adam, Adagrad

from torch.utils.data import DataLoader

from sklearn import preprocessing

import fasttext

import numpy as np

import math



from packages import config 

from image_text_network import ImageToText

from packages.custom_tensor_dataset import CustomTensorDatasetTriplet_Image2Text

import logging



logging.basicConfig(level=logging.INFO)

# Set the device we will be using to train the model

device = torch.device("cuda")

dtype = torch.float





ft_model = fasttext.load_model("/home/mcv/m5/fasttext_wiki.en.bin")



# max length in train is 250

seq_len = 300

vocab_size = len(ft_model.get_words())

embedding_dim = ft_model.get_dimension()



# Define data transforms

transform = transforms.Compose([

    transforms.Resize(224),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

])



# https://pytorch.org/vision/main/generated/torchvision.datasets.CocoCaptions.html

# Load the COCO dataset

traindataset = CustomTensorDatasetTriplet_Image2Text(root_dir='/home/mcv/datasets/COCO/train2014',

                                 annFile='/home/mcv/datasets/COCO/captions_train2014.json', type_data="train", transforms=transform)

testdataset = CustomTensorDatasetTriplet_Image2Text(root_dir='/home/mcv/datasets/COCO/val2014',

                                annFile='/home/mcv/datasets/COCO/captions_val2014.json', type_data="val", transforms=transform)





trainLoader = DataLoader(traindataset, batch_size=config.BATCH_SIZE, shuffle=True)

testLoader = DataLoader(testdataset, batch_size=config.BATCH_SIZE, shuffle=True)







# Calculate steps per epoch for training and validation set

trainSteps = math.ceil(len(trainLoader) / config.BATCH_SIZE)

valSteps = math.ceil(len(testLoader) / config.BATCH_SIZE)



# Create the model

model = ImageToText().to(device)



# Define the optimizer

opt = SGD(model.parameters(), lr=config.INIT_LR, momentum=0.9)

#opt = Adam(model.parameters(), lr=config.INIT_LR, weight_decay=0.001)


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



# Define your early stopping variables
best_val_loss = float('inf')
early_stop_counter = 0
patience = config.PATIENCE # the number of epochs to wait before stopping the training process




# Loop over our epochs

for e in range(0, config.EPOCHS):

    logging.info("epoch "+str(e))

    # set the model in training mode

    model.train()



    # Initialize the total training and validation loss

    totalTrainLoss = 0

    totalValLoss = 0



    for (triplet, _) in trainLoader:

        # send the input to the device

        anchor, positive, negative = triplet



        positive_tensors = np.zeros((anchor.size(0), 300))

        for i, positive_string in enumerate(positive):

            positive_words = np.array([])

            input_tokens = positive_string.split()

            for token in input_tokens:

                if token.lower() in ft_model:

                    positive_words = np.append(positive_words, ft_model[token.lower()])

            positive_tensors[i] = np.mean(positive_words, axis=0)


        #positive_tensors = np.mean(positive_tensors, axis=1)

        positive_tensors = torch.from_numpy(positive_tensors)

        

        negative_tensors = np.zeros((anchor.size(0), 300))

        for i, negative_string in enumerate(negative):

            negative_words = np.array([])

            input_tokens = negative_string.split()

            for token in input_tokens:

                if token.lower() in ft_model:

                    negative_words = np.append(negative_words, ft_model[token.lower()])

            negative_tensors[i] = np.mean(negative_words, axis=0)

        #negative_tensors = np.mean(negative_tensors, axis=1)

        negative_tensors = torch.from_numpy(negative_tensors)




        anchor = anchor.to(device)

        positive = positive_tensors.to(torch.float32).to(device)

        negative = negative_tensors.to(torch.float32).to(device)


        anchor_embedding, positive_embedding, negative_embedding = model(anchor, positive, negative)
        #print("positive_embedding",positive_embedding)
        #print("negative_embedding",negative_embedding)


        loss = triplet_loss_fn(anchor_embedding, positive_embedding, negative_embedding)

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





            positive_tensors = np.zeros((anchor.size(0), 300))

            for i, positive_string in enumerate(positive):

                positive_words = np.array([])

                input_tokens = positive_string.split()

                for token in input_tokens:

                    if token.lower() in ft_model:

                        positive_words = np.append(positive_words, ft_model[token.lower()])

                positive_tensors[i] = np.mean(positive_words, axis=0)


            #positive_tensors = np.mean(positive_tensors, axis=1)

            positive_tensors = torch.from_numpy(positive_tensors)



            negative_tensors = np.zeros((anchor.size(0), 300))

            for i, negative_string in enumerate(negative):

                negative_words = np.array([])

                input_tokens = negative_string.split()

                for token in input_tokens:

                    if token.lower() in ft_model:

                        negative_words = np.append(negative_words, ft_model[token.lower()])

                negative_tensors[i] = np.mean(negative_words, axis=0)

            #negative_tensors = np.mean(negative_tensors, axis=1)

            negative_tensors = torch.from_numpy(negative_tensors)

   



            anchor = anchor.to(device)

            positive = positive_tensors.to(torch.float32).to(device)

            negative = negative_tensors.to(torch.float32).to(device)



            # Forward + backward + optimize

            anchor_embedding, positive_embedding, negative_embedding = model(anchor, positive, negative)
            #print("positive_embedding_val",positive_embedding)
            #print("negative_embedding_val",negative_embedding)

            loss = triplet_loss_fn(anchor_embedding, positive_embedding, negative_embedding)

            totalValLoss += loss.item()



    # calculate the average training and validation loss

    avgTrainLoss = totalTrainLoss / trainSteps

    avgValLoss = totalValLoss / valSteps

    logging.info("avgTrainLoss "+str(avgTrainLoss))
    logging.info("avgValLoss "+str(avgValLoss))

    # update our training history

    H["train_loss"].append(avgTrainLoss)

    H["val_loss"].append(avgValLoss)



    # print the model training and validation information

    print("[INFO] EPOCH: {}/{}".format(e + 1, config.EPOCHS))

    print("Train loss: {:.6f}, Val Loss: {:.4f}".format(avgTrainLoss, avgValLoss))

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model,'/ghome/group06/m5/w5/weights_image_text.pth')
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    # Check if the training process should be stopped
    #if early_stop_counter >= patience:
    #    print('Early stopping after {} epochs without improvement.'.format(patience))
    #    break





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