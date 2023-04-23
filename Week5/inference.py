import sys
import torch
import torchvision.transforms as transforms
import fasttext
import numpy as np
import logging
import json
from PIL import Image
from image_text_network import ImageToText
from packages import config

logging.basicConfig(level=logging.INFO)

# Set the device we will be using to train the model
device = torch.device("cpu")
dtype = torch.float


def read_annotations(annFile, max_it):
    f = open(annFile)
    captionJson = json.load(f)
    f.close()

    ImageNcaption = []
    for i, caption in enumerate(captionJson["annotations"]):
        ImageNcaption.append([str(caption["image_id"]), str(caption["caption"])])
        if i >= max_it:
            break
    return ImageNcaption


# Load FastText model
ft_model = fasttext.load_model("wiki.en.bin")

# Define data transforms
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load the trained model
model = torch.load('weights_image_text.pth', map_location=torch.device('cpu')).to(device)


def get_best_captions(image_path, top_k=5):
    # Read image
    image = Image.open(image_path).convert('RGB')
    anchor = transform(image).unsqueeze(0).to(device)

    ImageNcaption = read_annotations('captions_val2014.json', max_it=39)

    with torch.no_grad():
        model.eval()

        scores = []
        for _, caption in ImageNcaption:
            positive_tensors = np.zeros((1, 300))

            positive_words = np.array([])
            input_tokens = caption.split()

            for token in input_tokens:
                if token.lower() in ft_model:
                    positive_words = np.append(positive_words, ft_model[token.lower()])

            positive_tensors[0] = np.mean(positive_words, axis=0)

            positive_tensors = torch.from_numpy(positive_tensors)
            positive = positive_tensors.to(torch.float32).to(device)

            image_val_o, text_val, _ = model(anchor, positive, positive)
            score = torch.dist(image_val_o, text_val)
            scores.append((score.item(), caption))

    # Sort scores and get top k captions
    best_captions = sorted(scores, key=lambda x: x[0])[:top_k]

    return [caption for _, caption in best_captions]


image_path = '/Users/advaitdixit/Documents/Masters/M5-Visual-Recognition/Week5/val2014/COCO_val2014_000000000641.jpg'  # Path to the image
top_k = 5

best_captions = get_best_captions(image_path, top_k)

print("Top {} captions for image {}:".format(top_k, image_path))
for idx, caption in enumerate(best_captions):
    print("{}. {}".format(idx + 1, caption))
