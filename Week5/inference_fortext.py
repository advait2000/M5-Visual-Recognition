import json

import torch
import torchvision.transforms as transforms
import fasttext
import numpy as np
from PIL import Image
from text_image_network import TextToImage
import os
from packages.custom_tensor_dataset import CustomTensorDatasetTriplet_Text2Image

# Load FastText model
ft_model = fasttext.load_model("wiki.en.bin")

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

# Load the trained model
model = torch.load('weights_text_image300.pth', map_location=torch.device('cpu'))

testdataset = CustomTensorDatasetTriplet_Text2Image(root_dir='val2014', annFile='captions_val2014.json',
                                                    type_data="val", transforms=transform)


def get_best_images(text, image_paths, top_k=5):
    # Embed the input text
    input_tokens = text.split()
    input_words = []
    for token in input_tokens:
        if token.lower() in ft_model:
            input_words.append(ft_model[token.lower()])

    input_words = np.array(input_words)
    text_embedding = np.mean(input_words, axis=0)
    text_tensor = torch.from_numpy(text_embedding[np.newaxis, :]).to(torch.float32)

    # Compute scores for each image
    scores = []
    with torch.no_grad():
        model.eval()

        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0)

            text_val, image_val_p, _ = model(text_tensor, image_tensor, image_tensor)
            score = torch.dist(text_val, image_val_p)
            scores.append((score.item(), image_path))

    # Sort scores and get top k images
    best_images = sorted(scores, key=lambda x: x[0])[:top_k]

    return [image_path for _, image_path in best_images]


input_text = "An office cubicle with four different types of computers."  # Your input text here

annFile = 'captions_val2014.json'
with open(annFile, 'r') as f:
    captions_data = json.load(f)

# Extract the image IDs
image_ids = [entry['image_id'] for entry in captions_data['annotations']][:500]

# Get the list of image paths from the COCO validation dataset
image_paths = [os.path.join('val2014', 'COCO_val2014_{:012d}.jpg'.format(image_id)) for
               image_id in image_ids]
top_k = 5

best_images = get_best_images(input_text, image_paths, top_k)

print("Top {} images for the input text '{}':".format(top_k, input_text))
for idx, image_path in enumerate(best_images):
    print("{}. {}".format(idx + 1, image_path))

    # Find the captions associated with the top images
    top_captions = []
    for best_image_id in best_images:
        captions = [entry['caption'] for entry in captions_data['annotations'] if entry['image_id'] == best_image_id]
        top_captions.append(captions)

    print("\nCaptions for the top {} images:".format(top_k))
    for idx, captions in enumerate(top_captions):
        print("{}. Image ID: {}".format(idx + 1, best_images[idx]))
        for caption in captions:
            print("   - {}".format(caption))
