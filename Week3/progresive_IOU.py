# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# Import necessary libraries
from PIL import Image
import numpy as np
import cv2
from imutils import paths
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from pycocotools.coco import COCO
import random
import matplotlib.pyplot as plt



def get_png(img):
    # Load the image and its alpha channel mask
    alpha = img.split()[-1]

    # Create a new image with a black background
    bg = Image.new('RGBA', img.size, (0, 0, 0, 255))

    # Paste the original image onto the new image using the alpha channel as a mask
    bg.paste(img, mask=alpha)

    # Convert the PIL image to a numpy array
    img_array = np.array(bg)

    # Convert the numpy array to a cv2 image
    cv2_image = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

    return cv2_image


# Initialize detectron2 configuration and load model weights
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.MODEL.DEVICE = 'cuda'

# Set the path for COCO test dataset and load its annotations
image = "/ghome/group06/m5/w3/000000000108.jpg"
object_image = "/ghome/group06/m5/w3/000000000108.png"

# Initialize the predictor
predictor = DefaultPredictor(cfg)

background  = cv2.imread(image)
overlay = cv2.imread(object_image, cv2.IMREAD_UNCHANGED) # IMREAD_UNCHANGED => open image with the alpha channel
height, width, _ = background.shape
tx = 150  # set the amount of pixels to translate horizontally
ty = 0    # set the amount of pixels to translate vertically
M = np.float32([[1, 0, -tx], [0, 1, -ty]])

# Apply the translation to the stacked image
overlay = cv2.warpAffine(overlay, M, (width, height))

# separate the alpha channel from the color channels
alpha_channel = overlay[:, :, 3] / 255 # convert from 0-255 to 0.0-1.0
overlay_colors = overlay[:, :, :3]

# To take advantage of the speed of numpy and apply transformations to the entire image with a single operation
# the arrays need to be the same shape. However, the shapes currently looks like this:
#    - overlay_colors shape:(width, height, 3)  3 color values for each pixel, (red, green, blue)
#    - alpha_channel  shape:(width, height, 1)  1 single alpha value for each pixel
# We will construct an alpha_mask that has the same shape as the overlay_colors by duplicate the alpha channel
# for each color so there is a 1:1 alpha channel for each color channel
alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

# The background image is larger than the overlay so we'll take a subsection of the background that matches the
# dimensions of the overlay.
# NOTE: For simplicity, the overlay is applied to the top-left corner of the background(0,0). An x and y offset
# could be used to place the overlay at any position on the background.
h, w = overlay.shape[:2]
background_subsection = background[0:h, 0:w]

# combine the background with the overlay image weighted by alpha
composite = background_subsection * (1 - alpha_mask) + overlay_colors * alpha_mask

# overwrite the section of the background image that has been updated
background[0:h, 0:w] = composite

plt.figure(figsize=(15, 7.5))
plt.imshow(background[..., ::-1])

outputs = predictor(background[..., ::-1])
print(outputs)
v = Visualizer(background[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imwrite("output.png", out.get_image()[:, :, ::-1])
plt.figure(figsize=(20, 10))
plt.imshow(out.get_image()[..., ::-1][..., ::-1])
