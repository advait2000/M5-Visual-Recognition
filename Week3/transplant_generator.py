# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# Import necessary libraries
import fiftyone as fo
import fiftyone.zoo as foz
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

# Initialize detectron2 configuration and load model weights
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.MODEL.DEVICE = 'cpu'

# Set the path for COCO test dataset and load its annotations
coco_test_dataset_path = "path/to/coco/test/dataset"
coco_test_annotations_path = "/Users/eduard.hogea/Documents/Facultate/Erasmus/UAB_sem2/M5/M5-Visual-Recognition/datasets/annotations/image_info_test2017.json"
coco_test_annotations = COCO(coco_test_annotations_path)

# Get the IDs of all images in the test dataset
test_image_ids = coco_test_annotations.getImgIds()

# Set a seed for reproducibility and get 10 random images from the test dataset
random.seed(11)
random_samples_ids_cut = random.sample(test_image_ids, 10)
random_samples_ids_target = random.sample(test_image_ids, 1)

# Initialize the predictor
predictor = DefaultPredictor(cfg)

# Set the desired class ID and the number of samples per class
class_id = 1
samples_per_class = 10

# Loop over all images in the test dataset
for image_id in test_image_ids:
    # Load the image and its annotations
    image_info = coco_test_annotations.loadImgs(image_id)[0]
    image = cv2.imread("/Users/eduard.hogea/Documents/Facultate/Erasmus/UAB_sem2/M5/M5-Visual-Recognition/datasets/test2017/" + image_info['file_name'])
    
    # Perform object detection and segmentation on the loaded image
    outputs = predictor(image)
    masks = outputs["instances"].pred_masks.cpu().numpy()
    class_ids = outputs["instances"].pred_classes.cpu().numpy()

    # Check if the image contains at least one instance of the desired class
    if any(ann == class_id for ann in class_ids):
        # Select object masks for the desired class
        object_masks = masks[class_ids == class_id].astype(np.uint8)
        object_mask = object_masks[0]

        # Get a random image from the test dataset to use as the target image
        random_sample_id_target = random.sample(test_image_ids, 1)
        image_info_target = coco_test_annotations.loadImgs(random_sample_id_target)[0]
        target_image = cv2.imread("/Users/eduard.hogea/Documents/Facultate/Erasmus/UAB_sem2/M5/M5-Visual-Recognition/datasets/test2017/" + image_info_target['file_name'])
        
        # Resize the object mask to fit onto the target image
        target_image_height, target_image_width, _ = target_image.shape
        object_height, object_width = object_mask.shape
        scale_factor = min(target_image_height/object_height, target_image_width/object_width) * 0.5
        object_mask_resized = cv2.resize(object_mask, (int(object_width*scale_factor), int(object_height*scale_factor)))

        # Find coordinates to paste object in target image
        y_min_target, x_min_target = int((target_image_height-object_mask_resized.shape[0])/2), int((target_image_width-object_mask_resized.shape[1])/2)
        y_max_target, x_max_target = y_min_target+object_mask_resized.shape[0], x_min_target+object_mask_resized.shape[1]

        # Paste object onto target image
        mask = (object_mask_resized == 0)
        target_image[y_min_target:y_max_target, x_min_target:x_max_target][mask == 0] = image[y_min_target:y_max_target, x_min_target:x_max_target][mask == 0]

        # Save output image
        cv2.imwrite("output_week3/output_image_" + str(image_id) + "_" +  ".jpg", target_image)
