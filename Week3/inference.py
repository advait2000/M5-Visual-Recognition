# Import required packages
import cv2
from imutils import paths
import argparse
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# Construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the images to inferred")
args = vars(ap.parse_args())

# Initialize the configs and load model weights
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

# Initialize the predictor
predictor = DefaultPredictor(cfg)

# Loop over the dataset
for i, path in enumerate(sorted(paths.list_images(args["dataset"]))):
    # Read image and pass through model
    im = cv2.imread(path)
    outputs = predictor(im)

    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Save the image
    cv2.imwrite("output{}.png".format(i), out.get_image()[:, :, ::-1])
    print("Saved {}".format(path))
