import os
from glob import glob
import random
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np


import json
import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch
import detectron2
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import build_detection_train_loader
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, SemSegEvaluator
from detectron2.engine import DefaultPredictor, DefaultTrainer, HookBase
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.model_zoo import model_zoo
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from pycocotools import coco

from kitti_mots import register_kitti_mots_dataset
from kitti_mots import get_kitti_mots_dicts
from detectron2_helpers import ValidationLoss
from detectron2_helpers import plot_losses
from detectron2_helpers import show_results


from detectron2.evaluation import COCOEvaluator, inference_on_dataset

'''Register the Kitti-Mots dataset'''
register_kitti_mots_dataset(
    "/home/mcv/datasets/KITTI-MOTS/training/image_02",
    annots_path="/home/mcv/datasets/KITTI-MOTS/instances_txt",
    dataset_names=("k", "t"),
    image_extension="png",
)


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("k",)
cfg.DATASETS.TEST = ("t",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.OUTPUT_DIR = "output"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()


evaluator = COCOEvaluator("t", ["bbox"], False, output_dir="output", allow_cached_coco=False) #evaluate the model with COCO metrics
results_coco = trainer.test(cfg, trainer.model, evaluators=[evaluator]) # !! it evaliuates on the cfg test data
with open("output/evaluate.json", "w") as outfile:
    json.dump(results_coco, outfile)

