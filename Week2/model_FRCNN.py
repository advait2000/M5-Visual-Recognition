import torch
import detectron2
from detectron2.model_zoo import model_zoo
from detectron2.config import get_cfg
import os
import cv2
from glob import glob

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from pycocotools import coco
import numpy as np
import detectron2.utils.comm as comm 
from detectron2.engine import HookBase  # For making hooks
from detectron2.data import (
    build_detection_train_loader,
)  # dataloader is the object that provides the data to the models
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator

from detectron2.engine import DefaultTrainer
import copy
import os
import json
import random
import cv2
import matplotlib.pyplot as plt

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer


KITTI_CORRESPONDENCES = {"Car": 0, "Pedestrian": 1}


def get_kitti_mots_dicts(images_folder, annots_folder, is_train, train_percentage=0.75, image_extension="png"):
    """
    Converts KITTI-MOTS annotations to COCO format and returns a list of dictionaries

    Args:
        images_folder (str): Path to the folder containing images
        annots_folder (str): Path to the folder containing annotations
        is_train (bool): True if creating training data, False otherwise
        train_percentage (float, optional): Percentage of sequences to use for training. Defaults to 0.75.
        image_extension (str, optional): Extension of image files. Defaults to "jpg".

    Returns:
        List[Dict]: A list of dictionaries where each dictionary contains information about an image
    """
    assert os.path.exists(images_folder)
    assert os.path.exists(annots_folder)

    annot_files = sorted(glob(os.path.join(annots_folder, "*.txt")))

    n_train_seqences = int(len(annot_files) * train_percentage)
    train_sequences = annot_files[:n_train_seqences]
    test_sequences = annot_files[n_train_seqences:]

    sequences = train_sequences if is_train else test_sequences

    kitti_mots_annotations = []
    for seq_file in sequences:
        seq_images_path = os.path.join(images_folder, seq_file.split("/")[-1].split(".")[0])
        kitti_mots_annotations += mots_annots_to_coco(seq_images_path, seq_file, image_extension)

    return kitti_mots_annotations


def mots_annots_to_coco(images_path, txt_file, image_extension):
    assert os.path.exists(txt_file)

    # Define the correspondences between class names and class IDs.

    correspondences = KITTI_CORRESPONDENCES

    # Extract the sequence number from the text file name.
    n_seq = int(txt_file.split("/")[-1].split(".")[0])

    mots_annots = []
    with open(txt_file, "r") as f:
        annots = f.readlines()
        annots = [l.split() for l in annots]

        annots = np.array(annots)

        # Iterate over frames in the sequence.
        for frame in np.unique(annots[:, 0].astype("int")):

            # Extract annotations for the current frame.
            frame_lines = annots[annots[:, 0] == str(frame)]
            if frame_lines.size > 0:

                # Extract the image height and width from the first annotation in the frame.
                h, w = int(frame_lines[0][3]), int(frame_lines[0][4])

                f_objs = []
                for a in frame_lines:
                    cat_id = int(a[2]) - 1
                    # Skip annotations that correspond to non-existent classes.
                    if cat_id in correspondences.values():
                        # Extract segmentation information from the annotation.
                        segm = {
                            "counts": a[-1].strip().encode(encoding="UTF-8"),
                            "size": [h, w],
                        }

                        # Convert the segmentation mask to a bounding box.
                        box = coco.maskUtils.toBbox(segm)
                        box[2:] = box[2:] + box[:2]
                        box = box.tolist()

                        # Convert the segmentation mask to a polygon.
                        mask = np.ascontiguousarray(coco.maskUtils.decode(segm))
                        contours, _ = cv2.findContours(
                            mask.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                        )
                        poly = []
                        for contour in contours:
                            contour = contour.flatten().tolist()
                            if len(contour) > 4:
                                poly.append(contour)
                        if len(poly) == 0:
                            continue

                        # Create an annotation dictionary for the current object.
                        annot = {
                            "category_id": cat_id,
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "bbox": box,
                            "segmentation": poly,
                        }
                        f_objs.append(annot)

                # Create a dictionary for the current frame.
                frame_data = {
                    "file_name": os.path.join(
                        images_path, "{:06d}.{}".format(int(a[0]), image_extension)
                    ),
                    "image_id": int(frame + n_seq * 1e6),
                    "height": h,
                    "width": w,
                    "annotations": f_objs,
                }
                mots_annots.append(frame_data)

    return mots_annots


"""
Registering function
"""


def register_kitti_mots_dataset(
    ims_path, annots_path, dataset_names, train_percent=0.75, image_extension="png"
):
    assert isinstance(
        dataset_names, tuple
    ), "dataset names should be a tuple with two strings (for train and test)"

    def kitti_mots_train():
        return get_kitti_mots_dicts(
            ims_path,
            annots_path,
            is_train=True,
            train_percentage=train_percent,
            image_extension=image_extension,
        )

    def kitti_mots_test():
        return get_kitti_mots_dicts(
            ims_path,
            annots_path,
            is_train=False,
            train_percentage=train_percent,
            image_extension=image_extension,
        )

    DatasetCatalog.register(dataset_names[0], kitti_mots_train)
    MetadataCatalog.get(dataset_names[0]).set(
        thing_classes=[k for k, v in KITTI_CORRESPONDENCES.items()]
    )
    DatasetCatalog.register(dataset_names[1], kitti_mots_test)
    MetadataCatalog.get(dataset_names[1]).set(
        thing_classes=[k for k, v in KITTI_CORRESPONDENCES.items()]
    )


register_kitti_mots_dataset(
    "/Users/eduard.hogea/Documents/Facultate/Erasmus/UAB_sem2/M5/M5-Visual-Recognition/datasets/KITTI_MOTS/data_tracking_image_2/training/image_02",
    annots_path="/Users/eduard.hogea/Documents/Facultate/Erasmus/UAB_sem2/M5/M5-Visual-Recognition/datasets/KITTI_MOTS/instances_txt",
    dataset_names=("k", "t"),
    image_extension="png",
)


'''Updating the config file'''

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
)
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
)

cfg.DATASETS.TRAIN = "k"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # number of classes
cfg.DATASETS.TEST = "t"

cfg.DATALOADER.NUM_WORKERS = 0
cfg.SOLVER.IMS_PER_BATCH = 4  # images per batch
# cfg.SOLVER.BASE_LR = 0.0002 * cfg.SOLVER.IMS_PER_BATCH * 1.4 / 16 # learning rate
cfg.SOLVER.BASE_LR = 0.0002 * 2 * 1.4 / 16  # learning rate
cfg.SOLVER.MAX_ITER = 52  # maximum number of iterations. Change if needed.
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # batch size per image
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
    0.5  # threshold used to filter out low-scored bounding boxes in predictions
)

cfg.MODEL.DEVICE = "cpu"

cfg.OUTPUT_DIR = "output_week2"  # output directory
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()  # takes init from HookBase
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
        self._loader = iter(
            build_detection_train_loader(self.cfg)
        )  # builds the dataloader from the provided cfg
        self.best_loss = float("inf")  # Current best loss, initially infinite
        self.weights = None  # Current best weights, initially none
        self.i = 0  # Something to use for counting the steps

    def after_step(self):  # after each step

        if self.trainer.iter >= 0:
            print(
                f"----- Iteration num. {self.trainer.iter} -----"
            )  # print the current iteration if it's divisible by 100

        data = next(self._loader)  # load the next piece of data from the dataloader

        with torch.no_grad():  # disables gradient calculation; we don't need it here because we're not training, just calculating the val loss
            loss_dict = self.trainer.model(data)  # more about it in the next section

            losses = sum(loss_dict.values())  #
            assert torch.isfinite(losses).all(), loss_dict
            loss_dict_reduced = {
                "val_" + k: v.item() for k, v in comm.reduce_dict(loss_dict).items()
            }
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            if comm.is_main_process():
                self.trainer.storage.put_scalars(
                    total_val_loss=losses_reduced, **loss_dict_reduced
                )  # puts these metrics into the storage (where detectron2 logs metrics)

                # save best weights
                if losses_reduced < self.best_loss:  # if current loss is lower
                    self.best_loss = losses_reduced  # saving the best loss
                    self.weights = copy.deepcopy(
                        self.trainer.model.state_dict()
                    )  # saving the best weights


"""Training part using hooks"""



trainer = DefaultTrainer(cfg)

val_loss = ValidationLoss(cfg)
trainer.register_hooks([val_loss])

trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]

trainer.resume_or_load(resume=True)
trainer.train()

cfg.MODEL.WEIGHTS = "/Users/eduard.hogea/Documents/Facultate/Erasmus/UAB_sem2/M5/M5-Visual-Recognition/output_week2/mymodelfull.pth"
trainer = DefaultTrainer(cfg)

val_loss = ValidationLoss(cfg)
trainer.register_hooks([val_loss])

trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]

trainer.resume_or_load(resume=True)

val_loss = ValidationLoss(cfg)
trainer.register_hooks([val_loss])

trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]

trainer.resume_or_load(resume=True)
trainer.train()


torch.save(trainer.model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "mymodelfull.pth"))



checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)


"""
Visualization
"""


def plot_losses(cfg):

    val_loss = []
    train_loss = []
    for line in open(os.path.join(cfg.OUTPUT_DIR, "metrics.json"), "r"):
        if (
            "total_val_loss" in json.loads(line).keys()
            and "total_loss" in json.loads(line).keys()
        ):
            val_loss.append(json.loads(line)["total_val_loss"])
            train_loss.append(json.loads(line)["total_loss"])

    plt.plot(val_loss, label="Validation Loss")
    plt.plot(train_loss, label="Training Loss")
    plt.legend()
    plt.show()


def show_results(cfg, dataset_dicts, predictor, samples=10):

    for data in random.sample(dataset_dicts, samples):
        im = cv2.imread(data["file_name"])
        outputs = predictor(im)
        # print(outputs)

        # outputs["instances"] = outputs["instances"][torch.where(outputs["instances"].pred_classes < 2)]
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get("t"), scale=0.5)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("Frame", v.get_image()[:, :, ::-1])
        cv2.waitKey(0)


plot_losses(cfg)

evaluator = COCOEvaluator("t", cfg, False, output_dir="output_week2")

predictor = DefaultPredictor(cfg)
predictor.model.load_state_dict(trainer.model.state_dict())

dataset_dicts = get_kitti_mots_dicts(
    "/Users/eduard.hogea/Documents/Facultate/Erasmus/UAB_sem2/M5/M5-Visual-Recognition/datasets/KITTI_MOTS/data_tracking_image_2/training/image_02",
    "/Users/eduard.hogea/Documents/Facultate/Erasmus/UAB_sem2/M5/M5-Visual-Recognition/datasets/KITTI_MOTS/instances_txt",
    is_train=False,
    image_extension="png",
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
show_results(cfg, dataset_dicts, predictor, samples=10)
