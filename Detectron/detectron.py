import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt

import torch
from detectron2.structures import BoxMode

new_dataset_path = "./new_ds"

def get_balloon_images(img_folder,d):
    json_file = os.path.join(img_folder, d+".json")
    with open(json_file) as f:
        imgs_anns = json.load(f)
    dataset_dicts = []
    objs = []
    for idx in range(len(imgs_anns['images'])):
        record = {}
        filename = os.path.join(img_folder, imgs_anns['images'][idx]["file_name"])
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = imgs_anns['images'][idx]["height"]
        record["width"] = imgs_anns['images'][idx]["width"]


        annos = imgs_anns["annotations"][idx]
        obj = {
            "bbox": annos['bbox'],
            "segmentation": annos['segmentation'],
            "category_id": annos['category_id'],
            "bbox_mode": BoxMode.XYWH_ABS
        }
        objs.append(obj)
    record["annotations"] = objs
    dataset_dicts.append(record)
    return dataset_dicts

for d in ["train", "val"]:
    DatasetCatalog.register("new_ds_" + d, lambda d=d: get_balloon_images(new_dataset_path+"/"+d,d))
    MetadataCatalog.get("new_ds_" + d).set(thing_classes=["dummy","billete","knife","monedero","pistol","smartphone","tarjeta"])
new_ds_metadata = MetadataCatalog.get("new_ds_train")

from detectron2.engine import DefaultTrainer
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("new_ds_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 50
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 100 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

torch.save(trainer.model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "mymodel.pth"))