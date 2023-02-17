# -*- coding: utf-8 -*-

# detectron2_train_eval.py - Python script to set DETECTRON2 Training and Evaluation configs for Colony Picking (COPICK) dataset. 

# GENERAL DESCRIPTION:
# -------------------
# When the script is executed, train and val datasets from the COPICK dataset are registered and DETECTRON2 Training and Evaluation configs are set to train and evaluate the model for colonies panoptic segmentation.

# It uses previously created files in -process_dataset_panoptic_segmentation.py- and -process_dataset_instances_segmentation.py- scripts. 


# INPUTS: NONE

# OUTPUTS: NONE

# ----------------------------------------------------------------------------------------------------------------------
#
#The MIT License (MIT)
#
#Copyright (c) 2023 David R. Espeso, Irene del Olmo
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
#
#  $Version: 1.0 $  $Date: 2023/02/15 $
#
# ----------------------------------------------------------------------------------------------------------------------


# Import required libraries
if __name__ == '__main__':
    
    # import CNN and pytorch
    import torch, torchvision
    import detectron2 
    
    # import some common libraries
    import wget 
    import numpy as np
    import os, json, cv2, random
    import matplotlib.pyplot as plt
    import copy
    
    # import some common detectron2 utilities
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer, _PanopticPrediction
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.data.datasets import register_coco_instances, load_coco_json
    from detectron2.utils.visualizer import ColorMode
    from detectron2.utils.logger import setup_logger
    setup_logger()
    from detectron2.engine import DefaultTrainer, SimpleTrainer
    from detectron2.data.datasets.coco_panoptic import register_coco_panoptic_separated 
    from detectron2.evaluation import COCOEvaluator, COCOPanopticEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader


## Settings for model reproducibility
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)
torch.use_deterministic_algorithms(True)


## Set dataset things and stuff categories and ids 
tings=['1 colonies', '2 colonies', '3 colonies']
stuffs=['things','out of plate', '0 colonies'] # added "things" category because colonies are present in the semantic masks but have a value = 0
stuffs_ids={102:0, 103:0, 104:0, 100:1, 101:2} # "things" are the categories ids=0 here (colony categories in the dataset)


## Register custom colonies dataset
# train dataset
name="my_dataset_train"
metadata={}
image_root="<path to original train dataset images folder>"
panoptic_root="<path to panoptic train masks folder>"
panoptic_json="<path to panoptic train json file>"
sem_seg_root="<path to semantic train masks folder>"
instances_json="<path to instances train json file>"
# We set register_coco_panoptic_separated() because we use a custom dataset
    #To perform the registration, we need previously created files in -process_dataset_panoptic_segmentation.py- and -process_dataset_instances_segmentation.py- scripts. 
register_coco_panoptic_separated(name, metadata, image_root, panoptic_root, panoptic_json, sem_seg_root, instances_json)
# define dataset info 
dataset_dicts_train = DatasetCatalog.get("my_dataset_train_separated")
MetadataCatalog.get("my_dataset_train_separated").set(thing_classes=tings, stuff_classes=stuffs) #, stuff_dataset_id_to_contiguous_id=stuffs_ids)
MetadataCatalog.get("my_dataset_val_separated").set(stuff_dataset_id_to_contiguous_id=stuffs_ids)
my_dataset_metadata_train=MetadataCatalog.get("my_dataset_train_separated")  


# val dataset
name_val="my_dataset_val"
metadata_val={}
image_root_val="<path to original val dataset images folder>"
panoptic_root_val="<path to panoptic val masks folder>"
panoptic_json_val="<path to panoptic val json file>"
sem_seg_root_val="<path to semantic val masks folder>"
instances_json_val="<path to instances val json file>"
# We set register_coco_panoptic_separated() because we use a custom dataset
    #To perform the registration, we need previously created files in -process_dataset_panoptic_segmentation.py- and -process_dataset_instances_segmentation.py- scripts. 
register_coco_panoptic_separated(name_val, metadata_val, image_root_val, panoptic_root_val, panoptic_json_val, sem_seg_root_val, instances_json_val)
# define dataset info
dataset_dicts_val = DatasetCatalog.get("my_dataset_val_separated")
MetadataCatalog.get("my_dataset_val_separated").set(thing_classes=tings, stuff_classes=stuffs)
MetadataCatalog.get("my_dataset_val_separated").set(stuff_dataset_id_to_contiguous_id=stuffs_ids)
my_dataset_metadata_val=MetadataCatalog.get("my_dataset_val_separated")    



# ## Show N random labeled images before training to check data is correctly loaded
#for d in random.sample(dataset_dicts_train, N):
    # img = cv2.imread(d["file_name"])
    # visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_metadata_train, scale=0.5)
    # out = visualizer.draw_dataset_dict(dataset_dicts_train[0])
    # imS = cv2.resize(out.get_image()[:, :, ::-1], (800, 800))   
    # #    cv2.imshow("w", out.get_image()[:, :, ::-1])
    # cv2.imshow("w", imS)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows() 
    #
#


## Set number of threads in cpu: number of processes in the machine
#torch.set_num_threads(int)
## get number of threads in cpu
#torch.get_num_threads(int)



## TRAINING
cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu' # change to gpu if possible
cfg.OUTPUT_DIR = "<path to output folder to store the trained model>"
# Load Panoptic FPN model
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train_separated")
#cfg.DATASETS.TEST = () #"my_dataset_val_separated"
cfg.DATALOADER.NUM_WORKERS = 1 
# Load Panoptic FPN model weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")  
cfg.SOLVER.IMS_PER_BATCH = 2 
cfg.SOLVER.BASE_LR = 0.00025 
cfg.SOLVER.MAX_ITER = 160 
cfg.SOLVER.STEPS = [] 
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 3 #n semantic+1  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # n thing classes in the dataset 

# Train
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train() 
#torch.save(trainer.model, 'checkpoint.pth')  # save trained model   # it is saved automatically


# Access trained model weights
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  

# Set model predictor and test parameters for inference and evaluation
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01  
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.01
cfg.TEST.DETECTIONS_PER_IMAGE = 500 #set maximum number of colonies/objects to be detected in a single image

cfg.MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH = 0.1 #0.5
cfg.MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT = 4096
cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.01 

cfg.DATASETS.TEST=("my_dataset_val_separated", ) 
predictor = DefaultPredictor(cfg)


## INFERENCE AND EVALUATION
## Visualize N random predicted segmentations
# for d in random.sample(dataset_dicts_val, N):    
#     imv = cv2.imread(d["file_name"])
#     v = Visualizer(imv[:, :, ::-1], 
#                     metadata= my_dataset_metadata_val,
#                     scale=1.2, 
#                     instance_mode=ColorMode.IMAGE  
#     )
#     panoptic_seg, segments_info = predictor(imv)["panoptic_seg"]  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
#     out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info) 
#     imV = cv2.resize(out.get_image()[:, :, ::-1], (750, 750))   
#     cv2.imshow("w", imV)
#     cv2.waitKey(0) 
#     cv2.destroyAllWindows() 
#     #
# #

# Use COCOEvaluator or COCOPanopticEvaluator
evaluator = COCOPanopticEvaluator("my_dataset_val_separated", output_dir="<path to store predictions>")
val_loader = build_detection_test_loader(cfg, "my_dataset_val_separated")
evals_results=inference_on_dataset(predictor.model, val_loader, evaluator)



