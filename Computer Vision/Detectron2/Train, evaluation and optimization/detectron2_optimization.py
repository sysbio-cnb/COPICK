# -*- coding: utf-8 -*-

# detectron2_optimization.py - Pyhton script to perform DETECTRON2 config hyperparameters optimization using OPTUNA library in order to improve the accuracy of panoptic segmentation model for colonies prediction. 

# GENERAL DESCRIPTION:
# --------------------
# When the script is executed, after train and val datasets are registered and DETECTRON2 Training and Evaluation configs are set, model hyperparameters optimization is performed following OPTUNA library instructions.

# See: https://optuna.org/

# Hyperparameters to optimize:
#     -ims_per_batch-: Detectron's _C.SOLVER.IMS_PER_BATCH 
#     -lr-: Detectron's _C.SOLVER.BASE_LR
#     -max_iters-: Detectron's _C.SOLVER.MAX_ITER
#     -roi_heads-: Detectron's _C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE

# Detectron2 CONFIG (_C) info in: https://detectron2.readthedocs.io/en/latest/modules/config.html


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
    import gc
    # import some common libraries
    import wget #needs previous install (>>>pip install wget)
    import numpy as np
    import os, json, cv2, random
    import matplotlib.pyplot as plt
    import copy
    import joblib
    import optuna
    
    # import some common detectron2 utilities
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg, set_global_cfg
    from detectron2.utils.visualizer import Visualizer, _PanopticPrediction
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.data.datasets import register_coco_instances, load_coco_json
    from detectron2.utils.visualizer import ColorMode
    from detectron2.utils.logger import setup_logger
    setup_logger()
    from detectron2.engine import DefaultTrainer, SimpleTrainer
    from detectron2.evaluation import COCOEvaluator, COCOPanopticEvaluator, inference_on_dataset, COCOEvaluator, SemSegEvaluator, DatasetEvaluators
    from detectron2.data import build_detection_test_loader

    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.data.datasets.coco_panoptic import register_coco_panoptic_separated 


## Pytorch settings for reproducibility
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)
torch.use_deterministic_algorithms(True)


## Set things and stuff categories
tings=['1 colonies', '2 colonies', '3 colonies']
stuffs=['things', 'out of plate', '0 colonies']
stuffs_ids={102:0, 103:0, 104:0, 100:1, 101:2}


## Register custom colonies dataset
# train dataset
name="my_dataset_train"
metadata={}
image_root="C:/Users/Cazorla/Documents/CNB/ColonyPicker/CNN codes/detectron2_w/datasets/coco_panoptic_v2/images/train/"
panoptic_root="C:/Users/Cazorla/Documents/CNB/ColonyPicker/CNN codes/detectron2_w/datasets/coco_panoptic_v2/annotations/panoptic_trainval/panoptic_train/"
panoptic_json="C:/Users/Cazorla/Documents/CNB/ColonyPicker/CNN codes/detectron2_w/datasets/coco_panoptic_v2/annotations/panoptic_train_v2.json"
sem_seg_root="C:/Users/Cazorla/Documents/CNB/ColonyPicker/CNN codes/detectron2_w/datasets/coco_panoptic_v2/annotations/panoptic_stuff/stuff_train/"
instances_json="C:/Users/Cazorla/Documents/CNB/ColonyPicker/CNN codes/detectron2_w/datasets/coco_panoptic_v2/annotations/instances_train_v2.json"
register_coco_panoptic_separated(name, metadata, image_root, panoptic_root, panoptic_json, sem_seg_root, instances_json)
# define dataset info 
dataset_dicts_train = DatasetCatalog.get("my_dataset_train_separated")
MetadataCatalog.get("my_dataset_train_separated").set(thing_classes=tings, stuff_classes=stuffs, stuff_dataset_id_to_contiguous_id=stuffs_ids)
my_dataset_metadata_train=MetadataCatalog.get("my_dataset_train_separated")  
   
# val dataset
name_val="my_dataset_val"
metadata_val={}
image_root_val="C:/Users/Cazorla/Documents/CNB/ColonyPicker/CNN codes/detectron2_w/datasets/coco_panoptic_v2/images/val/"
panoptic_root_val="C:/Users/Cazorla/Documents/CNB/ColonyPicker/CNN codes/detectron2_w/datasets/coco_panoptic_v2/annotations/panoptic_trainval/panoptic_val/"
panoptic_json_val="C:/Users/Cazorla/Documents/CNB/ColonyPicker/CNN codes/detectron2_w/datasets/coco_panoptic_v2/annotations/panoptic_val_v2.json"
sem_seg_root_val="C:/Users/Cazorla/Documents/CNB/ColonyPicker/CNN codes/detectron2_w/datasets/coco_panoptic_v2/annotations/panoptic_stuff/stuff_val/"
instances_json_val="C:/Users/Cazorla/Documents/CNB/ColonyPicker/CNN codes/detectron2_w/datasets/coco_panoptic_v2/annotations/instances_val_v2.json"
register_coco_panoptic_separated(name_val, metadata_val, image_root_val, panoptic_root_val, panoptic_json_val, sem_seg_root_val, instances_json_val)
# define dataset info
dataset_dicts_val = DatasetCatalog.get("my_dataset_val_separated")
MetadataCatalog.get("my_dataset_val_separated").set(thing_classes=tings, stuff_classes=stuffs, stuff_dataset_id_to_contiguous_id=stuffs_ids)
my_dataset_metadata_val=MetadataCatalog.get("my_dataset_val_separated")    


# Set counter to store optimization results after each iteration in different folders
counter=1 

# Set Detectron2 config and some config parameters
cfg= get_cfg()
set_global_cfg(cfg) 
cfg.SEED=1
cfg.MODEL.DEVICE = 'cpu' 
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")) #set panoptic fpn starting point model 
cfg.DATASETS.TRAIN = ("my_dataset_train_separated")
cfg.DATASETS.TEST = () 
cfg.DATALOADER.NUM_WORKERS = 0 

cfg.SOLVER.STEPS = []       
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 3 #n semantic+1  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  

#set model predictor for inference
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01  # set a custom testing threshold
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.01
cfg.TEST.DETECTIONS_PER_IMAGE = 500

cfg.MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH = 0.5 
cfg.MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT = 4096
cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.01 

evaluator = COCOPanopticEvaluator("my_dataset_val_separated")


# Define black box objective function with parameters to optimize following OPTUNA library instructions
def black_box_function(trial):
    global counter
    global evaluator
    
    output_dir=os.makedirs("<path to store model>/"+str(counter), exist_ok=True) #counter defines the number of optimization rounds to store the results in different folders
    cfg.OUTPUT_DIR = "<path to store model>/"+str(counter) 
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml") #set panoptic fpn starting point model weights
    
    # set model hyperparameters values range
    cfg.SOLVER.IMS_PER_BATCH =  trial.suggest_int("ims_per_batch", 1, 8) 
    cfg.SOLVER.BASE_LR =  trial.suggest_float("lr", 0.0001, 0.01) 
    cfg.SOLVER.MAX_ITER =  trial.suggest_int("max_iters", 80, 640) 
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE =  trial.suggest_int("roi_heads", 128, 512) 
    
    # train
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train() # train
   
    # access trained model weights
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  

    # set predictor
    cfg.DATASETS.TEST=("my_dataset_val_separated", ) 
    predictor = DefaultPredictor(cfg)

    # evaluate trained model predictions
    val_loader = build_detection_test_loader(cfg, "my_dataset_val_separated") 
    eval_results=inference_on_dataset(predictor.model, val_loader, evaluator)
    PQ=float(eval_results['panoptic_seg']['PQ'])
    SQ=float(eval_results['panoptic_seg']['SQ'])
    counter+=1
    
    del predictor, val_loader, eval_results #delete stored variables to free memory
    
    # gc.collect() #collect garbage
    
    return SQ # Maximize SQ (segmentation quality) parameter instead of PQ (overall panoptic quality).
    #
#


## OPTUNA hyperparameters optimization

# Create new study
study = optuna.create_study(study_name="Study_1", direction="maximize")  

# Invoke optimization of the objective function. gc_after_trial is meant to collect remaining garbage after each trial
study.optimize(black_box_function, n_trials=5, gc_after_trial=True)
#study.optimize(black_box_function, n_trials=6, gc_after_trial=True) # More than 6, or even 6 at times, leads to memory error in this case

# Save study after optimization
joblib.dump(study, "path to store study" + "/Study_1.pkl")


# Print trials info from selected study
# Get best trial info
print("Best trial until now:")
print(" Value: ", study.best_trial.value)
print(" Params: ")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")
    #
# 
# Or:
best_results=study.best_params

# Get all trials info
all_results=study.get_trials()
#

# Load saved study in case we want to reasume the optimization and perform previous steps again
study = joblib.load("path to Study_1.pkl") 
#
#
