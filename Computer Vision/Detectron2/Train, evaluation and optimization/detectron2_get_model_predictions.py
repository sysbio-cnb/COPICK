# -*- coding: utf-8 -*-

# detectron2_get_model_predictions.py - Python script to get predictions from trained panoptic segmentation model on Colony Picking (COPICK) dataset.

# GENERAL DESCRIPTION:
# --------------------
# When the script is executed, previously trained model and model weights are used to visualize and save colonies plates predictions/inferences.

# In order to use Detectron2 Visualizer function for inference, we have to register a dataset. 
    
# Additional non-registered photos can be used for inference without the Visualizer.
#     In this case, we obtain grayscale panoptic masks. 
    
    
# INPUTS: NONE

# OUTPUTS: NONE (saved predictions images).


#The MIT License (MIT)
#
#Copyright (c) 2022 David R. Espeso, Irene del Olmo
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
#  $Version: 1.0 $  $Date: 2022/07/26 $



# Import required libraries
if __name__ == '__main__':
    
    # import CNN and pytorch
    import torch, torchvision
    import detectron2
    
    # import some common libraries
    import wget 
    import numpy as np
    import os, json, random
    import cv2 as cv
    import matplotlib.pyplot as plt
    import copy, csv
    import math
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    
    # import some common detectron2 utilities
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.utils.visualizer import Visualizer, _PanopticPrediction
    from detectron2.utils.visualizer import ColorMode
    from detectron2.data.datasets.coco_panoptic import register_coco_panoptic_separated 
    
    from detectron2.utils.logger import setup_logger
    setup_logger()
    #
#


# Register short dataset for inference
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)
torch.use_deterministic_algorithms(True)


# ## Set dataset things and stuff categories and ids 
tings=['1 colonies', '2 colonies', '3 colonies']
stuffs=['things', 'out of plate', '0 colonies']
stuffs_ids={102:0, 103:0, 104:0, 100:1, 101:2}


# Register a dataset for model inference (to choose from Datasets/ folder)
name_val="my_dataset_val"
metadata_val={}
image_root_val="<path to dataset original images>"
panoptic_root_val="<path to panoptic dataset masks>"
panoptic_json_val="<path to panoptic json file>"
sem_seg_root_val="<path to semantic dataset masks>"
instances_json_val="path to instances json file>"
# We set register_coco_panoptic_separated() function because we use a custom dataset
register_coco_panoptic_separated(name_val, metadata_val, image_root_val, panoptic_root_val, panoptic_json_val, sem_seg_root_val, instances_json_val)
# define dataset info
dataset_dicts_val = DatasetCatalog.get("my_dataset_val_separated")
MetadataCatalog.get("my_dataset_val_separated").set(thing_classes=tings, stuff_classes=stuffs, stuff_dataset_id_to_contiguous_id=stuffs_ids)
my_dataset_metadata_val=MetadataCatalog.get("my_dataset_val_separated")    


# Set Detectron2 config, model and parameters for inference 
cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu' 
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 3 #n semantic+1  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
## Set model weights of interest
#cfg.MODEL.WEIGHTS="C:/Users/Cazorla/Documents/CNB/ColonyPicker/CNN codes/detectron2_w/projects/Initial trial projects/output/trial/model_final.pth"
#cfg.MODEL.WEIGHTS="C:/Users/Cazorla/Documents/CNB/ColonyPicker/CNN codes/detectron2_w/projects/Initial trial projects/output/optimization results 4/13/model_final.pth"
cfg.MODEL.WEIGHTS="C:/Users/Cazorla/Documents/CNB/ColonyPicker/CNN codes/detectron2_w/Optimization/Studies 2/64/model_final.pth"  #30, #37 #1, 10, 15, 23, 25 #"C:/Users/Cazorla/Documents/CNB/ColonyPicker/CNN codes/detectron2_w/projects/Initial trial projects/output/training_trial/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01  # set a low threshold to get the largest amount of predictions (colonies)
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.01
cfg.TEST.DETECTIONS_PER_IMAGE = 500
cfg.MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH = 0.5
cfg.MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT = 2048 #4096
cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.1
predictor = DefaultPredictor(cfg)


counter_im=0   
# Try model inference in a validation dataset 
for i in range(0, len(dataset_dicts_val)):
    #Get panoptic colored prediction
    imv = cv.imread(dataset_dicts_val[i]["file_name"]) 
    v = Visualizer(imv[:, :, ::-1], 
                    metadata= my_dataset_metadata_val, #train
                    scale=1.2, #0.5
                    instance_mode=ColorMode.IMAGE  #SEGMENTATION #IMAGE_BW     # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    panoptic_seg, segments_info = predictor(imv)["panoptic_seg"]  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info) #gpu
    imV = cv.resize(out.get_image()[:, :, ::-1], (750, 750))   
    ## Show panoptic prediction
    cv.imshow("w", imV)
    cv.waitKey(0) 
    cv.destroyAllWindows() 
    
    ## Save panoptic prediction mask
    #cv.imwrite("<path to store predicted masks>" +" image_inf_" + str(counter_im) + ".png", imV)
    counter_im+=1
    #
#


# Get model predictions of non-registered photos and save predicted masks
# List folder with selected images
image_folder=("<path to images folder>") 
dataset_dir=os.listdir(image_folder)

# Test multiple images
counter=0    
for filename in dataset_dir:
    filename= os.path.join(image_folder, filename)
    image_color=cv.imread(filename)
    preds_val=predictor(image_color)["panoptic_seg"]
    ## Show panoptic prediction
    plt.imshow(preds_val[0], cmap="gray")
    plt.show()
    im_panoptic=np.asarray(preds_val[0])  
    
    #Save panoptic predicted mask as *.png
    cv.imwrite("<path to store predicted masks>/image_pred_" + str(counter) + ".png", im_panoptic)
    counter+=1
    #
#

# Test single image
im=cv.imread("<path to image>")
plt.imshow(im)

# Get prediction
im_pred=predictor(im)["panoptic_seg"]
plt.imshow(im_pred[0], cmap="gray")

# Save panoptic prediction
im_panoptic_pred=np.asarray(im_pred[0])

cv.imwrite("<path to store image prediction>" + "image name", im_panoptic_pred)
#
