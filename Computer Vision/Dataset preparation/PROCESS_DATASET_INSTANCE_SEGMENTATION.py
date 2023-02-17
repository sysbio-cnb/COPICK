# -*- coding: utf-8 -*-

# process_dataset_instance_segmentation.py - Python script to create instances annotations of things categories (colonies) for the Panoptic Segmentation task.

# #################
# ¡¡TO CONSIDER!!:
    
# The Colony Picker dataset (COPICK) to be processed in this script was previously built following these steps:
#     - An initial set of 200 images was augmented to 3000 using the functions from file_namer.py and augmentation_stage.py scripts.
#     - The images were also cropped in order to reduce their size to the region of interest (plates with colonies) using the function from crop_photo.py script.
#     - The 3000 images were separated in two subsets: train dataset (2250 images) and val dataset (750 images).
    
# FILES USED IN THIS SCRIPT MAKE REFERENCE TO THE COPICK DATASET AND THEREFORE HAVE TO BE MODIFIED WHEN USING A DIFFERENT DATASET. Files used in this script:
#     json_template.json, panoptic_train.json/panoptic_val.json and panoptic train/val masks.
    
# #################    

# GENERAL DESCRIPTION:
# -------------------
# When executed, the script processes train or val datasets to obtain and save instance annotations (things categories (only colonies) *.json file) from panoptic masks after executing -process_dataset_panoptic_segmentation.py- script.


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
import numpy as np
import os
import json
import cv2 as cv
import copy
from skimage.measure import label, regionprops
#from matplotlib import pyplot as plt
from skimage import measure
#import pycocotools._mask as _mask
#from itertools import groupby
#from skimage.filters import threshold_otsu
#from panopticapi.utils import rgb2id, id2rgb


# Choose train or val dataset
train_session=True #True for train and False for val

# List and open previously created elements in process_dataset_panoptic_segmentation.py script
if train_session==False:
    dataset_dir = os.listdir("<path to panoptic masks val folder>")
    with open ("<path to panoptic val json file>") as json_file:
        panoptic_info = json.load(json_file) #load existing json file as a dictionary
else:
    dataset_dir = os.listdir("<path to panoptic masks train folder>")
    with open ("<path to panoptic val json file>") as json_file:
        panoptic_info = json.load(json_file) #load existing json file as a dictionary


# Open json template file to save dataset instances annotations (things categories: colonies)
with open ("<path to json template file>") as json_file:
    data_info = json.load(json_file) #load existing json file as a dictionary

# Change categories in json template to just pick things (isthing=1) in the dataset 
data_info["categories"]=[
 {'id': 102, 'name': '1 colonies', 'supercategory': 'colonies', 'isthing': 1},
 {'id': 103, 'name': '2 colonies', 'supercategory': 'colonies', 'isthing': 1},
 {'id': 104, 'name': '3 colonies', 'supercategory': 'colonies', 'isthing': 1}]


# Set instances annotations fields (following COCO dataset format: https://cocodataset.org/#format-data) 
annotations = {
    "segmentation":[],
    "id": int,
    "image_id": int, 
    "category_id": int, 
    "area": int,
    "bbox": [],
    "iscrowd": 0}
    
images = {
    "id": int,
    "file_name": "",
    "width": int, 
    "height": int,
    "date_captured": "",
    "license": "r",
    "url": "r"}
  
  
# Set counters to iterate through images in folder and respective annotations    
counter_image=0 # initialite image counter
annot_counter=0 #initialite annotation id

# Process and label images from train or val datasets
for filename in dataset_dir: 
    if train_session==False:
        filename = os.path.join("<path to panoptic masks val folder>", filename)
    else:
        filename = os.path.join("<path to panoptic masks train folder>", filename)
        #
    #
    # load panoptic image
    image_pan=cv.imread(filename) 
    # get panoptic image shape
    w,h,c=np.shape(image_pan)
   
    # create and add images info to instances json template (some of the information is obtained from previously created panoptic jsons)
    temp=copy.deepcopy(images)
    temp["id"]=panoptic_info["images"][counter_image]["id"]
    temp["file_name"]=panoptic_info["images"][counter_image]["file_name"]
    temp["width"]= w
    temp["height"]= h
    temp["date_captured"]=panoptic_info["images"][counter_image]["date_captured"]
    data_info["images"].append(temp)    

    # transform RGB panoptic image to ids image
    image_ids=rgb2id(image_pan) 
    
    # Get background, border (ring) and agar regions color ids from panoptic masks
    red_bgr=rgb2id([255, 0, 1]) #red (background)
    blue_ring=rgb2id([0, 0, 255]) #blue (ring)
    yellow_agar=rgb2id([255, 233, 0]) #yellow (agar)

    mask_seg=copy.deepcopy(image_ids)

    # Create background, border (ring) and agar regions masks based on their colors in panoptic masks
    mask_bgr=copy.deepcopy(mask_seg)
    mask_bgr[mask_bgr!=red_bgr]=0
    mask_bgr[mask_bgr==red_bgr]=1

    mask_ring=copy.deepcopy(mask_seg)
    mask_ring[mask_ring!=blue_ring]=0
    mask_ring[mask_ring==blue_ring]=1

    mask_agar=copy.deepcopy(mask_seg)
    mask_agar[mask_agar!=yellow_agar]=0
    mask_agar[mask_agar==yellow_agar]=1

    # Binarize: assign 0 to non-colonies regions and 1 to colonies in the mask
    mask_colonies=copy.deepcopy(mask_seg)
    mask_colonies[mask_bgr==1]=0
    mask_colonies[mask_ring==1]=0
    mask_colonies[mask_agar==1]=0
    mask_colonies[mask_colonies!=0]=1
    
    # Assign 0 to color ids of non-colonies regions in panoptic mask (background, border and agar)
    image_ids[image_ids==65791]=0 
    image_ids[image_ids==16711680]=0
    image_ids[image_ids==59903]=0
    
    # label modified mask
    image_ids_lab=label(image_ids, connectivity=2) #label image
    image_ids_obj=regionprops(image_ids_lab)
    
    # Get panoptic segments information from panoptic json file
    panoptic_things=panoptic_info["annotations"][counter_image]["segments_info"]
    panoptic_things_cat=[]
    
    # Get categories ids from things only in the dataset
    for i in range(3, len(panoptic_things)): # starts from 3 to exclude the 3 first elements of the panoptic mask (stuff elements: out of plate, ring and agar)
        panoptic_things_cat.append(panoptic_info["annotations"][counter_image]["segments_info"][i]["category_id"])
        #
    #
    
    # Create and add things (colonies) annotations
    # Calculate colonies contours to annotate polygons
    contours=[]
    contours_colonies = measure.find_contours(mask_colonies, 0.5)
    
    ## Draw contours
    # fig, ax = plt.subplots()
    # ax.imshow(mask_colonies, interpolation='nearest', cmap=plt.cm.gray)
    # for n, contour in enumerate(contours):
    #     ax.plot(contours_colonies[n][:, 1], contours_colonies[n][:, 0], linewidth=2)
          #
    #      
    # plt.show()

    # Flip contours
    for i in range(0, len(contours_colonies)):
        contours_colonies[i]=np.flip(contours_colonies[i], axis=1)
        #
    #
    # Add colonies contours
    for i in range(0, len(contours_colonies)):
        contours.append(contours_colonies[i]) #individual colonies contours
        #
    #
    areas=[]
    bbox=[]
    bbox2=[]
    segments=[]
    for i in image_ids_obj:
        areas.append(i.area)
        bbox.append(i.bbox)
        # recalculate bounding boxes due to coordinates order
        square=list(i.bbox)
        x=copy.deepcopy(square[1])
        y=copy.deepcopy(square[0])
        square[2]=abs(square[2]-square[0])
        square[3]=abs(square[3]-square[1])
        square[0]=x
        square[1]=y
        bbox2.append(tuple(square))
        #
    #
    # Generate colonies annotations 
    for i in range(0, len(areas)):
        template=copy.deepcopy(annotations)
        template["area"]=int(areas[i])
        template["bbox"]=list(bbox2[i])
        template["category_id"]=int(panoptic_things_cat[i]) 
        template["id"]=annot_counter 
        template["image_id"]=copy.deepcopy(temp["id"])
        segments=contours[i].ravel().tolist()
        template["segmentation"].append(segments)
        data_info["annotations"].append(template)
        annot_counter+=1
        #
    #
    counter_image += 1
#

# print(data_info)    
json.dumps(data_info) #save annotations data to json file

if train_session==False:
    with open("<path to store instances_val.json file>", 'w') as outfile:
        json.dump(data_info, outfile) #save modified json file      
else:
    with open("<path to store instances_train.json file>", 'w') as outfile:
        json.dump(data_info, outfile) #save modified json file        
        #
    #
#



