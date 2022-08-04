# -*- coding: utf-8 -*-

# process_dataset.py - Python script to process the COPICK dataset and get the panoptic annotations (*.json file and panoptic images) and semantic annotations masks of each image.
#                      See: https://detectron2.readthedocs.io/en/latest/tutorials/builtin_datasets.html (Expected dataset structure for PanopticFPN).

# #################
# ¡¡TO CONSIDER!!:
    
# The Colony Picker dataset (COPICK) to be processed in this script was previously built following these steps:
#     - An initial set of 200 images was augmented to 3000 using the functions from file_namer.py and augmentation_stage.py scripts.
#     - The images were also cropped in order to reduce their size to the region of interest (plates with colonies) during image pretreatment stage.
#     - The 3000 images were separated in two subsets: train dataset (2250 images) and val dataset (750 images).
    
# FILES USED IN THIS SCRIPT MAKE REFERENCE TO THE COPICK DATASET AND THEREFORE HAVE TO BE MODIFIED WHEN USING A DIFFERENT DATASET. Files used in this script:
#     json_template.json and Colony_Picker_database.csv provided files.

# Functions used in this script:
#     image_pretreatment.py, segmentation_stage.py and separate_semantic_from_panoptic.py. 
    
# ################# 
 

# GENERAL DESCRIPTION:
# -------------------
# When the script is executed, it processes train or val datasets to obtain and save the panoptic annotations (*.json file and panoptic images) and semantic masks of each image to train the Detectron2 AI model for panoptic segmentation.

# INPUTS: NONE

# OUTPUTS: NONE


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


# Import libraries
import numpy as np
from skimage import filters, morphology
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature
from skimage.measure import label, regionprops, find_contours
import copy
from skimage.color import label2rgb, rgb2lab
from skimage.morphology import convex_hull_image, convex_hull_object
import pandas as pd
from skimage.util import crop
from skimage.filters import threshold_otsu, threshold_local
import cv2 as cv
from skimage.filters import unsharp_mask
from statistics import mean
import os
import json
from panopticapi.utils import rgb2id, id2rgb

# Import previously generated functions
from image_pretreatment import image_pretreatment
from segmentation_stage import segmentation_stage
from separate_semantic_from_panoptic import separate_semantic_from_panoptic
#from process_panoptic_to_semantic import _process_panoptic_to_semantic


## Define and list paths to images folder

# Define original dataset folder
dataset_folder= "<path to images folder>/" # General path where the train and val datasets folders are stored

# Define train and val datasets folders
train_folder="<path to train dataset folder>/"
val_folder="<path to val dataset folder>/"

# Set train_session to compute train or val dataset
train_session=False #True for train and False for val

# Set counters to iterate through images in folder and respective annotations
counter_image=0
annots_counter=0

# Access train or val dataset folders
if train_session==True:
    dataset_dir=os.listdir(train_folder)
else:
    dataset_dir=os.listdir(val_folder)
    #
#

# Load existing json file template to store panoptic annotations 
with open ("<path to json template file>") as json_file:
    data_info = json.load(json_file) 
    #
#
## JSON FIELDS 
# info  # already described in the json template
# licenses  # already described in the json template
# categories # already described in the json template:
    #2 background categories (out of plate and plate), 3 colony categories (1 colony, 2 colonies and 3 or more colonies groups)

images = {
    "id": 0,  #
    "file_name": "",  #The one given by file_namer function
    "width": int, 
    "height": int,  
    "date_captured": "",  #Original image date (Stored in the Colony_Picker_database.csv file)
    "license": "r",
    "url": "r"}


annotations= {   #Panoptic annotations: per image, not object
  "image_id": int,  #Is the same as "id" in "images" json field 
  "file_name": str,  #The one given when creating the segmented panoptic images
  "segments_info": []}  #Per segment in panoptic image


segments_info = {
  "id": int,  #Segment label as defined by segmentation_stage function
  "category_id": int,
  "area": int,  #Segment area as defined by segmentation_stage function
  "bbox": [],  #Segment bbox as defined by segmentation_stage function
  "iscrowd": 0 #1 would be for a large GROUP of objects
  }


# Load CSV file with the dataset info (Colony_Picker_database.csv)
csvdata = pd.read_csv ("<path to csv file>", delimiter= ";")

im_name_csv=list(csvdata.filename)
im_id_csv=list(csvdata.id)
im_date_csv=list(csvdata.date)
im_colonies_csv=list(csvdata.colonies)

# Filter images by position in dataset
positions=[]
for i in dataset_dir:
    positions.append(im_name_csv.index(i))
    #
#
im_name=[]
im_id=[]
im_date=[]
im_colonies=[]
for i in positions:
    im_name.append(im_name_csv[i])
    im_id.append(im_id_csv[i])
    im_date.append(im_date_csv[i])  
    im_colonies.append(im_colonies_csv[i])
    #
#


# Set fixed color combinations list to be assigned to every segment in panoptic masks during segmentation
combinations=np.zeros((1000,3)) # Get a maximum of 1000 different color values to assign to the different segments in the images
for i in range(0,1000):
    np.random.seed(i)
    combinations[i,:]=np.round(np.random.rand(1,3)*255) # create random combinations of RGB color values
    #
#
# Assign red, blue and yellow color values to the 3 first segments in panoptic masks (corresponding to regions on the plate excluding colonies)
# Later on labeling starts from 0 so it is skipped to start from 1
combinations=combinations.astype(int)
combinations[1]=[255, 0, 1] #red
combinations[2]=[0, 0, 255] #blue
combinations[3]=[255, 233, 0] #yellow
elems=np.unique(combinations, axis=0) # check if every set of values from combinations is unique
print(len(elems))

# Create ids list for each color
id_segment_list=[]

for chosen in combinations:
    id_segment_list.append(rgb2id(chosen))
    #
#


# Process train or val dataset images to create corresponding panoptic masks and annotations
for filename in dataset_dir:
    if train_session==True:
        filename = os.path.join(train_folder, filename)
        panoptic_dir=os.makedirs(dataset_folder + "panoptic_train/", exist_ok=True) #path to store panoptic train masks
    else:
        filename = os.path.join(val_folder, filename)
        panoptic_dir=os.makedirs(dataset_folder + "panoptic_val/", exist_ok=True) #path to store panoptic val masks
        
    ## Apply transformations to each image
    # Read image 
    image_raw=cv.imread(filename)
    original_name=filename[-35:] #length of the actual name

    # Apply pretreatment
    image_raw_name, image_crop, mask_image_crop,mask_plate, mask_agar_inner, mask_agar_ring, mask_background,mask_colonies_inner_raw, mask_agar_border, centroid, areas_median= image_pretreatment(image_raw, filename)
    
    # Apply segmentation
    mask_colonies, n_colonies, mask_panoptic, area_colonies, label_colonies, bbox_colonies, mask_colonies_labeled, colony_objects, final_elems, mask_panoptic_labeled, panoptic_objects, mask_panoptic_rgb = segmentation_stage(image_raw_name, image_crop, mask_image_crop,mask_plate, mask_agar_inner, mask_agar_ring, mask_background,mask_colonies_inner_raw, mask_agar_border, centroid, areas_median, combinations, id_segment_list)
    
    # Save segmented masks in created folder
    panoptic_name=original_name.replace(".jpg", "_panoptic_mask") # remove substring extension from name and add new extension
    name=panoptic_dir+panoptic_name +'.png' # assign a final name with original label (to know their ancestry)
    cv.imwrite(name, mask_panoptic_rgb) # write file
    #
    # Get image shape
    mask_shape_w, mask_shape_h=mask_colonies_inner_raw.shape # all masks have the same shape
    
    # Add "images" data to json
    temp_im=copy.deepcopy(images)
    temp_im["id"]=im_id[counter_image]
    temp_im["file_name"]=im_name[counter_image]
    temp_im["date_captured"]=im_date[counter_image]
    temp_im["width"]=mask_shape_w
    temp_im["height"]=mask_shape_h
    data_info["images"].append(temp_im)    

    # Establish colony categories based on number of colonies registered in final_elems
    colony_category_bgr=[-1,0,0] #categories for empty plates
    final_elems[:,2][0]=-1 #for plates with colonies, we replace the category values for the 3 first detected elements corresponding to the region outside the plate, the border of the plate, and the inner region with the agar
    final_elems[:,2][1:3]=0 #for plates with colonies, we replace the category values for the 3 first detected elements corresponding to the region outside the plate, the border of the plate, and the inner region with the agar
    category_id_list=copy.deepcopy(final_elems[:,2])
    
    colony_category=[]
    for i in range(0, len(area_colonies)):
        if len(area_colonies)==3: #empty plate: 3 first detected elements
            colony_category=colony_category_bgr
        else:
            colony_category.append(int(category_id_list[i])) #if the plate has colonies, add the elements on previously created list
            #
        #
    #
    # Add "annotations" data to json
    temp_ann=copy.deepcopy(annotations)
    
    for i in range(0, len(area_colonies)): #annotations of each panoptic segment 
        temp_seg=copy.deepcopy(segments_info)
        temp_seg["id"]= id_segment_list[i+1]  #label of pixels belonging to the same object: i+1 to skip the first id (0)
        temp_seg["area"]=int(area_colonies[i])
        temp_seg["bbox"]=list(bbox_colonies[i])
        temp_seg["category_id"]=int(colony_category[i])+101 # +101 due to modifications on the categories (categories finally range from 100 to 104) # Establish colony categories based on number of colonies
        temp_ann["segments_info"].append(temp_seg)
        annots_counter+=1
   
    temp_ann["image_id"]=im_id[counter_image]
    temp_ann["file_name"]=name[-49:] #the length of the actual panoptic name
    data_info["annotations"].append(temp_ann)

    counter_image += 1
    #
#
#print(data_info)    
json.dumps(data_info) #save data to json file 
#print(json.dumps(data_info, indent=4)) 

if train_session==True:
    with open("<path to store panoptic_train_dataset.json>", 'w') as outfile:
        json.dump(data_info, outfile) #save json file with panoptic train annotations  
else:
    with open("<path to store panoptic_val_dataset.json>", 'w') as outfile:
        json.dump(data_info, outfile) #save json file with panoptic val annotations       
        #
    #
#
   
     
# Get semantic annotations for stuff segments (regions not corresponding to colonies in the image) from panoptic masks and panoptic json        
#Define masks and jsons paths
if train_session==True:
    panoptic_json="<path to panoptic train json file>"
    panoptic_root="<path to panoptic masks train folder>"
    sem_seg_root=os.makedirs("<path to semantic segmentation train masks>", exist_ok=True)

else:
    panoptic_json="<path to panoptic train json file>"
    panoptic_root="<path to panoptic masks val folder>"
    sem_seg_root=os.makedirs("<path to semantic segmentation val masks>", exist_ok=True)

#Get categories
categories=data_info["categories"]

#Call Detectron2 function to obtain semantic masks from panoptic masks 
separate_semantic_from_panoptic(panoptic_json, panoptic_root, sem_seg_root, categories)