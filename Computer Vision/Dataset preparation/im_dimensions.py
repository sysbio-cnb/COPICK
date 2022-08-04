# -*- coding: utf-8 -*-

# im_dimensions.py - Python script to crop and save images from the Colony picker database (COPICK), based on previous panoptic masks.

# GENERAL DESCRIPTION:
# -------------------
# When executed, the script checks images dimensions to be the same as previously created panoptic masks and crops and save the images from train or val datasets.

# Both this script and the function from -crop_photo.py- script come from the -image_pretreatment.py- script. The latter is used to crop and store images initially, before saving panoptic masks.
# As the provided dataset is already augmentated and cropped, these functions are just additional utils to create custom datasets when interested.


# INPUTS: NONE

# OUTPUTS: NONE (Cropped images are saved in specified path).


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
import copy
import os
import cv2 as cv
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
import numpy as np
from matplotlib import pyplot as plt

 
# Define and create paths of interest
# Choose train or val datasets
train_session=True #True for train dataset and False for val dataset

if train_session==True:
    dataset_dir="<path to folder with original train images>" #UNCROPPED
    dataset_dir_listed=os.listdir(dataset_dir)
    
    panoptic_masks_dir="<path to folder with panoptic train masks>"
    panoptic_masks_dir_listed=os.listdir(panoptic_masks_dir)
else:
    dataset_dir="<path to folder with original val images>" #UNCROPPED
    dataset_dir_listed=os.listdir(dataset_dir)
    
    panoptic_masks_dir="<path to folder with panoptic val masks>"
    panoptic_masks_dir_listed=os.listdir(panoptic_masks_dir)
    #
#

# Create new folders to store cropped images
train_crop_folder= os.makedirs("<path to store cropped train images>", exist_ok=True)
val_crop_folder=os.makedirs("<path to store cropped val images>", exist_ok=True)

# Access image properties
width_list=[]
height_list=[]
width_crop_list=[]
height_crop_list=[]
bbox_crop_image=[]

# Panoptic masks where created from cropped images but it was before storing cropped images, so we want to be sure cropped images to be stored have the same dimensions as the ones previously used and the panoptic masks
for i in panoptic_masks_dir_listed:
    file = os.path.join(panoptic_masks_dir, i)
    image_raw_pan=cv.imread(file)
    height, width, channels= image_raw_pan.shape
    
    # get panoptic masks dimensions to be compared with the following cropped images dimensions
    width_list.append(width) 
    height_list.append(height)
   
# Crop original images    
for j in dataset_dir_listed:
    filename = os.path.join(dataset_dir, j)
    image_raw=cv.imread(filename)
    # plt.imshow(image_raw)
    # plt.show()
    image_raw_name=filename[-35:] #actual length of image name

    mask_image_raw=copy.deepcopy(image_raw[:,:,2]) # Field 2 is the brightest and always select the whole agar plate
    otsu_threshold=threshold_otsu(mask_image_raw) #binarize image by Otsu's threshold
    mask_image_raw[mask_image_raw<=otsu_threshold]=0
    mask_image_raw[mask_image_raw>otsu_threshold]=1
    
    # plt.imshow(mask_image_raw, cmap="gray")
    # plt.show()    
    
    # Label binary objects
    mask_image_raw_labeled=label(mask_image_raw,return_num=False, connectivity=2) #connectivity=1
    info_objects=regionprops(mask_image_raw_labeled) 
    
    areas_bbox=[]
    area_colonies=[]
    for i in range(0, len(info_objects)): # Compute Bounding box areas
        area_bbox= info_objects[i].bbox_area       
        areas_bbox.append(area_bbox)
        area_colonies.append(info_objects[i].area)
        #
    #
    # Find real center of plate and radius based on plate footprint
    info_objects_ids=np.arange(0,len(info_objects), dtype=int)
    elems= np.asarray([info_objects_ids, areas_bbox, area_colonies]).T # Create array to filter and select the biggest objects by area
    elems=elems[elems[:,1].argsort()[::-1]]  #Order by descending area size
    labell= elems[0][0]   # The largest detected element will be the plate footprint
    centroid=info_objects[labell].centroid  # Store the centroid
    bbox= info_objects[labell].bbox # Get Bounding box Width and Height
    bbox=list(bbox)
    # get centroid coordinates of footpring (plate)
    cx=centroid[1]     
    cy=centroid[0]
    # get radius
    r=0.5*np.mean([bbox[2]-bbox[0], bbox[3]-bbox[1]])      
    margin_shift=10 # set number of pixels of tolerance
    
    
    ##### Compute cropping area to reduce image size & # Crop center of image (agar plate)
    # Recalculate bbox 
    bbox_xywh=[abs(round(cx-r-(0.5*margin_shift))),abs(round(cy-r-(0.5*margin_shift))), round(2*r+margin_shift), round(2*r+margin_shift)]
    bbox_crop_image.append(bbox_xywh)
    
    image_crop=copy.deepcopy(image_raw[bbox_xywh[1]:bbox_xywh[1]+bbox_xywh[3], bbox_xywh[0]:bbox_xywh[0]+bbox_xywh[2],:])
    # plt.imshow(image_crop)
    # plt.show()
    H,W,C=image_crop.shape
    
    height_crop_list.append(H)
    width_crop_list.append(W)
    
    #Compare image dimensions from lists and save cropped images:
    if height_crop_list==height_list and  width_crop_list==width_list:
        if train_session==True:
            cv.imwrite((train_crop_folder+image_raw_name), image_crop)
        else:
            cv.imwrite((val_crop_folder+image_raw_name), image_crop)
        #
    #
#
