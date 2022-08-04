# -*- coding: utf-8 -*-
# crop_photo_v3.py - Function to crop photos
#
# GENERAL DESCRIPTION:
# ----------------------------------------------------------------------------------------------------------------------
# Function to crop photos of the Colony picker dataset and store them. Similar to -im_dimensions- function but it is not based on previous panoptic masks.
#
# INPUT: 
# - image_raw: (original image taken with the installed camera in the Opentrons OT-2).
#
# OUTPUT:
# - center_plate: Pixel coordinates of the center of the agar plate
# - image_crop: Cropped image
# 
# ----------------------------------------------------------------------------------------------------------------------
#
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
#
#  ----------------------------------------------------------------------------------------------------------------------

# Import required libraries
# if __name__ == '__main__':
import numpy as np
import copy
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu


#Define Crop photo function (part of -image_pretreatment- function)
def crop_photo(image_raw):
    mask_image_raw=copy.deepcopy(image_raw[:,:,2]) # Field 2 is the brightest and always select the whole agar plate
    otsu_threshold=threshold_otsu(mask_image_raw) #binarize image by Otsu's threshold
    mask_image_raw[mask_image_raw<=otsu_threshold]=0
    mask_image_raw[mask_image_raw>otsu_threshold]=1
    
    # Label binary objects
    mask_image_raw_labeled=label(mask_image_raw,return_num=False, connectivity=2) #1
    info_objects=regionprops(mask_image_raw_labeled) 

    areas_bbox=[]
    area_colonies=[]
    for i in range(0, len(info_objects)): # Compute Bounding box areas
        area_bbox= info_objects[i].bbox_area       
        areas_bbox.append(area_bbox)
        area_colonies.append(info_objects[i].area)
        #
    #
    # Find real center of plate and radius
    info_objects_ids=np.arange(0,len(info_objects), dtype=int)
    elems= np.asarray([info_objects_ids, areas_bbox, area_colonies]).T 
    elems=elems[elems[:,1].argsort()[::-1]]  #Order by descending area size
    labell= elems[0][0]   # The largest detected element will be the agar footprint
    centroid=info_objects[labell].centroid  # Store the centroid
    bbox= info_objects[labell].bbox # Get Bounding box Width and Height
    bbox=list(bbox)
    cx=centroid[1]     # Get centroid coordinates of footpring
    cy=centroid[0]
    r=0.5*np.mean([bbox[2]-bbox[0], bbox[3]-bbox[1]])      
    margin_shift=10 # number of pixels of tolerance
    
    ##### Compute cropping area to reduce image size & # Crop center of image (agar plate)
    bbox_xywh=[abs(round(cx-r-(0.5*margin_shift))),abs(round(cy-r-(0.5*margin_shift))), round(2*r+margin_shift), round(2*r+margin_shift)]
    image_crop=copy.deepcopy(image_raw[bbox_xywh[1]:bbox_xywh[1]+bbox_xywh[3], bbox_xywh[0]:bbox_xywh[0]+bbox_xywh[2],:])
    
    # Create the agar plate mask
    H,W,C=image_crop.shape
    y=np.arange(0,H)     
    x=np.arange(0,W)     
    [X,Y]=np.meshgrid(x,y)
    
    cx_shifted=round(r+(0.5*margin_shift))
    cy_shifted=round(r+(0.5*margin_shift))    
    
    mask_plate=((X-cx_shifted)**2)+((Y-cy_shifted)**2)<=r**2   # Create binary mask of whole agar plate
    
    # Remove dark background color
    mask_plate_rgb=mask_plate
    image_crop[~mask_plate_rgb]=0
    
    center_plate=[cx,cy]
    
    return center_plate, image_crop
    #
#