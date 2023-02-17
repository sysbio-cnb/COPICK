# -*- coding: utf-8 -*-
# crop_photo-v8.py - Python script in charge of cropping an image with a Petri dish on it. 
#
# GENERAL DESCRIPTION:
#---------------------
# This function crops a raw image with a Petri dish adjusting the crop to a specific plate radius.
# When the GFP filter is activated, this function is called for the second time using the GFP raw image and the previously calculated values of the following input variables to crop the original raw image, so that the crops of both images match.
# 
# INPUT: 
#        -image_raw- (original raw image to crop). 
#        -bbox_xywh- (bounding box dimensions of the cropped region. If False, the function calculates it for the original raw image. If True, the function uses the variable to produce a matching crop in the raw GFP photo).
#        -r- (radius of the cropped region. By default it is equal to 780 inside the function. However, it can be modified and later also used by the function to produce a matching crop in the raw GFP photo).
#        -margin_shift- (number of pixels of tolerance in the margin of the crop. By default it is equal to 0 inside the function. However, it can be modified and later also used by the function to produce a matching crop in the raw GFP photo). 
#        -cx- (x coordinate of the center of the plate in the raw image. By default it is equal to 0. Once calculated, it can be later also used by the function to produce a matching crop in the raw GFP photo).
#        -cy- (y coordinate of the center of the plate in the raw image. By default it is equal to 0. Once calculated, it can be later also used by the function to produce a matching crop in the raw GFP photo).
#
# OUTPUT: 
#        -center_plate_labware- (center coordinates of the plate in the raw image).
#        -center_plate_crop- (center coordinates of the plate in the cropped image).
#        -image_crop- (cropped image).
#        -bbox_xywh- (bounding box of the cropped region).
#        -r- (radius of the cropped region).
#        -margin_shift- (number of pixels of tolerance in the margin of the crop).
#        -cx- (x coordinate of the center of the plate in the raw image).
#        -cy- (y coordinate of the center of the plate in the raw image).
#
#
#--------------------------------------------------------------------------------
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
#------------------------------------------------------------------------------

import numpy as np
#import matplotlib.pyplot as plt
import copy

from skimage import color
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny

def crop_photo(image_raw, bbox_xywh, r, margin_shift, cx, cy):
    
    if bool(bbox_xywh)==False: # If True, use the previously calculated input variables values to produce the crop instead of executing the loop. 
        # Load picture and detect edges
        gray_img  = color.rgb2gray(image_raw)
        
        # obtain edges
        edges = canny(gray_img, sigma=1)
        #plt.imshow(edges)
        
        # Detect circunferences
        hough_radii = [780] # <== Adjust according to your Petri dish diameter. Default value is adjusted for outer diameter D=86 mm.
        hough_res = hough_circle(edges, hough_radii)
        
        centers = []
        accums = []
        radii = []
        
        for radius, h in zip(hough_radii, hough_res):
            # For each radius, extract 5 circles
            num_peaks = 5
            peaks = peak_local_max(h, num_peaks=num_peaks)
            centers.extend(peaks)
            accums.extend(h[peaks[:, 0], peaks[:, 1]])
            radii.extend([radius] * num_peaks)
        
        x_coord=[]
        y_coord=[]
        for i in range(0,5):
            x_coord.append(centers[i][0])
            y_coord.append(centers[i][1])
            #
        #
        
        cx=np.ceil(np.mean(x_coord))
        cy=np.ceil(np.mean(y_coord))
        r=hough_radii[0]
        margin_shift=0 # number of pixels of tolerance
        
        ##### Compute cropping area to reduce image size & # Crop center of image (agar plate)
        bbox_xywh=[abs(round(cx-r-(0.5*margin_shift))),abs(round(cy-r-(0.5*margin_shift))), round(2*r+margin_shift), round(2*r+margin_shift)]
        #
    #
    image_crop=copy.deepcopy(image_raw[bbox_xywh[0]:bbox_xywh[0]+bbox_xywh[2],bbox_xywh[1]:bbox_xywh[1]+bbox_xywh[3],:])
    
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
    
    center_plate_crop=[int(0.5*W),int(0.5*H)]
    center_plate_labware=[cx,cy]
    
    # plt.imshow(image_crop)
    # plt.show()

    return center_plate_labware, center_plate_crop, image_crop, bbox_xywh, r, margin_shift, cx, cy
    
