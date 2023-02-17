# -*- coding: utf-8 -*-

# augmentation_stage.py - Python script used to augmentate a dataset of images.

# GENERAL DESCRIPTION:
# -------------------
# When the function is called, it generates and saves copies of images from an initial dataset using rotation and flip techniques.
# Function called in the process_dataset script.

# INPUTS: -original_image- (loaded image from folder), -original_name- (name of -original_image- from folder), -augmented_dataset_folder- (path to save the generated augmented dataset).

# OUTPUTS: the output of the function consists on saving each new modified image with their new names in the specified folder.

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

#!/usr/bin/env python

# Import required libraries
import numpy as np
from skimage.transform import rotate
import cv2 as cv
from file_namer import file_namer
import copy


def augmentation_stage(original_image,original_name, augmented_dataset_folder):
    rotation_angle=30 # spinning angle to augmentate the initial agar image into an expanded set of 360º/rotation_angle images
    angles=np.arange(0, 360, 30) # intervals of angles
    
    n_names=int(360/rotation_angle+3) # total number of random names to generate = rotated images (360º/rotation_angle) + 3 flipped images (X,Y,XY) 
    length_name=25 # length of name                                            
    init_names = file_namer(n_names, length_name) # generate random names for augmented images
    
    original_name= original_name.replace(".jpg", "") # remove substring extension from name string
    names=[]
    for j in range(0, len(init_names)):
        names.append(original_name+init_names[j]+".jpg") # assign a final name with original label (to know their ancestry)
        names=copy.deepcopy(names)
        
    # Generate rotated images
    for i in range(0, len(angles)): 
        rotated=rotate(original_image,angles[i], preserve_range=True) 
        cv.imwrite((augmented_dataset_folder+names[i]), rotated)  # write file
        #
    #    
    # Generate flipped images
    flipX= cv.flip(original_image,0) # flip image in X axis
    cv.imwrite((augmented_dataset_folder+names[len(angles)]), flipX) 
    flipY= cv.flip(original_image,1) # flip image in Y axis    
    cv.imwrite((augmented_dataset_folder+names[len(angles)+1]), flipY) 
    flipXY= cv.flip(original_image, -1) # flip image in XY axis 
    cv.imwrite((augmented_dataset_folder+names[len(angles)+2]), flipXY) 
    #
#


