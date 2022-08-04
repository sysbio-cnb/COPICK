# -*- coding: utf-8 -*-

# display_colonies_overlay.py - Python script to display detected colonies from panoptic segmentation model predictions over original images.

# GENERAL DESCRIPTION:
# --------------------
# When the script is executed, the panoptic prediction is tranformed to select just the colonies predictions and display them overlayed on top of the original image for visualization purposes.

# INPUTS: 
#     -image_raw: original cropped image either from train/val datasets or taken with the Opentrons installed camera. 
#     -panoptic_pred: image_raw panoptic segmentation prediction (got with detectron2_get_model_predictions.py script or during processing in Opentrons robot.  

# OUTPUTS: NONE.


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
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
import matplotlib.colors as colors



# Define colonies overlay function
def display_colonies_overlay(image_raw, panoptic_pred):
    labs=np.unique(panoptic_pred)  # the 2 last ids are always the plate and out of plate regions excluding the colonies

    # Modify panoptic prediction in order to compute the colonies coordinates in the image
    filtered_pred=copy.deepcopy(panoptic_pred)
    filtered_pred[filtered_pred==0]=labs[-3]+3 ## asign high id value to first predicted element so that it is not 0
    filtered_pred[filtered_pred==labs[-2]]=0 # assign 0 to outside of plate region

    # Modify panoptic prediction to assign 0 to colonies 
    panoptic_mod=copy.deepcopy(panoptic_pred)
   
    for elem in labs:
        if (elem != labs[-1] and elem != labs[-2]):
            panoptic_mod[panoptic_mod==elem]=0
            #
        #
    #
    # Get binary mask with the predicted colonies footprint only
    panoptic_binary=panoptic_mod==0

    #Display colonies overlay
    plt.figure()
    image_raw = color.rgb2gray(image_raw)    
    plt.imshow(image_raw, cmap="gray")
    cmapp = colors.ListedColormap(["white", "red"])
    plt.imshow(panoptic_binary, cmap=cmapp, alpha=0.2)
    plt.show()
    #
#
