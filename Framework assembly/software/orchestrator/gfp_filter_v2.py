# -*- coding: utf-8 -*-
# GFP_FILTER.py - Python script in charge of executing a colony picking protocol.
#
# GENERAL DESCRIPTION:
# ----------------------------------------------------------------------------------------------------------------------
# Function to filter and select colonies in plate by gfp intensity. 

# INPUT: 
# -image_color_cropped- (original image taken and then cropped in -get_prediction_filter_coordinates- workflow). 
# -panoptic_labeled- (modified panoptic prediction mask obtained from Detectron2 trained model. It is already filtered to remove noise and assign independent labels to all the colonies in -get_prediction_filter_coordinates- workflow).
#
# OUTPUT: -final_list- (an ordered list with the scores assigned according to this criterium). 
#
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
#  ----------------------------------------------------------------------------------------------------------------------
# Import required libraries
import numpy as np
import cv2 as cv
#

def gfp_filter(image_color_cropped, panoptic_labeled):
    labels_objects=np.unique(panoptic_labeled)
    image_gfp=cv.cvtColor(image_color_cropped, cv.COLOR_BGR2RGB)
    image_grayscale=cv.cvtColor(image_gfp, cv.COLOR_BGR2GRAY)

    element_list=[]

    # Set color filter parameters
    for lb in labels_objects:
        bool_mask= panoptic_labeled== lb
        
        # Compute average intensity value for colony 
        average_gfp_colony=np.mean(image_grayscale[bool_mask])
        
        # Store it in a list with its label
        element_list.append((lb,average_gfp_colony))
        #    
    #
    # Create sorted list with objects (colonies) labels and their scores 
    dtype = [('label', int), ('value', float)]
    list_sorted = np.array(element_list, dtype=dtype)
    list_sorted=np.sort(list_sorted, order='value')

    final_list=[] # Larger score means more aggreement with selected criterium
    score=len(list_sorted) # Scoring is a natural number series within range [1,len(list_sorted)]
    for tupla in list_sorted:
        final_list.append([tupla, score])
        score-=1
        #
    #
    final_list=np.asarray([[a,b,c] for [(a,b),c] in final_list])
    
    return final_list
    #
#