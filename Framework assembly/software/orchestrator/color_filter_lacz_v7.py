# -*- coding: utf-8 -*-
# color_filter_lacz_v7.py - Python script in charge of executing a colony picking protocol.
#
# GENERAL DESCRIPTION:
# ----------------------------------------------------------------------------------------------------------------------
# Function to filter and select colonies in plate by color. Given a target color value, selection criteria can be a fixed spectrum of simmilarity in color values 
# or a specific quantity (number) of colonies that are most similar to the target color

# INPUT: 
# -image_color_cropped- (original image taken and then cropped in -get_prediction_filter_coordinates- workflow). 
# -panoptic_labeled- (modified panoptic prediction mask obtained from Detectron2 trained model. It is already filtered to remove noise and assign independent labels to all the colonies in -get_prediction_filter_coordinates- workflow).
# -info_objects_sel- (labeled objects in panoptic_labeled mask).
# -target_reference_color_RGB- (RGB ordered color values corresponding to the target color we want to select).

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
# ----------------------------------------------------------------------------------------------------------------------

# Import required libraries and functions
import numpy as np
from convert_color_system_v2 import rgb_to_hsv

def color_filter_lacz(image_color_cropped, panoptic_labeled, info_objects_sel, target_reference_color_RGB):
    R=image_color_cropped[:,:,0]
    G=image_color_cropped[:,:,1] # check rgb are in 1,2,3 order in matrix
    B=image_color_cropped[:,:,2]

    # HSV color threshold values to discriminate which color is considered as "blue" and which one is "white"
    H_limit_low=95
    H_limit_up=220
    S_limit_up=0.35
    V_limit_down=0.70 

    target_reference_color_HSV=rgb_to_hsv(target_reference_color_RGB)

    element_list=[]
    final_list=[]
    
    # Set color filter parameters
    for i in range(0,len(info_objects_sel)):
        lb=info_objects_sel[i].label        # pick label
        slicee=info_objects_sel[i].slice    # pick slice (bounding box range) of selected object in image

        panoptic_labeled_slice=panoptic_labeled[slicee]  # store values of selected objects panoptic matrix 
        footprint_colony_slice=panoptic_labeled_slice>0    # generate binary footprint of detected object 
        
        # Select R, G, B slices per colony
        R_slicee=R[slicee]
        G_slicee=G[slicee]
        B_slicee=B[slicee]
        
        # Get pixel average color value of selected colony
        average_color_colony_RGB=[np.mean(R_slicee[footprint_colony_slice]),np.mean(G_slicee[footprint_colony_slice]),np.mean(B_slicee[footprint_colony_slice])]
        average_color_colony_HSV=rgb_to_hsv(average_color_colony_RGB) 

        # Color_dissimilarity is the Euclidean distances between average color and target color to evaluate it with an score: the shortest the distance the blueish the colony (more similar to target).        
        color_dissimilarity=np.sqrt(((average_color_colony_HSV[0]-target_reference_color_HSV[0])/360)**2+(average_color_colony_HSV[1]-target_reference_color_HSV[1])**2+(average_color_colony_HSV[2]-target_reference_color_HSV[2])**2)

        # If current pixel is within the selected HSV volume, discard the colony because it is considered as "blue" 
        if not(average_color_colony_HSV[0]<H_limit_low or average_color_colony_HSV[0]>H_limit_up):
            pick_colony=0
        elif not(average_color_colony_HSV[1]<S_limit_up):
            pick_colony=0
        elif not(average_color_colony_HSV[2]>V_limit_down):
            pick_colony=0
        else:
            pick_colony=1
        #
        element_list.append((lb, pick_colony, color_dissimilarity, average_color_colony_HSV[0], average_color_colony_HSV[1], average_color_colony_HSV[2]))
        #
    #
    # Create sorted list with objects (colonies) labels and their scores (more or less similar to target color)
    dtype = [('label', int), ('pick_colony',int), ('value', float), ('H', float), ('S', float), ('V', float)]
    
    list_sorted = np.array(element_list, dtype=dtype)
    list_sorted=np.sort(list_sorted, order=['pick_colony','value']) 
    list_sorted=list_sorted[::-1]  # reorder elements, because we want most dissimilar element first (white colonies)  
    
    final_list=np.asarray([list(elem) for elem in list_sorted])

    return final_list
    #
#