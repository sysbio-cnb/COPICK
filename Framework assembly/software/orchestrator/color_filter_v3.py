# -*- coding: utf-8 -*-
# color_filter_v3.py - Python script in charge of executing a colony picking protocol.
#
# GENERAL DESCRIPTION:
# ----------------------------------------------------------------------------------------------------------------------
# Function to filter and select colonies in plate by color. Given a target color value, selection criteria can be a fixed spectrum of simmilarity in color values 
# or a specific quantity (number) of colonies that are most similar to the target color

# INPUT: 
# -image_color_cropped- (original image taken and then cropped in -get_prediction_filter_coordinates- workflow). 
# -panoptic_completed- (modified panoptic prediction mask obtained from Detectron2 trained model. It is already filtered to remove noise and assign independent labels to all the colonies in -get_prediction_filter_coordinates- workflow).
# -target_reference_color- (BGR ordered color values corresponding to the target color we want to select).

# OUTPUT: An ordered list with the scores assigned according with this criterium 
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
# ----------------------------------------------------------------------------------------------------------------------


# if __name__ == '__main__':
    # Import required libraries
import numpy as np

#Define color_filter function
def color_filter(image_color_cropped, panoptic_labeled, target_reference_color):
    
    labels_objects=np.unique(panoptic_labeled)
    
    #In Opencv color order is in Blue-Green-Red (BGR) order instead of RGB
    R=image_color_cropped[:,:,2]
    G=image_color_cropped[:,:,1] # check rgb are in 1,2,3 order in matrix
    B=image_color_cropped[:,:,0]

    element_list=[]
    final_list=[]

    # Set color filter parameters
    for lb in labels_objects:
        bool_mask= panoptic_labeled== lb
        
        ## Inverted color order
        # average_color_colony=[np.mean(R[bool_mask]),np.mean(G[bool_mask]),np.mean(B[bool_mask])]
        
        #Get average color values in the image
        average_color_colony=[np.mean(B[bool_mask]),np.mean(G[bool_mask]),np.mean(R[bool_mask])]
        
        # Compute euclidean distances between average color and target color (input value) 
        dist=np.sqrt((average_color_colony[0]-target_reference_color[0])**2+(average_color_colony[1]-target_reference_color[1])**2+(average_color_colony[2]-target_reference_color[2])**2)
        element_list.append((lb,dist))
        final_list.append
        #    
    #
    # Create sorted list with objects (colonies) labels and their scores (more or less similar to target color)
    dtype = [('label', int), ('value', float)]
    list_sorted = np.array(element_list, dtype=dtype)
    list_sorted=np.sort(list_sorted, order='value')

    final_list=[]
    counter=len(list_sorted)
    for tupla in list_sorted:
        final_list.append([tupla, counter])
        counter-=1
        #
    #
    final_list=np.asarray([[a,b,c] for [(a,b),c] in final_list])

    return final_list
    #
#