# -*- coding: utf-8 -*-
# discriminate_colony_groups_v17.py - Python script in charge of splitting objects formed by multiple
# touching colonies. 
#
# GENERAL DESCRIPTION:
#---------------------
# This function uses a distance map referenced to the boundary of the target binary object
# to infer the number of colony centers (using a local maxima searcher algorithm) and further
# relabel the pixels inside the body applying a watershed algorithm  
# 
# INPUT: 
#        -panoptic_binary- Binary mask containing the pixel regions considered as colonies
#        -image_crop- Crop of the original white image
#        -panoptic_binary_labeled- Mask containing the labeled regions considered as valid segments after the consensus of the different inferences  
#        -binary_objects_labeled- list containing the regionprops associated object to each detected segment
#        -areas- list containing the areas of the segments in the binary mask
#        -slicee - list containing the slices objects of each segments in the binary mask
#        -max_label- latest label of the integer series forming the identifiers of each objects  
#        -xgrid - matrix grid containing the number of row for each position in the matrix itself 
#        -ygrid - matrix grid containing the number of column for each position in the matrix itself 

# OUTPUT: 
#        -panoptic_completed - Mask containing the updated labeled regions after splitting multicolony segments    
#        -number_of_colonies - Updated number of colonies after splitting multicolony segments
    
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
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.color import rgb2gray 
from skimage.segmentation import watershed
from scipy import ndimage as ndi

# import matplotlib.pyplot as plt
# from skimage.io import imread

def discriminate_colony_groups(panoptic_binary,panoptic_completed,image_crop,panoptic_binary_labeled,binary_objects_labeled,areas,slicee,max_label,xgrid,ygrid):
    
    # convert image to grayscale
    raw_image_gray=rgb2gray(image_crop)
    panoptic_binary=panoptic_completed>0     # obtain binary mask
    
    for i in range(0, len(binary_objects_labeled)):
        if areas[i]<=25 or areas[i]>=28000: # limit size (too small objects are considered noise)
            selected=binary_objects_labeled[i].label
            panoptic_binary[panoptic_binary_labeled==selected]=0
            #
        else:
            ## Detect tempative centers of colonies using max local peak intensity function
            # obtain grayscale slice
            # i=areas.index(sum(sum(panoptic_completed==24)))

            raw_image_sliced=1-raw_image_gray[slicee[i]]
            
            # remove from grayscale image all pixels belonging the bounding box but not selected as colony by Detectron
            group_box=panoptic_binary[slicee[i]]
            
            # remove spureous detections of Detectron from object slices         
            label_group_box=label(group_box)
            group_box_objects=regionprops(label_group_box)
            areass=[]
            pixels=[]
            for z in range(0, len(group_box_objects)):
                areass.append(group_box_objects[z].area)
                pixels.append(group_box_objects[z].coords)
                #
            #
            largest_object_index=areass.index(max(areass))
            
            for z in range(0, len(group_box_objects)):
                if z != largest_object_index:
                    for t in range(0,len(pixels[z])):
                        x=pixels[z][t][0]
                        y=pixels[z][t][1]
                        group_box[x][y]=False
                        #
                    #
                #
            #                
            
            raw_image_sliced[np.logical_not(group_box)]=0 
            
            # Generate the markers as local maxima of the distance to the background
            distance = ndi.distance_transform_edt(group_box) 
            coords = peak_local_max(distance, min_distance=8, footprint=np.ones((10, 10)), labels=group_box) #np.ones((6,6))
            
            if np.shape(coords)[0]==1:
                continue
            
            mask = np.zeros(distance.shape, dtype=bool)
            mask[tuple(coords.T)] = True
            markers, _ = ndi.label(mask)
            labels = watershed(-distance, markers, mask=group_box)

            # store global coordinates of crop
            xgrid_slice_global=xgrid[slicee[i]] # global coord crop
            ygrid_slice_global=ygrid[slicee[i]]
            
            # compute meshgrid of local crop image
            dimensions=np.shape(raw_image_sliced)
            row_range=np.arange(0,dimensions[0])
            column_range=np.arange(0,dimensions[1])
            xgrid_slice_local,ygrid_slice_local=np.meshgrid(column_range,row_range)               
        
            max_label=np.max(panoptic_completed) # update max label value
            new_labels=np.shape(coords)[0]
            next_labels=np.arange(max_label+1,max_label+new_labels+1)
            
            labels_colonies_local=np.arange(1,new_labels+1)
            for tag in range(0,len(labels_colonies_local)):
                panoptic_label=next_labels[tag]
                mask=labels==labels_colonies_local[tag]
                cordx_global_mask=xgrid_slice_global[mask]
                cordy_global_mask=ygrid_slice_global[mask]
                for p in range(0,len(cordx_global_mask)):
                    panoptic_completed[cordy_global_mask[p]][cordx_global_mask[p]]=panoptic_label
                #
            #
        #
    #
    number_of_colonies=len(np.unique(panoptic_completed))-1 # discard background label 
    
    # plt.imshow(panoptic_completed)
    # plt.imshow(label2rgb(panoptic_completed))
    # plt.imshow(labels)
    # plt.imshow(distance)
    # plt.imshow(mask)

    return  panoptic_completed,number_of_colonies
#

        