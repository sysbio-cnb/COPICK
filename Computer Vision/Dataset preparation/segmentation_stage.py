# -*- coding: utf-8 -*-

# segmentation_stage.py - Python script to perform (panoptic) segmentation of images from the COPICK dataset.

# (Panoptic segmentation task description paper: https://arxiv.org/abs/1801.00868)

# GENERAL DESCRIPTION:
# -------------------
# When the Function is called, it computes image and panoptic (every pixel in the image) segmentation on pretreated images from augmented dataset.
# Function called in -process_dataset- script after image_pretreatment function.

# INPUTS (outputs of image_pretreatment function):
# (-image_raw_name-, -image_crop-, -mask_image_crop-,
# -mask_plate-, -mask_agar_inner-, -mask_agar_ring-, -mask_background-,
# -mask_colonies_inner_raw-,-mask_agar_border-, -centroid- and -areas_median-)

# OUTPUTS (to be used in process_dataset script):
#        -mask_colonies- (final binary mask with all the colonies in the plate)
#        -mask_panoptic- (panoptic segmentation mask: every pixel in the image is segmented, not just the colonies)
#        -area_colonies- (list of final colonies areas)
#        -label_colonies- (list of final colonies labels)
#        -bbox_colonies- (list of final colonies bounding boxes)
#        -final_elems- (array of different filtered elements)
#        -mask_panoptic_labeled- (-mask_panoptic- labeled mask)
#        -panoptic_objects- (number of detected objects in -mask_panoptic-).

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
from skimage import morphology
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature
from skimage.measure import label, regionprops
import copy
#from skimage.color import label2rgb
#from skimage.morphology import convex_hull_image, convex_hull_object
import pandas as pd


def segmentation_stage(image_raw_name, image_crop, mask_image_crop,mask_plate, mask_agar_inner, mask_agar_ring, mask_background,mask_colonies_inner_raw, mask_agar_border, centroid, areas_median, combinations, id_segment_list):
    # Screen parameters used to estimate max ands min area size of colonies
    area_min_colony_pixels=50 # minimum area size that detected bodies must have to be considered as colonies (measured from images)
    height_photo=23 # cm
    pixels_heigth=2475
    pixels_per_cm=pixels_heigth/height_photo # pixels/cm
    
    diameter_max_colony=3 # cm
    diameter_pixels_max_colony=diameter_max_colony*pixels_per_cm
    area_max_colony_pixels=np.pi*diameter_pixels_max_colony**2/4  # maximum area size that detected bodies must have to be considered as colonies (measured from images)
    max_circularity=1.2 # manually fit
    min_circularity=0.4 # manually fit
    colonies2x_limit_axis_ratio=1.5 # manually fit
    colonies3x_limit_axis_ratio=2.5 # manually fit

    
    # Generate submask to detect colonies on the borders by using binary operations       
    image_levelset_ring = image_crop[:,:,0] # select the clearest channel of the image
    image_levelset_ring=255-image_levelset_ring # invert color image
    image_levelset_ring[~mask_agar_ring]=0 # remove all pixels not contained in agar outer ring
    # plt.imshow(image_levelset_ring, cmap="gray")
    # plt.show()

    #  apply random_noise, canny edge, erosion, fill_holes and top_hat filters to isolate colonies in the border of the agar
    image_canny = random_noise(image_levelset_ring, mode='s&p') # or mode='speckle'
    # plt.imshow(image_canny, cmap="gray")
    # plt.show()
    
    edges = feature.canny(image_canny, sigma=2)
    # plt.imshow(edges, cmap="gray")
    # plt.show()
    
    image_levelset_ring[edges]=0 # remove brightest lines of the border 
    # plt.imshow(image_levelset_ring, cmap="gray")
    # plt.show()
    
    # Filter image plates with data from the csv database
    csvdata = pd.read_csv (r'<path to csv database file>', delimiter= ";")
    im_name_csv=list(csvdata.filename)
    im_medium_csv=list(csvdata.medium)
    im_colonies_size_csv=list(csvdata.colonysize)
    im_category_csv=list(csvdata.category)
    
    position=im_name_csv.index(image_raw_name)
    
    im_name=im_name_csv[position]
    im_medium=im_medium_csv[position]
    im_colonies_size=im_colonies_size_csv[position]
    im_category=im_category_csv[position]
    
    # Establish a threshold intensity value based on Cit-M9 or LB plates to remove pixels with lower intensity values (colonies are the brightest elements in the image)
    if im_medium=="LB":
    #LB plates: keep intensity values over 218 (manually fit)
        image_levelset_ring[image_levelset_ring<218]=0  
    else:
    #Cit-M9 plates: keep intensity values over 165 (manually fit)
        image_levelset_ring[image_levelset_ring<165]=0 
        #
    #
    ring_eroded=morphology.binary_erosion(image_levelset_ring, morphology.disk(3)) # binary erosion of elements in the ring border area of the plate
    # plt.imshow(ring_eroded, cmap="gray")
    # plt.show()
    
    fill_holes=ndi.binary_fill_holes(ring_eroded,structure=np.ones((3,3))) # fill holes of connected elements after erosion
    # plt.imshow(fill_holes, cmap="gray")
    # plt.show()
    
    footprint = morphology.disk(3) # disk erosion filter
    top_hat = morphology.white_tophat(fill_holes, footprint) # apply top_hat filter
    
    ring_clear=fill_holes^top_hat # remove remaining noise in the ring border area of the plate
    # plt.imshow(ring_clear, cmap="gray")
    # plt.show()
   
    # Filter inner binary mask depending on colony size
    mask_colonies_inner_labeled=label(mask_colonies_inner_raw, return_num=False, connectivity=2) # label inner colonies
    info_objects_inner=regionprops(mask_colonies_inner_labeled)
    mask_colonies_inner=copy.deepcopy(mask_colonies_inner_raw)  # create filtered matrix

    areas=[]
    for i in range(0, len(info_objects_inner)): 
        areas.append(info_objects_inner[i].area) # create area list
        #
    #
    objects_histo=np.histogram(areas) # get areas distribution

    if areas_median > 1990:  # manually set for BIGGEST COLONIES
        for i in range(0, len(info_objects_inner)):
            if areas[i]<=objects_histo[1][1]:
                lab=info_objects_inner[i].label
                mask_colonies_inner[mask_colonies_inner_labeled==lab]=0 # remove the smallest objects in the areas histogram
    else: # for smallest and medium colonies
        info_objects_ids=np.arange(0,len(info_objects_inner), dtype=int)
        
        elems=np.asarray([info_objects_ids, areas]).T 
        elems=elems[elems[:,-1].argsort()[::-1]]
        
        indices=np.arange(0,len(elems))
        indices=indices[elems[:,1]>=area_min_colony_pixels]
        target=np.delete(elems,indices,0)  # select those with areas smaller than "area_min_colony_pixels"
        
        mask_colonies_inner=copy.deepcopy(mask_colonies_inner_raw)  # create filtered matrix
        
        for i in range(0, len(target)):
            mask_colonies_inner[mask_colonies_inner_labeled==info_objects_inner[int(target[i][0])].label]=0 # remove objects smaller than "area_min_colony_pixels"
            #
        #
    #   
    # plt.imshow(mask_colonies_inner, cmap="gray")
    # plt.show()
    
    
    # Final MASK with all colonies: filter depending on the presence of colonies (1) or not (0) in the plate, as categorized in the csv database
    mask_all_agar=~mask_agar_border
    if im_category==0:
        mask_colonies=copy.deepcopy(mask_all_agar)
        mask_colonies_labeled  = label(mask_colonies, return_num=False, connectivity=2)
        #info_objects_final = regionprops(mask_colonies_labeled)
    else:    
        mask_colonies= mask_colonies_inner + ring_clear 
        mask_colonies_labeled =label(mask_colonies, return_num=False, connectivity=2)
        #info_objects_final = regionprops(mask_colonies_labeled) # obtain properties of found objects 
        #
    #
    # plt.imshow(mask_colonies, cmap="gray")
    # plt.show()
    
    # display original image + colonies mask overlay
    plt.figure()
    plt.imshow(mask_image_crop, cmap="gray")
    plt.imshow(mask_colonies, cmap="gray", alpha=0.5)
    plt.show()
    

    #  labeled mask with positive background to create panoptic mask containing background, agar and colonies          
    mask_panoptic= np.zeros(np.shape(mask_colonies)) 
    mask_panoptic[mask_background]=1 # assing label 1 to dark background out of the plate
    # plt.imshow(mask_panoptic, cmap="gray")
    # plt.show()                             
    mask_plate_outer_border=mask_agar_border^mask_background>0
    # plt.imshow(mask_plate_outer_border, cmap="gray")
    # plt.show()
    mask_panoptic[mask_plate_outer_border]=2  # assing label 2 to plastic border of plate
    # plt.imshow(mask_panoptic, cmap="gray")
    # plt.show() 
    mask_panoptic[mask_all_agar]=3 # assing label 3 to agar
    # plt.imshow(mask_panoptic, cmap="gray")
    # plt.show()                                     

    # obtain objects properties
    colony_objects= regionprops(mask_colonies_labeled) 
    n_colonies=len(colony_objects)
    
    if n_colonies!=1: #1 means an empty plate
        mask_colonies_labeled[mask_colonies_labeled!=0]=mask_colonies_labeled[mask_colonies_labeled!=0]+3  # shift labels of colonies 3 units
        mask_panoptic[mask_colonies_labeled!=0]=mask_colonies_labeled[mask_colonies_labeled!=0]  # annotate colony labels in panoptic mask as they are given
        #
    #
    w,h,c=np.shape(image_crop)
    mask_panoptic_rgb = np.zeros((w,h,3), dtype=int)        
    mapp_ids2rgbs= np.vectorize(lambda inputt : combinations[int(inputt)]) # create a function to assign colors to ids based on a color combinations list

    for i in range(0,mask_panoptic.shape[0]): 
        for j in range(0,mask_panoptic.shape[1]):
            color=mapp_ids2rgbs(mask_panoptic[i][j])
            mask_panoptic_rgb[i][j][0]=color[0]
            mask_panoptic_rgb[i][j][1]=color[1]
            mask_panoptic_rgb[i][j][2]=color[2]
            #
        #
    # 
    plt.imshow(mask_panoptic_rgb)
    plt.show()
    
    mapp_ids = np.vectorize(lambda inputt : id_segment_list[int(inputt)]) # create a function to assign the specific id to corresponding color (colored segment in image)
    mask_panoptic=mapp_ids(mask_panoptic) 

    # label panoptic mask and get final variables of interest from objects properties
    mask_panoptic_labeled=label(mask_panoptic, return_num=False,connectivity=2)
    panoptic_objects=regionprops(mask_panoptic_labeled)
    
    area_colonies=[]
    label_colonies=[]
    bbox_colonies=[]
    axis_ratio=[]
    centroid=[]
    ncolony=[]

    for i in range(0, len(panoptic_objects)): 
        area_colonies.append(panoptic_objects[i].area) 
        label_colonies.append(panoptic_objects[i].label)
        # adjust bounding box values 
        square=list(panoptic_objects[i].bbox)
        bbx=copy.deepcopy(square[1])
        bby=copy.deepcopy(square[0])
        square[2]=abs(square[2]-square[0])
        square[3]=abs(square[3]-square[1])
        square[0]=bbx
        square[1]=bby
        bbox_colonies.append(tuple(square))
        # 
    #
    # filter objects by axis ratio
    for i in panoptic_objects:
        try:
            new_axis_ratio=i.major_axis_length/i.minor_axis_length 
        except: 
            new_axis_ratio=0.0 
        axis_ratio.append(np.float(new_axis_ratio))
        centroid.append(i.centroid)
        ncolony.append(1)
        #
    #    
    n_objects=len(axis_ratio)
    final_elems = np.asarray([(np.arange(0,n_objects)), axis_ratio, ncolony]).T 
    
    # Differentiate 2x and 3x groups of colonies with the same label  
    indices=np.arange(0,len(final_elems))
    indices_sel=indices[final_elems[:,1]>colonies2x_limit_axis_ratio]  # select elements with indicated colony size (2x)
    final_elems[indices_sel,2]=2*np.ones(len(indices_sel)) # assign number of colonies=2 for the selected elements
    indices_sel=indices[final_elems[:,1]>colonies3x_limit_axis_ratio]  # select elements with indicated colony size (3x)
    final_elems[indices_sel,2]=3*np.ones(len(indices_sel)) # assign number of colonies=3 for the selected elements
    
    ## Visualize panoptic mask
    #mask_panoptic_rgb = label2rgb(mask_panoptic, bg_label=0) 
    #plt.imshow(mask_panoptic_rgb)
    #plt.show()

    return mask_colonies, n_colonies, mask_panoptic, area_colonies, label_colonies, bbox_colonies, mask_colonies_labeled, colony_objects, final_elems, mask_panoptic_labeled, panoptic_objects, mask_panoptic_rgb 
    #
#