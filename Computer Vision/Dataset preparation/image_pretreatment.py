# -*- coding: utf-8 -*-

# image_pretreatment.py - Python script to transform images from the colony picking dataset (COPICK) before segmentation performance. 

# GENERAL DESCRIPTION:
# -------------------
# When the function is called, original images from the augmented dataset are pre-processed and filtered using different masks for further segmentation steps.
# Function called in process_dataset script after augmentation_stage function.

# INPUTS: -image_raw- (loaded image from augmented dataset folder), -filename- (image name path).

# OUTPUTS (to be used in the segmentation_stage function):
#        -image_raw_name- (image name).
#        -image_crop- (cropped image adjusted to the plate dimensions. As the images from the COPICK dataset are already cropped with the crop_photo.py script, it is a copy of -image_raw-).
#        -mask_image_crop- (binary mask of image_crop).
#        -mask_plate- (binary mask of the region relative to the plate).
#        -mask_agar_inner- (binary mask of the inner region corresponding to the agar in the plate).
#        -mask_agar_ring- (binary mask of the region corresponding to the border of the plate).
#        -mask_background- (binary mask of the region outside the plate).
#        -mask_colonies_inner_raw- (binary mask of colonies in the inner region of the plate).
#        -mask_agar_border- (binary mask of the region corresponding to the plate without the border or ring).
#        -centroid- (center of image and plate).
#        -areas_median- (median area value of colonies in the plate).

#--------------------------------------------------------------------------------
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
#  $Version: 1.0 $  $Date: 2023/02/15 $
#
#--------------------------------------------------------------------------------

# Import required libraries
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
import cv2 as cv
#from matplotlib import pyplot as plt
#import matplotlib
import numpy as np
from statistics import mean
import copy
from skimage.morphology import convex_hull_object


def image_pretreatment(image_raw, filename):

    image_raw_name=filename[-35:] # pick last characters from filename path to obtain image name (previously created by file_namer fuction)
    max_size_colony=340 # manually fit
    
    
    # Detect whole agar plate area, estimate the radius and centroid
    mask_image_raw=image_raw[:,:,2] # field 2 is the brightest and always selects the whole agar plate
    otsu_threshold=threshold_otsu(mask_image_raw) # binarize image by Otsu's threshold
    mask_image_raw[mask_image_raw<=otsu_threshold]=0
    mask_image_raw[mask_image_raw>otsu_threshold]=1
    #plt.imshow(mask_image_raw, cmap="gray")
    #plt.show()    
    
    # label binary objects
    mask_image_raw_labeled=label(mask_image_raw,return_num=False, connectivity=2)
    info_objects_raw=regionprops(mask_image_raw_labeled) 

    # get objects areas
    areas_bbox=[]
    area_colonies=[]
    for i in range(0, len(info_objects_raw)): # compute bounding box areas and areas of objects in binary mask
        area_bbox= info_objects_raw[i].bbox_area       
        areas_bbox.append(area_bbox)
        area_colonies.append(info_objects_raw[i].area)
        #
    #
    #  find real center of plate and radius based on plate footprint
    info_objects_ids=np.arange(0,len(info_objects_raw), dtype=int)
    elems= np.asarray([info_objects_ids, areas_bbox, area_colonies]).T # create array to filter and select the biggest objects by area
    elems=elems[elems[:,1].argsort()[::-1]]  # order by descending area size
    label_plate= elems[0][0]   # the largest detected element will be the plate footprint
    centroid=info_objects_raw[label_plate].centroid  # store the centroid
    plate_bbox= info_objects_raw[label_plate].bbox # get bounding box width and height
    plate_bbox=list(plate_bbox)
    # get centroid coordinates of footpring (plate)
    cx=centroid[1]     
    cy=centroid[0]
    # get radius
    r=0.5*np.mean([plate_bbox[2]-plate_bbox[0], plate_bbox[3]-plate_bbox[1]])  
    margin_shift=10 # set number of pixels of tolerance
    
    
    ##### If images are not cropped to fit the plate size use:
    ## Compute cropping area to reduce image size and crop center of image (plate)
    # bbox_xywh=[abs(round(cx-r-(0.5*margin_shift))),abs(round(cy-r-(0.5*margin_shift))), round(2*r+margin_shift), round(2*r+margin_shift)]
    # image_crop=image_raw[bbox_xywh[1]:bbox_xywh[1]+bbox_xywh[3], bbox_xywh[0]:bbox_xywh[0]+bbox_xywh[2],:]
    
    # Save cropped images
    # cv.imwrite(<path to store cropped images> + image_raw_name, image_crop)
    #####

    # If images are already cropped use image_crop directly
    image_crop=copy.deepcopy(image_raw) 
    #plt.imshow(image_crop, cmap="gray")
    #plt.show()
    

    # Create the new cropped agar plate mask
    H,W,C=image_crop.shape # get image_crop dimensions
    y=np.arange(0,H)     
    x=np.arange(0,W)     
    [X,Y]=np.meshgrid(x,y)
    
    #  get adjusted centroid coordinates
    cx_shifted=round(r+(0.5*margin_shift))
    cy_shifted=round(r+(0.5*margin_shift))    
    
    #  create binary mask of whole agar plate
    mask_plate=((X-cx_shifted)**2)+((Y-cy_shifted)**2)<=r**2  
    #plt.imshow(mask_plate, cmap="gray")
    #plt.show()
    
    
    # Remove dark background color from image
    mask_plate_rgb=mask_plate
    image_crop[~mask_plate_rgb]=0
    # plt.imshow(image_crop, cmap="gray")
    # plt.show()
    
    # Create mask of original cropped image
    mask_image_crop=image_crop[:,:,0].astype(np.uint8) # select clearest channel of image
    # plt.imshow(mask_image_crop, cmap="gray")
    # plt.show()
    
    mask_image_crop_blurred = cv.GaussianBlur(mask_image_crop, (41, 41), 0) # blurred (with manually fit values) mask of the original cropped image in order to apply thresholding methods later
    # plt.imshow(mask_image_crop_blurred, cmap="gray")
    # plt.show()


    # Preview and filter images per colony size before applying different threshold methods 
    mask_crop_inner=copy.deepcopy(mask_image_crop)
    colonies_size_preview=((X-cx_shifted)**2)+((Y-cy_shifted)**2)<(0.8*r)**2 # apply reduced inner mask for median colony size preview on each plate
    # plt.imshow(colonies_size_preview, cmap="gray")
    # plt.show()
    mask_crop_inner[~colonies_size_preview]=0
    # plt.imshow(mask_crop_inner, cmap="gray")
    # plt.show()
    
    mask_crop_inner_blurred = cv.GaussianBlur(mask_crop_inner, (41, 41), 0) # blurred (with manually fit values) mask of the inner cropped image in order to apply thresholding methods
    # plt.imshow(mask_crop_blurred, cmap="gray")
    # plt.show()
 
    mask_crop_inner_thresh = cv.adaptiveThreshold(mask_crop_inner_blurred,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 61, 2) # apply adaptive gaussian threshold (with manually fit values)
    mask_crop_inner_thresh=255-mask_crop_inner_thresh #invert mask values
    # plt.imshow(mask_crop_inner_thresh, cmap="gray")
    # plt.show()
    
    colonies_size_preview_border=((X-cx_shifted)**2)+((Y-cy_shifted)**2)<(0.78*r)**2 # create preview mask to select colonies in the inner region of the plate (excluding border)
    # plt.imshow(colonies_size_preview_border, cmap="gray")
    # plt.show()
   
    mask_crop_inner_thresh[~colonies_size_preview_border]=0 # select colonies of the inner region of the plate for a preview of their size
    # plt.imshow(mask_crop_inner_thresh, cmap="gray")
    # plt.show()
    
    mask_crop_convex = convex_hull_object(mask_crop_inner_thresh) # connect points belonging to the same colony by convex hull method after thresholding
    # plt.imshow(mask_crop_convex, cmap="gray")
    # plt.show()
    
    mask_crop_convex_labeled=label(mask_crop_convex, return_num=False, connectivity=2) # label preview inner colonies mask
    info_objects_preview = regionprops(mask_crop_convex_labeled)  
    
    areas_inner_colonies=[]
    
    for i in range(0, len(info_objects_preview)):
        areas_inner_colonies.append(info_objects_preview[i].area) # get inner colonies areas
        #
    #
    areas_inner_hist=np.histogram(areas_inner_colonies) # get preview of colonies areas values distribution
 
    who_is_noise=areas_inner_colonies<areas_inner_hist[1][1] # define smallest area value points as noise
    areas_numeric=np.asarray(areas_inner_colonies)
    areas_median=np.median(areas_numeric[~who_is_noise]) # get median area value of the rest of points excluding the noise
    
    # Apply different binary adaptive threshold method to original cropped image depending on the estimated median area value of colonies in the previewed inner mask
    if areas_median < 1990: #1993 (min median value in the whole dataset) or 10935(max median value in the whole dataset). Manually tested
        # for small and medium colonies, lower threshold values are chosen in order to avoid noise
        mask_binary_positive = cv.adaptiveThreshold(mask_image_crop_blurred,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 39, 2) 
    else:    
        # for biggest colonies, threshold value has to be higher in order to avoid overcounting
        mask_binary_positive = cv.adaptiveThreshold(mask_image_crop_blurred,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 63, 2) 
        #
    #
    # plt.imshow(mask_binary_positive, cmap="gray")
    # plt.show()

    # Label colonies in the binary image after filter and thresholding
    mask_binary_positive_labeled=label(mask_binary_positive, return_num=False, connectivity=2) 
    info_objects_positive = regionprops(mask_binary_positive_labeled)  
    
    #  compute different region properties of the objects (colonies) in the binary image
    areas_bbox=[]
    area_colonies=[]
    height=[]
    width=[]
    bbox_coords=[]
    centers=[]
    for i in range(0, len(info_objects_positive)):  
        bbox_coords.append(info_objects_positive[i].bbox)  
        centers.append(info_objects_positive[i].centroid) 
        areas_bbox.append(info_objects_positive[i].bbox_area)   
        height.append(info_objects_positive[i].bbox[3]-info_objects_positive[i].bbox[1]) 
        width.append(info_objects_positive[i].bbox[2]-info_objects_positive[i].bbox[0])
        area_colonies.append(info_objects_positive[i].area) 
        #
    #
    # Filter objects based on bounding box areas
    info_objects_ids=np.arange(0,len(info_objects_positive), dtype=int)
    elems= np.asarray([info_objects_ids, areas_bbox]).T 
    elems=elems[elems[:,1].argsort()] # get elements in ascending order
    indices=np.arange(0,len(elems))
    indices=indices[elems[:,1]<1900**2]
    elems=np.delete(elems,indices,0) # discard small objects 


    # Get inner center of plate based on euclidean distances of points in binary image to create different region masks
    euc_distances=[]
    for i in range(0, len(elems)):  
        d1=np.sqrt(((centers[elems[i][0]][0])-(0.5*len(mask_binary_positive)))**2 + ((centers[elems[i][0]][1])-(0.5*len(mask_binary_positive)))**2) #euclidean distance of image geometric center coords
        d2=np.sqrt(((centers[elems[i][0]][0])-(cx_shifted))**2 + ((centers[elems[i][0]][1])-(cy_shifted))**2) # euclidean distance between centers
        distances=[d1,d2]
        euc_distances.append(mean(distances))
        criteria=np.asarray([euc_distances, elems[:,1]]).T # set criteria based on euclidean distances and bounding box areas of previously filtered elements in the plate
        #
    #
    mark= (1-criteria[:,0]/max(criteria[:,0])) + (1-criteria[:,1]/max(criteria[:,1])) # apply criteria
    
    chosen=elems[mark==max(mark)] # get chosen elements based on criteria
    
    inner_center=[((info_objects_positive[chosen[0][0]].bbox[0])+(0.5*(info_objects_positive[chosen[0][0]].bbox[2]-info_objects_positive[chosen[0][0]].bbox[0]))), ((info_objects_positive[chosen[0][0]].bbox[1])+(0.5*(info_objects_positive[chosen[0][0]].bbox[3]-info_objects_positive[chosen[0][0]].bbox[1])))] # get final inner center of plate  
    

    # Create all required masks for efficient image segmentation

    # create reduced mask plate (most inner agar)
    mask_agar_inner=((X-inner_center[0])**2 + (Y-inner_center[1])**2) <= (0.87*r)**2
    #plt.imshow(mask_agar_inner, cmap="gray")
    #plt.show()
    
    #  create mask for dark outer background
    mask_background=~mask_plate   
    #plt.imshow(mask_background, cmap="gray") 
    #plt.show()
    
    #  create mask to point to all agar surface able to host colonies
    mask_agar_border=((X-cx_shifted)**2 + (Y-cy_shifted)**2 >= (0.93*r)**2 - mask_background)>0 
    #plt.imshow(mask_agar_border, cmap="gray")
    #plt.show()
    
    #  create mask to point to the "hard segmentation area" of the agar: the outermost part of the whole agar available to host colonies. It has a ring shape
    mask_agar_ring=~(mask_agar_border^mask_agar_inner) 
    #plt.imshow(mask_agar_ring, cmap="gray")
    #plt.show()
    
    #  perform required operations to create a mask containing only detected objects (potential colonies) in the inner part of agar
    mask_colonies_inner_raw= mask_binary_positive 
    #plt.imshow(mask_colonies_inner_raw, cmap="gray")
    #plt.show()
    
    #  create mask excluding the points out of the inner region of the plate
    mask_colonies_inner_raw[~mask_agar_inner]=0
    #plt.imshow(mask_colonies_inner_raw, cmap="gray")
    #plt.show()
    
    #  include border region without the colonies
    mask_colonies_inner_raw=(mask_colonies_inner_raw + (mask_plate^mask_agar_inner))>0 
    #plt.imshow(mask_colonies_inner_raw, cmap="gray")
    #plt.show()
    
    #  invert values
    mask_colonies_inner_raw=(1-mask_colonies_inner_raw)>0
    #plt.imshow(mask_colonies_inner_raw, cmap="gray")
    #plt.show()
    
    #  exclude background from binary mask
    mask_colonies_inner_raw[mask_background]=0    
    #plt.imshow(mask_colonies_inner_raw, cmap="gray")    
    #plt.show()
    

    # Label objects from binary mask and get properties of interest
    mask_colonies_inner_raw_labeled=label(mask_colonies_inner_raw, return_num=False, connectivity=2)
    info_objects = regionprops(mask_colonies_inner_raw_labeled) 
    areas_bbox=[]
    area_colonies=[]
    label_colonies=[]
    height=[]
    width=[]
    for i in range(0, len(info_objects)): 
        height.append(info_objects[i].bbox[3]-info_objects[i].bbox[1]) 
        width.append(info_objects[i].bbox[2]-info_objects[i].bbox[0]) 
        areas_bbox.append(info_objects[i].bbox_area)
        area_colonies.append(info_objects[i].area) 
        tag=info_objects[i].label
        label_colonies.append(tag) 
        #
    #
    # Final filtering of inner region of the plate (where most of the colonies are: mask_colonies_inner_raw)
    aspect=np.asarray(height)/np.asarray(width)
    aspect[aspect<1]=1/aspect[aspect<1]
    info_objects_ids=np.arange(0,len(info_objects), dtype=int)
    elems=np.asarray([info_objects_ids, areas_bbox, area_colonies, label_colonies, height, width, aspect]).T
    
    # set filter conditions based on size
    chosen= np.logical_or(np.logical_or(np.logical_or((np.asarray(areas_bbox) >= max_size_colony**2),(np.asarray(height)>= max_size_colony)), (np.asarray(width) >= max_size_colony)),(aspect > 3))                                  
    if sum(chosen)!=0:
        labels_chosen=elems[chosen,3] 
        for i in range(0, sum(chosen)): 
            mask_colonies_inner_raw[mask_colonies_inner_raw_labeled==labels_chosen[i]]=0 # filter mask
        #
    #
    #  additional filter to final mask based on colonies areas median
    if areas_median > 800: # arbitrary size limit, lower than minimum areas_median value to include more points for convex hull method
        mask_colonies_inner_raw= convex_hull_object(mask_colonies_inner_raw)  
    else: # avoid convex hull for the smallest objects, specially when they are very close, in order to avoid miscounting
        mask_colonies_inner_raw=copy.deepcopy(mask_colonies_inner_raw)
        #
    #
    # plt.imshow(mask_colonies_inner_raw, cmap="gray")
    # plt.show()

    
    return  image_raw_name, image_crop, mask_image_crop,mask_plate, mask_agar_inner, mask_agar_ring, mask_background,mask_colonies_inner_raw, mask_agar_border, centroid, areas_median
