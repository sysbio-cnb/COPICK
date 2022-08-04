# -*- coding: utf-8 -*-

# process_panoptic_to_semantic.py - Modified Detectron2 Python script (-prepare_panoptic_fpn.py- in ...detectron2/datasets/) to obtain semantic annotations (masks) from panoptic masks in the dataset.

# GENERAL DESCRIPTION:
# -------------------
# When the script is executed, it assigns new ids to things and stuff elements to create semantic segmentation masks.
# Function called inside -separate_semantic_from_panoptic.py- script. 

# INPUTS (defined inside separate_semantic_from_panoptic.py):
#     -input_panoptic: loaded panoptic mask.
#     -segments: detected segments in panoptic masks. 
#     -id_map: map with ids reassignation in order to make stuff segments to have an integer id and things segments (colonies) to have an id=0. 
#     -sem_seg_root: path to store semantic segmentation masks.
    
    
# OUTPUTS: NONE (Semantic masks are stored in specified path)


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
import numpy as np
import cv2 as cv
import copy
from PIL import Image
from panopticapi.utils import rgb2id
from matplotlib import pyplot as plt
#from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES                                                                 


def process_panoptic_to_semantic(input_panoptic, segments, id_map, sem_seg_root): #(input_panoptic, output_semantic, segments, id_map, sem_seg_root)
    name=input_panoptic[-49:] # select last characters of panoptic image name
    panoptic = np.asarray(cv.imread(input_panoptic), dtype=np.uint32)
    panoptic = rgb2id(panoptic) # transform RGB values to ids
    output = np.zeros_like(panoptic, dtype=np.uint8) + 255
    for seg in segments:
        cat_id = seg["category_id"] # check segment category
        new_cat_id = id_map[cat_id] # id_map is a new id mapping created in -separate_semantic_from_panoptic- function
        output[panoptic == seg["id"]] = new_cat_id # assign id depending on thing/stuff category
        #
    #
    ## Display output
    # plt.imshow(output)
    # plt.show()
    #Save semantic masks
    cv.imwrite(sem_seg_root+name, output) # save semantic masks with Opencv instead of Image (due to image transformations)
    #
#



## Previous function version by Detectron2

# def _process_panoptic_to_semantic(input_panoptic, output_semantic, segments, id_map):
                                                                             
    # panoptic = np.asarray(Image.open(input_panoptic), dtype=np.uint32)
    # panoptic = rgb2id(panoptic)
    # output = np.zeros_like(panoptic, dtype=np.uint8) + 255
    # for seg in segments:
        # cat_id = seg["category_id"]
        # new_cat_id = id_map[cat_id]
        # output[panoptic == seg["id"]] = new_cat_id
    # Image.fromarray(output).save(output_semantic)                                           


