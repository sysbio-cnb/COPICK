# size_filter.py - Function to filter detected colonies by size.
#
# GENERAL DESCRIPTION:
#---------------------
# Function to filter detected colonies by size.
#
# INPUT: 
# -max_size- (maximum size of colonies to select).
# -min_size- (minimum size of colonies to select).
# -panoptic_completed- (modified panoptic prediction mask obtained from Detectron2 trained model. It is already filtered to remove noise and assign independent labels to all the colonies in -get_prediction_filter_coordinates- workflow).
#
# OUTPUT: An ordered list with the scores assigned according with this criterium 
#
#--------------------------------------------------------------------------------
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
#--------------------------------------------------------------------------------

# Import required libraries
# if __name__ == '__main__':
import numpy as np
from skimage.measure import regionprops

# Define size_filter function
def size_filter(max_size, min_size, panoptic_labeled):
    element_list=[]
    # Label filtered mask and get colonies areas
    info_objects_pred=regionprops(panoptic_labeled)
    
    for i in range(0, len(info_objects_pred)):
        area_filtered=info_objects_pred[i].area
        label_filtered=info_objects_pred[i].label
        element_list.append((label_filtered, area_filtered))
        #
    #
    dtype = [('label', int), ('value', float)]
    list_sorted = np.array(element_list, dtype=dtype)
    list_sorted=np.sort(list_sorted, order='value')
    list_sorted=list_sorted[::-1]  # reorder elements, because we want largest element first (descending order)  

    final_list=[]
    counter=len(list_sorted)
    for tupla in list_sorted:
        if tupla[1]>min_size and tupla[1]<max_size: # exclude colonies with size outside of the range
            final_list.append([tupla, counter])
            #
        counter-=1
        #
    #
    final_list=np.asarray([[a,b,c] for [(a,b),c] in final_list])

    return final_list