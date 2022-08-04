# -*- coding: utf-8 -*-

# file_namer.py - Python script to generate random names for images.

# GENERAL DESCRIPTION:
# --------------------
# When the function is called, it generates random names from original names of images in a dataset.
# Function called in the augmentation_stage function.

# INPUTS: -n_names- (number of names to be generated) and -length_name- (length of each new name).

# OUTPUTS: -names- is used in the augmentation_stage function.

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


#!/usr/bin/env python

# Import required libraries
import numpy as np
import copy


def file_namer(n_names, length_name): # generate "n_names" random name of length "length_name" 
    unicode_letters=list(range(97,123)) # decimal numbers equivalent to latin leters in unicode format
    unicode_numbers=list(range(48,58)) # decimal numbers equivalent to numbers in unicode format
    unicode_signs=[*unicode_letters, *unicode_numbers] #combine letters and number signs
   
    random_integer_string = np.random.randint(0, len(unicode_signs), size=(n_names, length_name)) #create array of random unicode signs with size determined by the input variables
    
    names=[]
    for i in range(0, len(random_integer_string)): # create final list of names
        char_list=[chr(unicode_signs[j]) for j in random_integer_string[i]]  
        #print(char_list)
        names.append("".join(char_list))
        names=copy.deepcopy(names)
        #print(names)  
    return names
    #
#
