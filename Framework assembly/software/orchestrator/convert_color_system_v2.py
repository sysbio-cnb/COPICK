# -*- coding: utf-8 -*-
# convert_color_system_v2.py - Python script in charge of transforming RGB color values to HSV system or viceversa.
#
# GENERAL DESCRIPTION:
#---------------------
# This script includes two functions to transform a color value from HSV into RGB (hsv_to_rgb) or from RGB into HSV (rgb_to_hsv) color systems, respectively. 
#
# INPUT:  -rgb_color-/-hsv_color- (lists of color values to transform).
#
# OUTPUT: -color_hsv- (list of transformed color values in H, S, V order).
#        --------------------------------------------------------------------------------------------
#         -rgb_color_normalized- (list of transformed and normalized color values in R, G, B order).          
#         -bgr_color- (list of transformed and normalized color values in B, G, R order).
#
#--------------------------------------------------------------------------------
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
#------------------------------------------------------------------------------

import colorsys

def rgb_to_hsv(rgb_color):

    # Get rgb percentage: range (0-1, 0-1, 0-1)
    # Target reference color is in RGB order
    red_percentage= rgb_color[0]/float(255)
    green_percentage= rgb_color[1]/float(255)
    blue_percentage=rgb_color[2]/float(255)
    
    # get hsv percentage: range (0-1, 0-1, 0-1)
    color_hsv_percentage=colorsys.rgb_to_hsv(red_percentage, green_percentage, blue_percentage) 
    
    # get normal hsv: range (0-360, 0-1, 0-1)
    color_h=round(360*color_hsv_percentage[0]) #360º because it is a color wheel
    color_s=color_hsv_percentage[1]
    color_v=color_hsv_percentage[2]
    
    color_hsv=[color_h, color_s, color_v]
    
    return color_hsv
    #
#


def hsv_to_rgb(hsv_color):
    # Modify HSV color values
    hsv_color= (hsv_color[0]/360, hsv_color[1], hsv_color[2]) # H_value/360 due to colorsys values format
    rgb_color=colorsys.hsv_to_rgb(hsv_color[0], hsv_color[1], hsv_color[2])
    rgb_color_normalized=(rgb_color[0]*255, rgb_color[1]*255, rgb_color[2]*255) # normalize values
    bgr_color=(rgb_color_normalized[2], rgb_color_normalized[1], rgb_color_normalized[0]) # get color values in BGR order too. 
    
    return rgb_color_normalized, bgr_color
    #
#
    
    
    
    
    
    
    