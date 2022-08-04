# -*- coding: utf-8 -*-
#
# white_screen_10s.py - Python script to connect the white light of Adafruit RGB LED Pannel for 5 seconds
#
# GENERAL DESCRIPTION:
#----------------------
# When the script is executed, it connect LED Matrix to emmit white light for 5 seconds.
#
# INPUT: None
#
# OUTPUT: None
#
#The MIT License (MIT)
#
#Copyright (c) 2020 David R. Espeso, Irene del Olmo
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

import time
import sys

from rgbmatrix import RGBMatrix, RGBMatrixOptions

options=RGBMatrixOptions()
options.rows=32
options.cols=64
options.hardware_mapping='adafruit-hat'
options.brightness=100
options.gpio_slowdown=4
options.disable_hardware_pulsing=True
options.pwm_bits=11


matrix=RGBMatrix(options=options)
#matrix.luminanceCorrect=True
max_x=matrix.width
max_Y=matrix.height

matrix.Fill(255,255,255) # R,B G
#matrix.Fill(255,125,255) # pale yellow

time.sleep(5)
sys.exit(0)
