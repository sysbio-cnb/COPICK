# -*- coding: utf-8 -*-
#
# colony_picking_jupyter.py - Python script added into a jupyter notebook inside OT-2 in charge of executing the picking of a colony plate
#
# GENERAL DESCRIPTION:
#----------------------
# When the script is executed via SSH, the script home the robot and position the camera on place (transilluminator). Then, it waits for the orchestrator to receive the
# data related to picking operation. To do so, it checks a txt file called "is_ready.txt". If the file contains the word "YES", then the script starts the picking process
# by reading the position of selected colonies from the uploaded file "colony_list_plate_0.csv" and start picking them one by one. Each colony is picked by the robot and
# transferred to a 96-well plate/1-well plate for further incubation. 
#
# INPUT: None
#
# OUTPUT: None
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

import opentrons.execute
from opentrons import types
import sys
sys.path.append('/var/lib/jupyter/notebooks')
import csv, math, os, json

# Load protocol API
protocol = opentrons.execute.get_protocol_api('2.13')

# First robot homing
protocol.home()

# Load tips 
tiprack_p20 = protocol.load_labware('opentrons_96_tiprack_20ul', '3')
tiprack_p300 = protocol.load_labware('opentrons_96_tiprack_300ul', '9')
tip_box_list_p20=[tiprack_p20]
tip_box_list_p300=[tiprack_p300]
   
# Load Pipettes
left_p300_pipette = protocol.load_instrument('p300_single_gen2', 'left', tip_racks=tip_box_list_p300) # default aspiration flow rate = 46.43 ul/s 
right_p20_pipette = protocol.load_instrument('p20_single_gen2', 'right', tip_racks=tip_box_list_p20)  # default aspiration flow rate = 150 ul/s  
    
# Load target plate to transfer the picked colonies
target_wellplate=protocol.load_labware('corning_96_wellplate_360ul_flat', '5') # we load a 96-well plate definition to get all the wells positions, but actually used a one-well plate filled with agar to transfer the picked colonies to the respective positions.

# Parse json file in notebook to gather transilluminator labware object
with open('<transillum_labware_definition.json>') as labware_file:
    labware_def = json.load(labware_file)
    transillum = protocol.load_labware_from_definition(labware_def, 7)
    #
#


#####  PROTOCOL ####
    
# STEP 1: Move the camera to the center of the transilluminator    
offset_camera_x=22 # offset of the camera in x direction
offset_camera_y=-53 # offset of the camera in y direction
center_location = transillum['A1'].center()              
adjusted_location = center_location.move(types.Point(x=offset_camera_x, y=offset_camera_y, z=15))
right_p20_pipette.move_to(adjusted_location)

# STEP 2A: Wait 10 seconds to gather the image and process the data
colony_list_ready=''
for i in range(0,10): # try during 2 min and 30 sec
    with open('/var/lib/jupyter/notebooks/isready.txt','r') as txt_file:
        is_ready=txt_file.readline()
        txt_file.close()
        #
    #
    if is_ready=='NO':
        colony_list_ready=False
    elif is_ready=='YES':
        colony_list_ready=True
        with open('/var/lib/jupyter/notebooks/isready.txt', 'r+') as txt_file:  # Erase content
            txt_file.truncate(0)
            txt_file.write('NO')
            txt_file.close()
            #
        #
        break
    #
    protocol.delay(10)
    #
#
#protocol.set_rail_lights(True)

# STEP 3: Start picking the colonies or halt the protocol in case image gathering / anaysis failed 

if colony_list_ready == False:  # if csv update and ready flag is not raised, finish the protocol by homing
   protocol.home() 
   #
elif colony_list_ready == True: # STEP 3: Start picking only if boolean flag has been set as True
       
    # Load csv list with samples to process
    colony_coord_cart_dict=[]
    row_number=0
    with open('/var/lib/jupyter/notebooks/colony_list_plate_0.csv','r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for row in csv_reader:
            colony_coord_cart_dict.append(row)
            #
        #
    #
    transferred_colonies=0
    z_agar_surface=-38.5  # adjust depending on agar thickness. Default value works fine with 25 ml of agar in a Petri dish.
    z_agar_pick=-39       # adjust depending on agar thickness. Default value works fine with 25 ml of agar in a Petri dish.
    z_up= -28 
    z_up_labware=15 
    
    transferred_colonies=0
    
for colony in colony_coord_cart_dict:
    x_coord=float(colony['colony_robot_shift_x']) 
    y_coord=float(colony['colony_robot_shift_y'])
    z_coord=z_up

    if transferred_colonies < 96: # Change total number of colonies to pick

        ### Picking commands
        # Pick up tip
        #right_p20_pipette.reset_tipracks() 
        right_p20_pipette.pick_up_tip(presses=1, increment=0)
        
        # go to XY position up
        center_location = transillum['A1'].center()              
        adjusted_location = center_location.move(types.Point(x=x_coord, y=y_coord, z=z_up))
        right_p20_pipette.move_to(adjusted_location, speed=100)             

        # Move down the tip to pick the colony
        adjusted_location = center_location.move(types.Point(x=x_coord, y=y_coord, z=z_agar_pick)) 
        right_p20_pipette.move_to(adjusted_location, speed=100)
        protocol.delay(1)
        #protocol.pause()

        # drag biomass                
        adjusted_location = center_location.move(types.Point(x=x_coord, y=y_coord-0.2, z=z_agar_surface)) 
        right_p20_pipette.move_to(adjusted_location,speed=100)
        adjusted_location = center_location.move(types.Point(x=x_coord+0.2, y=y_coord, z=z_agar_surface)) 
        right_p20_pipette.move_to(adjusted_location,speed=100)

        # Move up the tip 
        adjusted_location = center_location.move(types.Point(x=x_coord, y=y_coord, z=z_up_labware))
        right_p20_pipette.move_to(adjusted_location, speed=100)                    
       

        # Move to next empty well of target plate
        next_well=target_wellplate.wells()[transferred_colonies]
        center_location = next_well.center()
        adjusted_location = center_location.move(types.Point(x=0, y=0, z=-2.75)) # adjust z coordinate depending on the thickness of the agar in the target plate.
        right_p20_pipette.move_to(adjusted_location, speed=100)
        protocol.delay(1)
        
        right_p20_pipette.drop_tip()
        transferred_colonies += 1
        #
    #
#
#