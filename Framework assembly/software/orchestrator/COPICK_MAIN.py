# COPICK_MAIN.py - Python script in charge of executing a colony picking protocol.
#
# GENERAL DESCRIPTION:
#---------------------
# When the script is executed in Orchestrator PC, the script connect OT-2 via SSH, and position the camera on place (transilluminator). Then, it capture a snapshot of the plate in transilluminator
# and generate a prediction using a trained neural network (based in Detectron). The output is a panoptic segmentation that is used to identify the colonies and apply different set of filters to select the colonies of interest.
# Finally, the script create a csv file containing the robot coordinates of selected colonies and upload it to the OT-2 server with a flag file that the robot will read and start the picking process
#
# INPUT: None
#
# OUTPUT: None
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
#------------------------------------------------------------------------------

#Import required libraries
import os
os.path.join('C:\\OT-2\\colony_picking\\') 
os.chdir('C:\\OT-2\\colony_picking\\')

if __name__ == '__main__':
    
    import paramiko
    from scp import SCPClient
    import time
    import datetime
    
    from vimba import *
    import myCamera_Alvium as myCamera    
    
    # import CNN and pytorch
    import torch, torchvision
    import detectron2
    
    # import some common libraries
    import numpy as np
    import cv2 as cv
    import matplotlib.pyplot as plt
    import copy
    import csv
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    
    #import required functions from folder
    from color_filter_v3 import color_filter
    from gfp_filter_v1 import gfp_filter
    from crop_photo_v3 import crop_photo
    from size_filter_v3 import size_filter
    
    # import some common detectron2 utilities
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    
    from detectron2.utils.logger import setup_logger
    setup_logger()
    
# CAMERA SETTINGS

camera_settings=dict()
camera_settings['IMG_PATH']='C:/OT-2/colony_picking/images/'
camera_settings['FORMAT']='.JPG'
camera_settings['WIDTH_IMG']=2592
camera_settings['HEIGHT_IMG']=1944
camera_settings['GAIN_GFP']=300000 #
camera_settings['GAIN_WHITE']=13872.99#4586 #17331.52 #

# DETECTRON WEIGHT PATHS
weights_paths=["C:/OT-2/colony_picking/model_weights/optuna_study2_37/model_final.pth","C:/OT-2/colony_picking/model_weights/optuna_study2_30/model_final.pth"]

transillum_settings=dict()
transillum_settings['f']=0.051 # mm/pixel equivalence ratio
transillum_settings['CX_labware']=1295+5# X-coordinate (pixel) of labware center
transillum_settings['CY_labware']=987+5# Y-coordinate (pixel) of labware center

# TRANSILLUMINATOR RASPBERRY PI SETTINGS
raspberry_pi=dict()
raspberry_pi['host']="192.168.1.2" # Raspberry pi ip address
raspberry_pi['port']=22 # Raspberry pi port
raspberry_pi['username']="My_user_name" # Raspberry pi user
raspberry_pi['password']="1234" # My Raspberry pi password

raspberry_pi['command']=dict()
raspberry_pi['command']['switch_white_light_5s']="cd rpi-rgb-led-matrix \n cd bindings \n cd python \n cd samples \n sudo python white_screen_5s.py \n"
raspberry_pi['command']['switch_white_light_10s']="cd rpi-rgb-led-matrix \n cd bindings \n cd python \n cd samples \n sudo python white_screen_10s.py \n"
raspberry_pi['command']['switch_white_light_60s']="cd rpi-rgb-led-matrix \n cd bindings \n cd python \n cd samples \n sudo python white_screen_60s.py \n"   

raspberry_pi['command']['switch_blue_light_5s']="cd rpi-rgb-led-matrix \n cd bindings \n cd python \n cd samples \n sudo python blue_screen_5s.py \n"
raspberry_pi['command']['switch_blue_light_10s']="cd rpi-rgb-led-matrix \n cd bindings \n cd python \n cd samples \n sudo python blue_screen_10s.py \n"
raspberry_pi['command']['switch_blue_light_60s']="cd rpi-rgb-led-matrix \n cd bindings \n cd python \n cd samples \n sudo python blue_screen_60s.py \n"   


# SSH DATA OF OT-2
robot_ip = '192.168.1.1' # OT-2 IP address
usr_name='root' # OT-2 username
ot2_key_path='C:\\Users\\My_path\ot2_ssh_key' # OT-2 ssh key path

# PATH DIRECTORIES
pc_isready_path='C:\\OT-2\\colony_picking\\isready.txt'
ot2_isready_path='/var/lib/jupyter/notebooks/isready.txt'

pc_csv_path='C:\\OT-2\\colony_picking\\images\\colony_list_plate_0.csv'
ot2_csv_path='/var/lib/jupyter/notebooks'

# COLONY SELECTION SETTINGS
only_counting=False # Set true if you want only a counting of colonies
filter_criterium=['size'] # ['color','counting','gfp'] # set the criterium to be used in the colony selection
target_reference_color=[47.09, 60.77, 8.21] # green-blueish color #Inverted values in BGR order # configure the color to be used as reference for selection
n_colonies_to_pick=30 # number of colonies to pick

max_size_colony=25*transillum_settings['f'] # max and min allowable size of colonies in mm = pixel diameter * scaling factor to mm
min_size_colony=28000*transillum_settings['f']

if 'gfp' in filter_criterium: 
    is_gfp=True
else:
    is_gfp=False
#

def evaluate_colony(mark_size,mark_color,mark_gfp): # function to evaluate the "quality" of each colony based on the criterium selected
    weight_color=2/5 #  importance of each trait in % per . Their summatory must 
    weight_size=3/5
    weight_gfp=0
    try:
        assert weight_color+ weight_size+ weight_gfp==1
    except AssertionError:
        print('The sum of all weights must be equal 1')
    #
    final_score=weight_size*mark_size+weight_color*mark_color+weight_gfp*mark_gfp
    return final_score
      

def upload_file_to_OT2(path_pc, path_ot2): # Function to upload a file to the OT2 server
    with paramiko.SSHClient() as ssh:
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=robot_ip, username=usr_name, key_filename=ot2_key_path,disabled_algorithms={'pubkeys': ['rsa-sha2-256', 'rsa-sha2-512']})
        
        # send file
        scp = SCPClient(ssh.get_transport())
        scp.put(path_pc, path_ot2)
        scp.close()
        ssh.close()
        #
    #
#


def switch_light(color, duration): # Function to switch the light on the Raspberry Pi

    if color=='white': # switch white light
        if duration<=5:
            cmd=raspberry_pi['command']['switch_white_light_5s']
        elif duration > 5 and duration <= 10:
            cmd=raspberry_pi['command']['switch_white_light_10s']
        else:
            cmd=raspberry_pi['command']['switch_white_light_60s']
            #
        #
    #
    if color == 'blue': # switch blue light
        if duration<=5:
            cmd=raspberry_pi['command']['switch_blue_light_5s']
        elif duration > 5 and duration <= 10:
            cmd=raspberry_pi['command']['switch_blue_light_10s']
        else:
            cmd=raspberry_pi['command']['switch_blue_light_60s']
            #
        #            
    #
 
    with paramiko.SSHClient() as ssh: # connect to the Raspberry Pi
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(raspberry_pi['host'], raspberry_pi['port'], raspberry_pi['username'], raspberry_pi['password'],timeout=5)
        
        stdin,stdout,stderr = ssh.exec_command(cmd)
        ssh.close()
        #
    #
#


def acquire_image(color): # Function to acquire an image from the camera
    if color=='white':
        basename = 'ME3I_white_' # set a base name for the image 
        suffix = datetime.datetime.now().strftime("%d%m%y_%H%M%S") # add the current date and time to the image name
        filename = "_".join([basename, suffix]) # join the base name and the date and time
        print("gather data and store in path....")
        #
    #
    if color=='blue':
        basename = 'ME3I_gfp_' #str(uuid.uuid4())+'_'+color 
        suffix = datetime.datetime.now().strftime("%d%m%y_%H%M%S")
        filename = "_".join([basename, suffix]) 
        print("gather data and store in path....")
        #
    #
    
    IMG_NAME=filename+camera_settings['FORMAT'] # final name of the image
    PATH_FILE= camera_settings['IMG_PATH']+ IMG_NAME #  path of the image
    
    # Create myCamera object
    cam = myCamera.myCamera(0, color=True)
    
    # Set some properties
    cam.set('width', camera_settings['WIDTH_IMG'])
    cam.set('height', camera_settings['HEIGHT_IMG'])
    if color=='white':
        cam.set('exposure', camera_settings['GAIN_WHITE']) 
    elif color=='blue':
        cam.set('exposure', camera_settings['GAIN_GFP']) 
    #
    
    # Retrieve some properties
    print('Resolution: %dx%d' % ( cam.get('width'), cam.get('height')) )
    
    # Take a picture
    frame = cam.snapshot()
    frame_rotated=cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE) # rotate 90º because camera snapshot and robot axis handle are shifted 90 degrees
    # Save picture
    cv.imwrite(PATH_FILE, frame_rotated)

    # Close camera
    cam.close()
    #
    return PATH_FILE, frame_rotated
#

    
def grab_image(is_gfp): # composed function to acquire an image by activating the light
    color='white'
    duration=3
    
    time_flag_ini=time.time()
    switch_light(color, duration)   # switch on light
    PATH_FILE_WHITE, camera_image_white=acquire_image(color)              # take the image
    while (time.time()-time_flag_ini) <= duration:
        time.sleep(1)               # wait until transilluminator is off
    #
    if (is_gfp == True):
        color='blue'
        duration=10  
        
        time_flag_ini=time.time()
        switch_light(color, duration)   # switch on light
        PATH_FILE_GFP, camera_image_gfp=acquire_image(color)              # take the image
        while (time.time()-time_flag_ini) <= duration:
            time.sleep(1)               # wait until transilluminator is off
        #
        return PATH_FILE_WHITE, camera_image_white, PATH_FILE_GFP, camera_image_gfp
    else:
        return PATH_FILE_WHITE, camera_image_white
    #
#

def load_detectron(weight): # Function to load the configuration and weights of detectron2
    #Set config
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu' 
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 3 #n semantic+1  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.WEIGHTS=weight

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01  # set a low threshold to get the largest amount of predictions (colonies)
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.01
    cfg.TEST.DETECTIONS_PER_IMAGE = 500
    cfg.MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH = 0.5
    cfg.MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT = 2048 #4096
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.1
    predictor = DefaultPredictor(cfg)
    
    return predictor
# 

def detect_colonies(images_raw,paths): # Function to detect colonies in the image
    
    image_raw_white=images_raw[0]
    path_white=paths[0]
    # CROP GATHERED PHOTO WITH PLATE AND RETRIEVE ITS CENTER IN IMAGE
    center_plate_labware, image_crop_white=crop_photo(image_raw_white ) 

    crop_images=[image_crop_white]
    #Save cropped image
    pos_dot=path_white.find('.')
    crop_path_white=path_white[0:pos_dot]+'_crop'+path_white[pos_dot:len(path_white)]
    cv.imwrite(crop_path_white, image_crop_white)         

    if len(images_raw) > 1:
        image_raw_gfp=images_raw[1]
        path_gfp=paths[1]
        dummy1, image_crop_gfp=crop_photo(image_raw_gfp, path_gfp)  
        pos_dot=path_gfp.find('.')
        crop_path_gfp=path_gfp[0:pos_dot]+'_gfp_crop'+path_gfp[pos_dot:len(path_gfp)]
        cv.imwrite(crop_path_gfp, image_crop_gfp) 
        crop_images.append(image_crop_gfp)         
        #
    #
    image_crop= image_crop_white   
    
    binary_predictions_weights=[]

    for weights in weights_paths: # perform the detection using two different predictions by using different weights

    # STEP 0: LOAD DETECTRON
        predictor=load_detectron(weights)

        # GENERATE PANOPTIC IMAGE
        preds_val=predictor(image_crop)["panoptic_seg"]
        
        # Save predicted mask as *.png
        im_panoptic=np.asarray(preds_val[0])   
        
        # POSTPROCESS PREDICTED IMAGE TO REMOVE NO REAL COLONIES (NOISE)
        panoptic_pred=copy.deepcopy(im_panoptic) #image is saved as a 3 channel image and we pick the last channel
    
        # Get ids (labels) from predicted panoptic segmentation masks
        labs=np.unique(panoptic_pred)  # the 2 last ids are always the plate and out of plate regions excluding the colonies
    
        # Modify panoptic prediction in order to compute the center coordinates of the image
        filtered_pred=copy.deepcopy(panoptic_pred)
        filtered_pred[filtered_pred==0]=labs[-3]+3 ## asign high id value to first predicted element so that it is not 0
        filtered_pred[filtered_pred==labs[-2]]=0 # assign 0 to outside of plate region
    
        dimensions=np.shape(filtered_pred)
        row_range=np.arange(0,dimensions[0])
        column_range=np.arange(0,dimensions[1])
        xgrid,ygrid=np.meshgrid(row_range,column_range)    
        
        x_center_crop=np.round(np.mean(xgrid[filtered_pred != 0]))
        y_center_crop=np.round(np.mean(ygrid[filtered_pred != 0]))
        
        center_plate_crop=[x_center_crop, y_center_crop]
        
        # Modify panoptic prediction to assign 0 to colonies 
        panoptic_mod=copy.deepcopy(panoptic_pred)
       
        for elem in labs:
            if (elem != labs[-1] and elem != labs[-2]):
                panoptic_mod[panoptic_mod==elem]=0
                #
            #
        #
        # Get binary mask with the predicted colonies footprint only
        panoptic_binary=panoptic_mod==0
        
        # Label binary mask and filter colonies by size
        label_panoptic_binary=label(panoptic_binary)
        binary_objects=regionprops(label_panoptic_binary)
        
        areas=[]  
        for i in range(0, len(binary_objects)):
            areas.append(binary_objects[i].area)
            #
        #
        for i in range(0, len(binary_objects)):
            if areas[i]<=25 or areas[i]>=28000: # limit size (too small objects are considered noise and the biggest are the stickers used to fix the plate)
                selected=binary_objects[i].label
                panoptic_binary[label_panoptic_binary==selected]=0
                #
            #
        #
        binary_predictions_weights.append(panoptic_binary)
        #
    #
    binary_predictions_weights=np.asarray(binary_predictions_weights,dtype=bool)
    panoptic_binary=np.zeros(np.shape(panoptic_pred),dtype=bool)
    shapee=np.shape(panoptic_binary)
    for i in range(0,shapee[0]):
        for j in range(0,shapee[1]):
            panoptic_binary[i,j]=binary_predictions_weights[0,i,j] and binary_predictions_weights[1,i,j]
            #
        #
    #
    
    # Label filtered binary mask and assign a different label to each colony (object)
    panoptic_completed=np.zeros(np.shape(panoptic_pred))
    panoptic_binary_filtered=label(panoptic_binary)
    binary_objects_filtered=regionprops(panoptic_binary_filtered)
    
    number_of_colonies=len(binary_objects_filtered)    
    for obj in binary_objects_filtered:
        lab=obj.label
        pix=obj.coords
        for elem in pix:
            panoptic_completed[elem[0],elem[1]]=lab
            #
        #
    #
    plt.imshow(panoptic_completed, cmap="gray")
    plt.show()
    
    pos_crop=crop_path_white.find('crop')    
    panoptic_path=crop_path_white[0:pos_crop]+'_prediction'+'.png'
    cv.imwrite(panoptic_path, panoptic_completed)
    
    return panoptic_completed, crop_images, number_of_colonies, center_plate_labware, center_plate_crop     
    #
#

def colony_selection(crop_images, panoptic_completed, filter_criterium, n_colonies_to_pick): # function to select the colonies to be picked from the whole set detected in the plate 
    if 'counting' in filter_criterium:
        only_counting=True
        return only_counting
    else:
        sorted_list_size=[]
        sorted_list_color=[]
        sorted_list_gfp=[]
        panoptic_labeled=label(panoptic_completed) 
        info_objects_sel=regionprops(panoptic_labeled)
        
        labels=np.unique(panoptic_labeled)
        labels=labels[1:len(labels)]
        if 'size' in filter_criterium: # if size is in the filter criterium, we need to sort the colonies by size in descending order
            sorted_list_size=size_filter(max_size_colony, min_size_colony, panoptic_labeled)
        if 'color' in filter_criterium: # if color is in the filter criterium, we need to evaluate how close is each colony to reference color. Then sort the colonies by color in ascending order (smaller score means more similarity in color)
            sorted_list_color=color_filter(crop_images[0], panoptic_labeled, target_reference_color)
        if 'gfp' in filter_criterium: # if gfp is in the filter criterium, we need to sort the colonies by light intensity in descending order
            sorted_list_gfp=gfp_filter(crop_images[1], panoptic_labeled)
        #
        
        score_list=[]
        mark_size=0
        mark_color=0
        mark_gfp=0
        for lab in labels:
            if 'size' in filter_criterium:
                pos=np.where(sorted_list_size[:,0]==lab)
                mark_size=sorted_list_size[pos[0][0],2]   
            if 'color' in filter_criterium:
                pos=np.where(sorted_list_color[:,0]==lab)
                mark_color=sorted_list_size[pos[0][0],2]
            if 'color' in filter_criterium:
                pos=np.where(sorted_list_gfp[:,0]==lab)
                mark_gfp=sorted_list_gfp[pos[0][0],2]
                #
            final_score= evaluate_colony(mark_size,mark_color,mark_gfp)   
            score_list.append((lab, mark_size, mark_color, mark_gfp, final_score))
            #
        #
        dtype = [('label', np.int64), ('mark_size', float),('mark_color', float),('mark_gfp', float),('total_score', float)]
        score_list = np.array(score_list, dtype=dtype)
        score_list=np.sort(score_list, order='total_score')        
        score_list_sorted = score_list[::-1]
        
        # erase last objects after number of desired colonies
        counter=0
        for index in score_list_sorted:  
            # first n elements (number of elements to select from the total)
            if counter>=n_colonies_to_pick:
                bool_mask= panoptic_labeled== index[0]
                panoptic_labeled[bool_mask]=0 # remove element because it has not passed filter 
                #
            #
            counter+=1     
            #
        #
        # truncate list according to maximum number of desired colonies
        score_list_sorted=score_list_sorted[0:n_colonies_to_pick]
        
        return info_objects_sel,score_list_sorted
    #
#

def create_csv(info_objects_sel,score_list_sorted, center_plate_labware,center_plate_crop):   # function to create a csv file with the robot coordinates of the selected colonies
    labels=[] #get objects ids
    centroids=[] #get objects centroids
    areas2=[] #get objects areas
    
    offset_labware_to_plate_x=center_plate_labware[1]-transillum_settings['CX_labware'] # Incremental shift between plate position and labware center
    offset_labware_to_plate_y=center_plate_labware[0]-transillum_settings['CY_labware']
   
    for i in range(0, len(score_list_sorted)):
        pos=score_list_sorted[i][0]-1
        areas2.append(info_objects_sel[pos].area)
        labels.append(info_objects_sel[pos].label)
        centroids.append(info_objects_sel[pos].centroid) 
        #
    #
    # Get cartesian relative coordinates (respect to labware center) of selected colonies for Opentrons
    cart_centroids=[] # transform centroid values to radians: 
    for i in range(0, len(centroids)):
        offset_colony_to_center_crop_x=centroids[i][0]-center_plate_crop[0] # Incremental shift between colony position and plate center 
        offset_colony_to_center_crop_y=centroids[i][1]-center_plate_crop[1]
        
        colony_labware_shift_x = offset_labware_to_plate_x + offset_colony_to_center_crop_x
        colony_labware_shift_y = offset_labware_to_plate_y + offset_colony_to_center_crop_y
        
        colony_robot_shift_x=colony_labware_shift_x*transillum_settings['f']
        colony_robot_shift_y=colony_labware_shift_y*transillum_settings['f']
        
        cart_centroids.append((colony_robot_shift_y, -colony_robot_shift_x)) # change x by y and invert sign of y to match image coord system with robot coord system
        
        #
    # write csv file
    header = ['label', 'area', 'colony_robot_shift_x', 'colony_robot_shift_y']

    with open(camera_settings['IMG_PATH']+ 'colony_list_plate_0'+'.csv', 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f)
    
        # write the header
        writer.writerow(header)
    
        # write the data
        for i in range(0, len(centroids)):
            data = [labels[i], areas2[i], cart_centroids[i][0], cart_centroids[i][1]]
           
            writer.writerow(data)    
            #
        #    
    #
#

#### PROTOCOL ####

# STEP 0: CREATE ISREADY FILE IN ORCHESTRATOR PC
path_pc=pc_isready_path
with open(path_pc,'w') as txt_file:  # overwrite existing isready file with a NO statement to halt the protocol when read in OT-2
    txt_file.write('NO')
    txt_file.close()
#
path_pc=pc_isready_path
path_ot2=ot2_isready_path
upload_file_to_OT2(path_pc, path_ot2) # upload isready file to OT-2


# STEP 1: START PROTOCOL IN OT-2
cmd='cd /var/lib/jupyter/notebooks \n jupyter nbconvert --ExecutePreprocessor.timeout=600 --execute colony_picking_jupyter.ipynb'
with paramiko.SSHClient() as ssh2:
    ssh2.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh2.connect(hostname=robot_ip, username=usr_name, key_filename=ot2_key_path,disabled_algorithms={'pubkeys': ['rsa-sha2-256', 'rsa-sha2-512']})
   
    stdin,stdout,stderr = ssh2.exec_command(cmd)
    ssh2.close()
    #
#
print('ORDER SENT BY SSH')

# STEP 2: WAIT TILL OT-2 POSITION THE CAMERA IN PLACE TO TAKE THE PICTURE
positioning_timeout=60 # seconds
time.sleep(positioning_timeout)

# STEP 3: GATHER THE IMAGE
if is_gfp == True:
    PATH_FILE_WHITE, camera_image_white, PATH_FILE_GFP, camera_image_gfp = grab_image(is_gfp)
    image_layers=[camera_image_white,camera_image_gfp]
    path_layers=[PATH_FILE_WHITE,PATH_FILE_GFP]
else:
    PATH_FILE_WHITE, camera_image_white=grab_image(is_gfp)
    image_layers=[camera_image_white]
    path_layers=[PATH_FILE_WHITE]
#

# STEP 4: PROCESS IMAGE, DETECT PLATE COLONIES USING DETECTRON 2 
panoptic_completed, crop_images, number_of_colonies, center_plate_labware, center_plate_crop=detect_colonies(image_layers,path_layers)
print('TOTAL NUMBER OF DETECTED COLONIES = ' + str(number_of_colonies))

# STEP 5: APPLY SELECTION CRITERIA
info_objects_sel,score_list_sorted= colony_selection(crop_images, panoptic_completed, filter_criterium, n_colonies_to_pick)

# STEP 6: GENERATE OUTPUT CSV FILE
create_csv(info_objects_sel,score_list_sorted, center_plate_labware,center_plate_crop)

if only_counting == False:
    # STEP 7: UPLOAD CSV FILE TO OT-2
    path_pc=pc_csv_path
    path_ot2=ot2_csv_path
    upload_file_to_OT2(path_pc, path_ot2)
    
    # STEP 8: UPLOAD ISREADY.TXT FILE = TRUE
    path_pc=pc_isready_path
    path_ot2=ot2_isready_path
    
    with open(path_pc,'w') as txt_file:  # overwrite existing isready file with a YES statement to keep picking
        txt_file.write('YES')
        txt_file.close()
    #
    upload_file_to_OT2(path_pc, path_ot2) # upload isready file to OT-2
#
# STEP 9: WAITING THE ROBOT TO FINISH THE JOB