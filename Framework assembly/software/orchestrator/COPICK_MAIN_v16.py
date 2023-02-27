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

#Import required libraries
import os
# Set directory to work in
os.path.join('X:\\XXX\\XXX\\') 
os.chdir('X:\\XXX\\XXX\\')

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
    import math
    import cv2 as cv
    import matplotlib.pyplot as plt
    import copy
    import csv
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.color import label2rgb
    
    #import required functions from folder
    from gfp_filter_v2 import gfp_filter
    from crop_photo_v8 import crop_photo
    from size_filter_v5 import size_filter
    from discriminate_colony_groups_v17 import discriminate_colony_groups
    from color_filter_lacz_v7 import color_filter_lacz
    
    # import some common detectron2 utilities
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    
    from detectron2.utils.logger import setup_logger
    setup_logger()
    #
#

# CAMERA SETTINGS
camera_settings=dict()
camera_settings['IMG_PATH']='X:/XXX/XXX/images/' # Access path to store the images
camera_settings['FORMAT']='.JPG'
camera_settings['WIDTH_IMG']=2592
camera_settings['HEIGHT_IMG']=1944
camera_settings['GAIN_GFP']=1800000 #
camera_settings['GAIN_WHITE']= 300000 #(with filter) #(13572.99 without filter)

# DETECTRON WEIGHT PATHS
# Chose and load weights from folder
weights_paths=["X:/XXX/XXX/model_weights/SQ_30/model_final.pth","X:/XXX/XXX/model_weights/SQ_38/model_final.pth"]

# TRANSILLUMINATOR SETTINGS
transillum_settings=dict()
transillum_settings['f']=0.049 # 0.051 mm/pixel equivalence ratio
# Shift center labware in X and Y axes: Change manually to calibrate transilluminator labware and colonies positions !!!!!!!!
transillum_settings['CX_labware']=1378 # Shift center labware in Y axis: larger value means UPWARDS in robot axis
transillum_settings['CY_labware']=966  # Shift center labware in X axis: larger value means go closer to LEFT in robot axis


# PIXEL TO MM CALIBRATION FACTORS (Obtained from applying a linear fit between mm and pixel coordinates during calibration)
mx_fit_px_to_mm=0.05111254 # slope of the linear fitting when correlating mm to pixel coordinates in X axis
bx_fit_px_to_mm=-0.45055484 # Independent term of the linear fitting when correlating mm to pixel coordinates in X axis

my_fit_px_to_mm=0.05249069 # slope of the linear fitting when correlating mm to pixel coordinates in Y axis
by_fit_px_to_mm=0.125688  # Independent term of the linear fitting when correlating mm to pixel coordinates in Y axis    


# TRANSILLUMINATOR RASPBERRY PI SETTINGS
raspberry_pi=dict()
raspberry_pi['host']="XXX.XXX.XX.XX" # Set Raspberry pi ip address
raspberry_pi['port']=22 # Set Raspberry pi port
raspberry_pi['username']="xxx" # Set Raspberry pi user
raspberry_pi['password']="xxx" # Set Raspberry pi password

# Define Raspberry pi commands for led panel control
raspberry_pi['command']=dict()
raspberry_pi['command']['switch_white_light_5s']="cd rpi-rgb-led-matrix \n cd bindings \n cd python \n cd samples \n sudo python white_screen_5s.py \n"
raspberry_pi['command']['switch_white_light_10s']="cd rpi-rgb-led-matrix \n cd bindings \n cd python \n cd samples \n sudo python white_screen_10s.py \n"
raspberry_pi['command']['switch_white_light_60s']="cd rpi-rgb-led-matrix \n cd bindings \n cd python \n cd samples \n sudo python white_screen_60s.py \n"   

raspberry_pi['command']['switch_blue_light_5s']="cd rpi-rgb-led-matrix \n cd bindings \n cd python \n cd samples \n sudo python blue_screen_5s.py \n"
raspberry_pi['command']['switch_blue_light_10s']="cd rpi-rgb-led-matrix \n cd bindings \n cd python \n cd samples \n sudo python blue_screen_10s.py \n"
raspberry_pi['command']['switch_blue_light_60s']="cd rpi-rgb-led-matrix \n cd bindings \n cd python \n cd samples \n sudo python blue_screen_60s.py \n"   


# SSH DATA OF OT-2 
robot_ip = 'XXX.XXX.XXX.XXX' # >>> Insert OT-2 IP address
usr_name='root' # >>> Insert OT-2 username
ot2_key_path='X:\\XXX\\ssh keys\\ot2_ssh_key' # >>> Insert OT-2 ssh key path

# PATH DIRECTORIES
pc_isready_path='X:\\XXX\\XXX\\isready.txt' # Access path to -isready.txt- file in pc
ot2_isready_path='/var/lib/jupyter/notebooks/isready.txt' # Access path to -isready.txt- file in OT2 (via Jupyter Notebook)

pc_csv_path='X:\\XXX\\XXX\\XXX\\colony_list_plate_0.csv' # Access path to -colony_list_plate_0.csv- file with the colonies coordinates in pc
ot2_csv_path='/var/lib/jupyter/notebooks' # Access path to -colony_list_plate_0.csv- file with the colonies coordinates in OT2 (via Jupyter Notebook)


# COLONY SELECTION SETTINGS
only_counting=False # Set to True if only a counting of colonies is required
filter_criterium=['size'] # ['size', 'color','counting','gfp'] # Set the criterium/s to be used in the colony selection
target_reference_color_RGB=[56.0,87.0,108.0] # Introduce values in R,G,B order

n_colonies_to_pick=700 # Set maximum number of colonies to pick

# Set max and min allowable size of colonies in pixel= max size [mm]/ scaling factor to mm
min_size_colony_pixels=np.pi/4*1.275**2/transillum_settings['f'] 
max_size_colony_pixels=np.pi/4*28**2/transillum_settings['f']


if 'gfp' in filter_criterium: 
    is_gfp=True
else:
    is_gfp=False
    #
#

def evaluate_colony(mark_size,mark_color,mark_gfp): # Function to evaluate the "quality" of each colony based on the selected criterium 
    # Importance of each trait in %. Their summatory must be equal to 1 but they can be manually changed depending on preference. 
    weight_color=1/5 #2/5 
    weight_size=1/5 #3/5
    weight_gfp=3/5 #0
    try:
        assert weight_color+ weight_size+ weight_gfp==1
    except AssertionError:
        print('The sum of all weights must be equal to 1')
    #
    final_score=weight_size*mark_size+weight_color*mark_color+weight_gfp*mark_gfp
    return final_score
    #
#

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
#
    
def grab_image(is_gfp): # Composed function to acquire an image by activating the light
    color='white'
    duration=3
    
    time_flag_ini=time.time()
    switch_light(color, duration)   # switch on light
    PATH_FILE_WHITE, camera_image_white=acquire_image(color) # take the image
    while (time.time()-time_flag_ini) <= duration:
        time.sleep(1)               # wait until transilluminator is off
    #
    if (is_gfp == True):
        color='blue'
        duration=10  
        
        time_flag_ini=time.time()
        switch_light(color, duration)   # switch on light
        PATH_FILE_GFP, camera_image_gfp=acquire_image(color) # take the image
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
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")) # Load base model
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 3 # n semantic categories +1  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.WEIGHTS=weight

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01  # set a low threshold to get the largest amount of predictions (colonies)
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.01
    cfg.TEST.DETECTIONS_PER_IMAGE = 500
    cfg.MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH = 0.5
    cfg.MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT = 2048 
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.1
    predictor = DefaultPredictor(cfg)
    
    return predictor
    #
# 

def detect_colonies(images_raw,paths): # Function to detect colonies in the image
    image_raw_white=images_raw[0]
    path_white=paths[0]
    # CROP GATHERED PHOTO WITH PLATE AND RETRIEVE ITS CENTER IN IMAGE
    bbox_xywh=[]
    r=0
    margin_shift=0
    cx=0
    cy=0
    center_plate_labware, center_plate_crop, image_crop_white, bbox_xywh_crop, r_crop, margin_shift_crop, cx_plate, cy_plate=crop_photo(image_raw_white, bbox_xywh, r, margin_shift, cx, cy) 
    crop_images=[image_crop_white]
    #Save cropped image
    pos_dot=path_white.find('.')
    crop_path_white=path_white[0:pos_dot]+'_crop'+path_white[pos_dot:len(path_white)]
    cv.imwrite(crop_path_white, image_crop_white)         
    
    # Process GFP image if present
    if len(images_raw) > 1:
        image_raw_gfp=images_raw[1]
        path_gfp=paths[1]
        dummy1, dummy2, image_crop_gfp, bbox_xywh_crop2, r_crop2, margin_shift_crop2, cx_plate2, cy_plate2=crop_photo(image_raw_gfp, bbox_xywh_crop, r_crop, margin_shift_crop, cx_plate, cy_plate)  
        pos_dot=path_gfp.find('.')
        crop_path_gfp=path_gfp[0:pos_dot]+'_gfp_crop'+path_gfp[pos_dot:len(path_gfp)]
        cv.imwrite(crop_path_gfp, image_crop_gfp) 
        crop_images.append(image_crop_gfp)         
        #
    #
    # Turn cropped image into RGB order
    image_crop= cv.cvtColor(image_crop_white, cv.COLOR_BGR2RGB)
    crop_images[0]=image_crop 
    
    binary_predictions_weights=[]

    for weights in weights_paths: # perform the detection using two (or more) different predictions by using different weights

    # STEP 0: LOAD DETECTRON
        predictor=load_detectron(weights)

        # Generate panoptic image
        preds_val=predictor(image_crop)["panoptic_seg"]
        
        # Save predicted mask as *.png
        im_panoptic=np.asarray(preds_val[0])   
        
        # Postprocess predicted image to remove no real colonies (noise) 
        panoptic_pred=copy.deepcopy(im_panoptic) #image is saved as a 3 channel image and we pick the last channel
    
        # Get ids (labels) from predicted panoptic segmentation masks
        labs=np.unique(panoptic_pred)  # the 2 last ids are always the plate and out of plate regions excluding the colonies
    
        # Modify panoptic prediction in order to compute the center coordinates of the image
        filtered_pred=copy.deepcopy(panoptic_pred)
        filtered_pred[filtered_pred==0]=labs[-3]+3 # asign high id value to first predicted element so that it is not 0
        filtered_pred[filtered_pred==labs[-2]]=0 # assign 0 to outside of plate region
    
        dimensions=np.shape(filtered_pred)
        row_range=np.arange(0,dimensions[0])
        column_range=np.arange(0,dimensions[1])
        xgrid,ygrid=np.meshgrid(row_range,column_range)    
        
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
            if areas[i]<=min_size_colony_pixels or areas[i]>=max_size_colony_pixels: # limit size (too small objects are considered noise)
                selected=binary_objects[i].label
                panoptic_binary[label_panoptic_binary==selected]=0
                #
            #
        #
        binary_predictions_weights.append(panoptic_binary)
        #
    #
    # Make the consensus binary mask between infereces delivered by selected weights
    binary_predictions_weights=np.asarray(binary_predictions_weights,dtype=bool) # transform non-continous labeled regions (because we have prefiltered colonyes by size) to True binary value
    panoptic_binary=np.zeros(np.shape(panoptic_pred),dtype=bool)
    shapee=np.shape(panoptic_binary)
    for i in range(0,shapee[0]):
        for j in range(0,shapee[1]):
            panoptic_binary[i,j]=binary_predictions_weights[0,i,j] and binary_predictions_weights[1,i,j] # this is for two set of weights, change the conditional to extend the consensus to more weight inferences
            #
        #
    #
    
    # Label filtered binary mask and assign a different label to each colony (object)
    panoptic_completed=np.zeros(np.shape(panoptic_pred))
       
    panoptic_binary_labeled=label(panoptic_binary)
    binary_objects_labeled=regionprops(panoptic_binary_labeled)

    slicee=[] # obtain pixels contained into object bounding box
    areas=[]     
    max_label=0
    for obj in binary_objects_labeled:
        lab=obj.label
        if lab>max_label:
            max_label=int(lab)
        #
        pix=obj.coords
        areas.append(obj.area)
        slicee.append(obj.slice)        
        for elem in pix:
            panoptic_completed[elem[0],elem[1]]=lab
            #
        #
    #
    
    # Identify and split merged colonies in large groups using wathershed and local peak maxima algorithms
    panoptic_completed,number_of_colonies=discriminate_colony_groups(panoptic_binary,panoptic_completed,image_crop,panoptic_binary_labeled,binary_objects_labeled,areas,slicee,max_label,xgrid,ygrid)
    
    # Reassign labels and compute colony segment properties
    dummy_binary=panoptic_completed>0
    panoptic_labeled=label(panoptic_completed) 
    
    info_objects_sel=regionprops(panoptic_labeled)    
    
    # Convert panoptic labeled image into rgb visual matrix to deliver a coloured image
    panoptic_rgb=label2rgb(dummy_binary) 
    panoptic_rgb_colors=label2rgb(panoptic_labeled) # Colors in this function are repeated, so the background and some colonies are displayed in the same color and therefore those seem to be missing
    #plt.imshow(panoptic_completed, cmap="gray")
    plt.imshow(panoptic_rgb, cmap="gray")
    plt.show()
    plt.imshow(panoptic_rgb_colors, cmap="gray")
    plt.show()
    
    pos_crop=crop_path_white.find('crop')    
    panoptic_path=crop_path_white[0:pos_crop]+'_prediction'+'.png'
    panoptic_rgb_path=crop_path_white[0:pos_crop]+'_rgb_prediction'+'.png'

    # Save both panoptic labeled and panoptic rgb_image
    cv.imwrite(panoptic_path, panoptic_labeled)
    cv.imwrite(panoptic_rgb_path, panoptic_rgb)

    return panoptic_labeled, info_objects_sel, crop_images, number_of_colonies, center_plate_labware, center_plate_crop, xgrid, ygrid     
    #
#

def colony_selection(crop_images, panoptic_labeled, info_objects_sel, filter_criterium, n_colonies_to_pick): # Function to select the colonies to be picked from the whole set detected in the plate 
    if 'counting' in filter_criterium:
        only_counting=True
        return only_counting
    else:
        sorted_list_size=[]
        sorted_list_color=[]
        sorted_list_gfp=[]
        
        # Get number of labels
        labels=np.unique(panoptic_labeled)
        labels=labels[1:len(labels)]
        
        # Filter by selected criterium/s
        if 'size' in filter_criterium: # if size is in the filter criterium, we need to sort the colonies by size in descending (default) or ascending order
            sorted_list_size=size_filter(max_size_colony_pixels, min_size_colony_pixels, panoptic_labeled, info_objects_sel)
            max_mark_size=np.max(sorted_list_size[:,2])
        if 'color' in filter_criterium: # if color is in the filter criterium, we need to evaluate how close is each colony to the reference color. Then sort the colonies by color similarity in ascending or descending (default) order (smaller score means more similarity in color)
            sorted_list_color=color_filter_lacz(crop_images[0], panoptic_labeled, info_objects_sel, xgrid, ygrid, target_reference_color_RGB) 
            max_mark_color=np.max(sorted_list_color[:,2])
        if 'gfp' in filter_criterium: # if gfp is in the filter criterium, we need to sort the colonies by light intensity in descending (default) or ascending order
            sorted_list_gfp=gfp_filter(crop_images[1], panoptic_labeled)
            max_mark_gfp=np.max(sorted_list_gfp[:,2])  # marks in every criterium are normalized to their values before weighing 
            #
        #
        
        score_list=[]
        
        for lab in labels:
            if 'size' in filter_criterium:
                pos=np.where(sorted_list_size[:,0]==lab)
                mark_size=(sorted_list_size[pos[0][0],2])/max_mark_size 
            else:
                mark_size=0
            #
            if 'color' in filter_criterium:
                aux=np.asarray(sorted_list_color[:,0],dtype=int)
                pos=np.where(aux==lab)
                mark_color=(sorted_list_color[pos[0][0],2])/max_mark_color
            else:
                mark_color=0
            #
            if 'gfp' in filter_criterium:
                pos=np.where(sorted_list_gfp[:,0]==lab)
                mark_gfp=(sorted_list_gfp[pos[0][0],2])/max_mark_gfp 
            else:
                mark_gfp=0
            #
            final_score= evaluate_colony(mark_size,mark_color,mark_gfp)   
            score_list.append((lab, mark_size, mark_color, mark_gfp, final_score))
            #
        #
        dtype = [('label', np.int64), ('mark_size', float),('mark_color', float),('mark_gfp', float),('total_score', float)]
        score_list = np.array(score_list, dtype=dtype)
        score_list=np.sort(score_list, order='total_score')        
        score_list_sorted = score_list[::-1]
        
        if 'gfp' in filter_criterium:
            score_list_sorted = score_list[::1] # Reverse order to select the brightest colonies
            #
        #

        # Truncate list according to maximum number of desired colonies
        score_list_sorted=score_list_sorted[0:n_colonies_to_pick]
        
        return score_list_sorted, sorted_list_color
    #
#

def create_csv(info_objects_sel,score_list_sorted, center_plate_labware,center_plate_crop,transillum_settings,mx_fit_px_to_mm,bx_fit_px_to_mm,my_fit_px_to_mm,by_fit_px_to_mm):   # Function to create a csv file with the robot coordinates of the selected colonies
    labels=[] # get objects ids
    centroids=[] # get objects centroids
    areas2=[] # get objects areas
    
    for i in range(0, len(score_list_sorted)):
        pos=score_list_sorted[i][0]-1
        areas2.append(info_objects_sel[pos].area)
        labels.append(info_objects_sel[pos].label)
        centroids.append(info_objects_sel[pos].centroid) 
        #
    #
        
    # Compute offset position of colonies respect transilluminator (labware) center  

    offset_labware_to_plate_x=(center_plate_labware[0]-transillum_settings['CX_labware'])*mx_fit_px_to_mm+bx_fit_px_to_mm
    offset_labware_to_plate_y=(center_plate_labware[1]-transillum_settings['CY_labware'])*my_fit_px_to_mm+by_fit_px_to_mm 

    cart_centroids=[] 
    for i in range(0, len(centroids)):
        
        # Get get incremental offset vector (in pixels) from detected colonies to detected center in crop image. As crop size is a subset of pixels of global image, pixel lengths match in both images (crop and original)
        offset_colony_to_center_crop_x=(centroids[i][0]-center_plate_crop[0])*mx_fit_px_to_mm+bx_fit_px_to_mm # Incremental shift between colony position and plate center 
        offset_colony_to_center_crop_y=(centroids[i][1]-center_plate_crop[1])*my_fit_px_to_mm+by_fit_px_to_mm # Incremental shift between colony position and plate center
        
        # Get incremental offset pointing to each colony from labware center
        colony_robot_shift_x = offset_labware_to_plate_x + offset_colony_to_center_crop_x
        colony_robot_shift_y = offset_labware_to_plate_y + offset_colony_to_center_crop_y
        
        # Annotate obtained X and Y incremental coordinates taking into account that python image frame and OT-2 coordinate frame are rotated pi/2 radians one each other  
        cart_centroids.append((colony_robot_shift_y, -colony_robot_shift_x)) # change x by y and invert sign of y to match image coord system with robot coord system
        #
    #    
    # Write csv file
    header = ['label', 'area', 'colony_robot_shift_x', 'colony_robot_shift_y']

    with open(camera_settings['IMG_PATH']+ 'colony_list_plate_0'+'.csv', 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f)
    
        # Write the header
        writer.writerow(header)
    
        # Write the data
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
panoptic_labeled, info_objects_sel, crop_images, number_of_colonies, center_plate_labware, center_plate_crop, xgrid, ygrid=detect_colonies(image_layers,path_layers)
print('TOTAL NUMBER OF DETECTED COLONIES = ' + str(number_of_colonies))

# STEP 5: APPLY SELECTION CRITERIA
score_list_sorted, sorted_list_color = colony_selection(crop_images, panoptic_labeled, info_objects_sel, filter_criterium, n_colonies_to_pick)

# STEP 6: GENERATE OUTPUT CSV FILE
create_csv(info_objects_sel,score_list_sorted, center_plate_labware,center_plate_crop,transillum_settings,mx_fit_px_to_mm,bx_fit_px_to_mm,my_fit_px_to_mm,by_fit_px_to_mm)

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
#
