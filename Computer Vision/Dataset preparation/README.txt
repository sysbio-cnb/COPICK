**Dataset preparation**

# Files description to prepare the Colony Picker (COPICK) custom dataset for Panoptic Segmentation tasks.


Folder "json and csv files" contains:

	-json_template.json: generic template to create panoptic annotations json file (including panoptic segments: both things and stuffs) and instances (things) annotations json file. 
	
	-Colony_picker_database.csv: csv file with significant data of the dataset (images names are generated with function file_namer.py). 


		
The rest of files are *py.files regarding different functions to be used in the overall workflow (all of them are located in the same folder for a better use).


IMAGE PROCESSING AND MASKS GENERATION

	-file_namer.py: it is used during the dataset augmentation step (augmentation_stage.py) and generates random names for the augmented images based on the reference image.
 
	-augmentation_stage.py: it is used for dataset augmentation and creates 15 transformed copies based on the reference image.

	-image_pretreatment.py: it is used for image pretreatment and first masks generation to be used for posterior segmentation techniques.

	-segmentation_stage.py: it is used to create panoptic masks from images (unique ids and colors per segment in the image). Uses data from "Colony_Picker_database.csv".
	


DATASET PROCESSING

Previous functions of image processing and masks generation are used inside the next codes where the whole dataset is processed and necessary masks and json files are generated and saved.

	-PROCESS_DATASET_PANOPTIC_SEGMENTATION.py: it is the whole dataset process workflow to obtain panoptic masks and json files following COCO dataset and Detectron2 format --> panoptic_train.json and panoptic_val.json in Datasets/Complete dataset/annotations/.	

	-PROCESS_DATASET_INSTANCES_SEGMENTATION.py: it is used to process panoptic masks and obtain info from segments categorized as things, transforming that into instances or things annotations in a json file --> instances_train.json and instances_val.json.


	#Modified functions from Detectron2 -prepare_panoptic_fpn.py- script:

	-process_panoptic_to_semantic.py: it is used inside the function used to obtain semantic masks (stuffs) from panoptic masks (separate_panoptic_from_semtantic.py).

	-separate_panoptic_from_semtantic.py: it is used to obtain semantic masks (ids=0 (things); ids≠0 ranging from 1 to number of stuff segments in the image (stuffs)). In the COPICK dataset there are just two stuff categories and segments referring to: "out of plate" (1) and "plate" (2) regions. This function is called inside -process_dataset_panoptic_segmentation.py- script after getting panoptic masks. 




DATASET MODIFICATIONS

	-im_dimensions.py: it is used to store cropped images after processing based on their dimensions. It checks cropped images dimensions and compare them with their corresponding panoptic masks dimensions before storing them in a specified folder. This function is partially based on -image_pretreatment- function to crop original images. As the dataset is already cropped and augmentated, this is only an additional script in case of interest when preparing a custom dataset. 



