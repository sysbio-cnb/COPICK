# -*- coding: utf-8 -*-

# separate_semantic_from_panoptic.py - Modified Detectron2 Python script (-prepare_panoptic_fpn.py- in ...detectron2/datasets/) to obtain semantic annotations (masks) from panoptic masks in the dataset. 
#                                      See: https://detectron2.readthedocs.io/en/latest/tutorials/builtin_datasets.html (Expected dataset structure for PanopticFPN).

# GENERAL DESCRIPTION:
# -------------------
# When the script is executed, it processes train or val datasets to obtain semantic masks (relative to stuffs categories in the dataset: regions NOT corresponding to colonies in the image) from respective panoptic masks. Detailed inside the function down below.

# The function -_process_panoptic_to_semantic- is separated and modified in other file and called inside -separate_semantic_from_panoptic- modified function down below.

# INPUTS (also described inside the function down below):
#     - panoptic_json (str): path to the panoptic json file, in COCO's format. #contains info per image. Each image is an annotation from panoptic_root
#     - panoptic_root (str): a directory with panoptic annotation files, in COCO's format. #annotation files are PNG images representing class-agnostic image segmentation: divided in segments
#     - sem_seg_root (str): a directory to output semantic annotation files.
#     - categories (list[dict]): category metadata. Each dict needs to have:
#         "id": corresponds to the "category_id" in the json annotations
#         "isthing": 0 or 1

# OUTPUTS: NONE (Semantic masks are stored in specified path)

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


# Import required libraries
import functools
import json
import multiprocessing as mp
#import numpy as np
import os
import time
# from fvcore.common.download import download
# from panopticapi.utils import rgb2id
#from PIL import Image
from process_panoptic_to_semantic import process_panoptic_to_semantic
# from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES



# Function modified and separated in other file
# def _process_panoptic_to_semantic(input_panoptic, output_semantic, segments, id_map):
    # ...
#


def separate_semantic_from_panoptic(panoptic_json, panoptic_root, sem_seg_root, categories):
    """
    Create semantic segmentation annotations from panoptic segmentation
    annotations, to be used by PanopticFPN.

    It maps all thing categories to class 0, and maps all unlabeled pixels to class 255.
    It maps all stuff categories to contiguous ids starting from 1.
    
    Args:
        panoptic_json (str): path to the panoptic json file, in COCO's format. #contains info per image. Each image is an annotation from panoptic_root
        panoptic_root (str): a directory with panoptic annotation files, in COCO's format. #annotation files are PNG images representing class-agnostic image segmentation: divided in segments
        sem_seg_root (str): a directory to output semantic annotation files.
        categories (list[dict]): category metadata. Each dict needs to have:
            "id": corresponds to the "category_id" in the json annotations
            "isthing": 0 or 1
    """
    os.makedirs(sem_seg_root, exist_ok=True)

    stuff_ids = [k["id"] for k in categories if k["isthing"] == 0]
    thing_ids = [k["id"] for k in categories if k["isthing"] == 1]
    id_map = {}  # map from category id to id in the output semantic annotation
    assert len(stuff_ids) <= 254
    for i, stuff_id in enumerate(stuff_ids):
        id_map[stuff_id] = i + 1
    for thing_id in thing_ids:
        id_map[thing_id] = 0
        
    #id_map[0] = 255 ## adds new field to id_map so that labels with category_id=0 are set with id values of 255 and therefore are considered as void and ignored during training
        # In our case category_id=0 does not exist in the dataset 
        # No unlabeled pixels in panoptic images. This line maps elements with id_category=0 to class 255 (in COCO dataset id_category=0 is probably assigned to unlabeled pixels)
        #
    #
    with open(panoptic_json) as f:
        obj = json.load(f)
        #
    #
    pool = mp.Pool(processes=max(mp.cpu_count() // 2, 4))

    def iter_annotations():
        for anno in obj["annotations"]:
            file_name = anno["file_name"]
            segments = anno["segments_info"]
            input = os.path.join(panoptic_root, file_name)
            output = os.path.join(sem_seg_root, file_name)
            yield input, output, segments
            #
        #
    #
    print("Start writing to {} ...".format(sem_seg_root))
    start = time.time()
    pool.starmap(
        functools.partial(process_panoptic_to_semantic, id_map=id_map, sem_seg_root=sem_seg_root),
        iter_annotations(),
        chunksize=100,
    )
    print("Finished. time: {:.2f}s".format(time.time() - start))
    #
#



## Previous function version by Detectron2 

# def separate_coco_semantic_from_panoptic(panoptic_json, panoptic_root, sem_seg_root, categories):
#     """
#     Create semantic segmentation annotations from panoptic segmentation
#     annotations, to be used by PanopticFPN.

#     It maps all thing categories to class 0, and maps all unlabeled pixels to class 255.
#     It maps all stuff categories to contiguous ids starting from 1.
    
#     Args:
#         panoptic_json (str): path to the panoptic json file, in COCO's format. #contains info per image. Each image is an annotation from panoptic_root
#         panoptic_root (str): a directory with panoptic annotation files, in COCO's format. #annotation files are PNG images representing class-agnostic image segmentation: divided in segments
#         sem_seg_root (str): a directory to output semantic annotation files.
#         categories (list[dict]): category metadata. Each dict needs to have:
#             "id": corresponds to the "category_id" in the json annotations
#             "isthing": 0 or 1
#     """
#     os.makedirs(sem_seg_root, exist_ok=True)

#     stuff_ids = [k["id"] for k in categories if k["isthing"] == 0]
#     thing_ids = [k["id"] for k in categories if k["isthing"] == 1]
#     id_map = {}  # map from category id to id in the output semantic annotation
#     assert len(stuff_ids) <= 254
#     for i, stuff_id in enumerate(stuff_ids):
#         id_map[stuff_id] = i + 1
#     for thing_id in thing_ids:
#         id_map[thing_id] = 0
#     id_map[0] = 255

#     with open(panoptic_json) as f:
#         obj = json.load(f)

#     pool = mp.Pool(processes=max(mp.cpu_count() // 2, 4))

#     def iter_annotations():
#         for anno in obj["annotations"]:
#             file_name = anno["file_name"]
#             segments = anno["segments_info"]
#             input = os.path.join(panoptic_root, file_name)
#             output = os.path.join(sem_seg_root, file_name)
#             yield input, output, segments

#     print("Start writing to {} ...".format(sem_seg_root))
#     start = time.time()
#     pool.starmap(
#         functools.partial(_process_panoptic_to_semantic, id_map=id_map),
#         iter_annotations(),
#         chunksize=100,
#     )
#     print("Finished. time: {:.2f}s".format(time.time() - start))


# if __name__ == "__main__":
#     dataset_dir = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "coco")
#     for s in ["val2017", "train2017"]:
#         separate_coco_semantic_from_panoptic(
#             os.path.join(dataset_dir, "annotations/panoptic_{}.json".format(s)),
#             os.path.join(dataset_dir, "panoptic_{}".format(s)),
#             os.path.join(dataset_dir, "panoptic_stuff_{}".format(s)),
#             COCO_CATEGORIES,
#         )

#     # Prepare val2017_100 for quick testing:

#     dest_dir = os.path.join(dataset_dir, "annotations/")
#     URL_PREFIX = "https://dl.fbaipublicfiles.com/detectron2/"
#     download(URL_PREFIX + "annotations/coco/panoptic_val2017_100.json", dest_dir)
#     with open(os.path.join(dest_dir, "panoptic_val2017_100.json")) as f:
#         obj = json.load(f)

#     def link_val100(dir_full, dir_100):
#         print("Creating " + dir_100 + " ...")
#         os.makedirs(dir_100, exist_ok=True)
#         for img in obj["images"]:
#             basename = os.path.splitext(img["file_name"])[0]
#             src = os.path.join(dir_full, basename + ".png")
#             dst = os.path.join(dir_100, basename + ".png")
#             src = os.path.relpath(src, start=dir_100)
#             os.symlink(src, dst)

#     link_val100(
#         os.path.join(dataset_dir, "panoptic_val2017"),
#         os.path.join(dataset_dir, "panoptic_val2017_100"),
#     )

#     link_val100(
#         os.path.join(dataset_dir, "panoptic_stuff_val2017"),
#         os.path.join(dataset_dir, "panoptic_stuff_val2017_100"),
#     )
