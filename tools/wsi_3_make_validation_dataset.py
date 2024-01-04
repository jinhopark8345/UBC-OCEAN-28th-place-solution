import os
import pandas as pd
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

import gc
import glob
import multiprocessing as mproc
import os
import random
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyvips
from joblib import Parallel, delayed
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

class TMAConfig:
    # TMA model settings
    tma_model_name = 'maxvit_tiny_tf_512.in1k'
    tma_model_weight = ""
    tma_labels = ["CC", "EC", "HGSC", "LGSC", "MC", "Other"]
    num_classes = len(tma_labels)
    tma_label2idx = {label: idx for idx, label in enumerate(tma_labels)}

    tma_center_crop_size = 2560
    tma_tile_size = 2048
    tma_tile_scale = 0.25
    # tma_tile_stride = 512
    tma_tile_stride = 256
    tma_tile_drop_thr = 1.0
    tma_tile_white_thr = 240

    tma_decision_rule = 'vote' # vote, sum

class WSIConfig:
    # WSI model settings
    tumor_classifier = "tf_efficientnetv2_s_in21ft1k"
    tumor_classifier_model_weight = ""
    tumor_classifier_classes = ["tumor", "normal"]
    tumors_classifier_num_classes = len(tumor_classifier_classes)

    tumor_subtype_classifier = 'maxvit_tiny_tf_512.in1k'
    tumor_subtype_classifier_model_weight = ""
    tumor_subtype_classifier_classes = ["CC", "EC", "HGSC", "LGSC", "MC", "Other"]
    tumor_subtype_classifier_num_classes = len(tumor_subtype_classifier_classes)
    tumor_subtype_class2idx = {label: idx for idx, label in enumerate(tumor_subtype_classifier_classes)}

    # first phase varialbes
    thumbnail_tile_size = 200
    thumbnail_tile_stride = 200
    thumbnail_tile_scale = 1.0
    thumbnail_tile_white_thr = 240
    thumbnail_tile_black_bg_thr = 0.8
    thumbnail_tile_white_bg_thr = 0.8

    # second phase variables
    tumor_tile_size = 1024
    tumor_tile_scale = 0.5
    tumor_tile_white_thr = 227
    tumor_tile_info_cut_thr = 0.4
    tumor_tile_info_save_thr = 0.7
    tumor_tile_color_cut_thr = 7
    tumor_tile_color_save_thr = 12

    tumor_tile_full_search_thr = 5

class Config:
    debug = True
    set_is_tma_rule = 'by_image_size' # 'by_image_size', 'by_thumbnail_folder'
    mode = "test"
    run = 'wsi' # 'tma', 'wsi', 'tma_wsi'

    dataset_dir = "/kaggle/input/UBC-OCEAN/"
    image_dir = os.path.join(dataset_dir, f"{mode}_images")
    thumbnail_dir = os.path.join(dataset_dir, f"{mode}_thumbnails")

    tile_folder = "./tmp_tiles"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    num_workers = 4

    tma_config = TMAConfig
    wsi_config = WSIConfig

if Config.debug:
    print("-------------------------- DEBUG MODE IS ON --------------------------")
    Config.mode = "train"
    Config.image_dir = os.path.join(Config.dataset_dir, f"{Config.mode}_images")
    Config.thumbnail_dir = os.path.join(
        Config.dataset_dir, f"{Config.mode}_thumbnails"
    )



wsi_config = Config.wsi_config

def calculate_saturation(image):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate the saturation channel
    saturation_channel = hsv_image[:, :, 1]

    # Calculate the average saturation value
    average_saturation = np.mean(saturation_channel)

    return average_saturation


def get_information_ratio(tile, white_thr):
    black_bg_mask = np.sum(tile, axis=2) == 0
    white_bg_mask = np.mean(tile, axis=2) > white_thr
    background_tot = np.sum(black_bg_mask) + np.sum(white_bg_mask)

    return 1 - background_tot / np.prod(black_bg_mask.shape)


def is_valid_tumor_tile(img, white_thr, info_cut_thr, info_save_thr, color_cut_thr, color_save_thr):
    info_ratio = get_information_ratio(img, white_thr=white_thr)
    color_saturation = calculate_saturation(img)

    if info_ratio >= info_save_thr or color_saturation >= color_save_thr:
        return True

    if info_ratio < info_cut_thr:
        return False

    if color_saturation < color_cut_thr:
        return False

    return True

def extract_tiles_from_tumor_tiles(
    wsi_image_path: str,
    tumor_tile_info_list: str,
    tile_size: int = 1024,
    tile_scale: float = 0.5,
    tile_full_search_thr: int = 5,
    tile_white_thr: int = 227,
    tile_info_cut_thr: float = 0.4,
    tile_info_save_thr: float = 0.7,
    tile_color_cut_thr: int = 7,
    tile_color_save_thr: int = 12,
):
    img = pyvips.Image.new_from_file(wsi_image_path)
    new_size = (int(tile_size * tile_scale), int(tile_size * tile_scale))
    img_width, img_height = img.width, img.height
    tile_width = tile_height = tile_size
    images = []

    if img_width < tile_size or img_height < tile_size:
        cur_tile = Image.open(img_path).resize(new_size, Image.LANCZOS)
        images.append(cur_tile)
        return images

    idxs = []

    for tumor_tile_info in tumor_tile_info_list:
        x, y, w, h = tumor_tile_info

        # if tile is smaller than tile size, then just use the whole tile
        if w < tile_width or h < tile_height:
            idxs.append((y, y+tile_height, x, x + tile_width))
            continue

        # if there are less or equal threshold_val tumor tiles, then use all tiles from it
        if len(tumor_tile_info_list) <= tile_full_search_thr:
            tumor_idxs = [
                (cy, cy + tile_height, cx, cx + tile_width)
                for cy in range(
                    y, y+h - tile_height + 1, tile_size
                )  # Include stride in the y range
                for cx in range(
                    x, x+w - tile_width + 1, tile_size
                )  # Include stride in the x range
            ]
            idxs.extend(tumor_idxs)

        # if there are more than threshold_val tumor tiles, then use center tile only from each tumor tile
        else:
            # otherwise, center crop the tumor tile
            left_margin, top_margin = (w - tile_width) // 2, (h - tile_height) // 2
            center_y = y + top_margin
            center_x = x + left_margin

            if center_y + tile_height > img_height or center_x + tile_width > img_width:
                idxs.append((y, y+tile_height, x, x + tile_width))
                continue

            idxs.append((center_y, center_y+tile_height, center_x, center_x + tile_width))


    for y, y_, x, x_ in idxs:
        if x + tile_width > img_width or y + tile_height > img_height:
            continue

        try:
            tile = img.crop(x, y, tile_width, tile_height).numpy()[..., :3]
            if tile.shape[:2] != (tile_height, tile_width):
                tile_ = tile
                tile_size = (
                    (tile_height, tile_width)
                    if tile.ndim == 2
                    else (tile_height, tile_width, tile.shape[2])
                )
                tile = np.zeros(tile_size, dtype=tile.dtype)
                tile[: tile_.shape[0], : tile_.shape[1], ...] = tile_

            new_size = (int(tile_size * tile_scale), int(tile_size * tile_scale))
            resized_tile = cv2.resize(tile, new_size, interpolation=cv2.INTER_LANCZOS4)

            if is_valid_tumor_tile(
                img=resized_tile,
                white_thr=tile_white_thr,
                info_cut_thr=tile_info_cut_thr,
                info_save_thr=tile_info_save_thr,
                color_cut_thr=tile_color_cut_thr,
                color_save_thr=tile_color_save_thr
            ):
                images.append(Image.fromarray(resized_tile))
        except:
            print(f"extract_tiles_from_tumor_tiles, Error in extracting tile from {wsi_image_path, img_width, img_height} at {x=}, {y=}, {tile_width=}, {tile_height=}")

    return images


def extract_func(row, image_id2tumor_tile_info_list, root_dir):
    image_id = str(row["image_id"])
    print(f'processing.. {image_id = }')
    image_width = row["image_width"]
    image_height = row["image_height"]
    wsi_thumbnail_image_path = os.path.join(Config.thumbnail_dir, f"{image_id}_thumbnail.png") # use thumbnail image for WSI
    wsi_image_path = os.path.join(Config.image_dir, f"{image_id}.png")
    thumbnail_width, thumbnail_height = Image.open(wsi_thumbnail_image_path).size

    cur_image_dir = os.path.join(root_dir, image_id)
    os.makedirs(cur_image_dir, exist_ok=True)

    if image_id not in image_id2tumor_tile_info_list:
        return

    tumor_tile_info_list = image_id2tumor_tile_info_list[image_id]

    ####################### 2nd phase ##############################
    wsi_tumor_tiles = extract_tiles_from_tumor_tiles(
        wsi_image_path=wsi_image_path,
        tumor_tile_info_list=tumor_tile_info_list,
        tile_size=wsi_config.tumor_tile_size,
        tile_scale=wsi_config.tumor_tile_scale,
        tile_full_search_thr=wsi_config.tumor_tile_full_search_thr,
        tile_white_thr=wsi_config.tumor_tile_white_thr,
        tile_info_cut_thr=wsi_config.tumor_tile_info_cut_thr,
        tile_info_save_thr=wsi_config.tumor_tile_info_save_thr,
        tile_color_cut_thr=wsi_config.tumor_tile_color_cut_thr,
        tile_color_save_thr=wsi_config.tumor_tile_color_save_thr,
    )

    for idx, tmp_image in enumerate(wsi_tumor_tiles):
        tmp_image.save(os.path.join(cur_image_dir, f"{idx}.jpg"))

    print(f'processing done {image_id = }')

def wrapper_func(args):
    return extract_func(*args)

def inference_wsi(df: pd.DataFrame):

    # need to load it from somewhere
    tmp_image_id2tumor_tile_info_list = {}
    import pickle
    with open("image_id2tumor_tile_info_list_v3.pkl", "rb") as f:
        tmp_image_id2tumor_tile_info_list = pickle.load(f)

    df = df[df["is_tma"] == False]
    df = df[200:]

    root_dir = "tumor_tile_testset_save2/v3"
    os.makedirs(root_dir, exist_ok=True)
    num_threads = 12

    print(f"found rows: {len(df)}")
    pool = mproc.Pool(num_threads)
    tqdm_bar = tqdm(total=len(df))
    for _ in pool.imap_unordered(
        wrapper_func,
        (
            (row, tmp_image_id2tumor_tile_info_list, root_dir)
            for idx, row in df.iterrows()
        ),
    ):
        tqdm_bar.update()
    pool.close()
    pool.join()

def setup_df(default_label="Something"):
    def set_is_tma(image_dir, thumbnail_dir, df_row):
        image_id = df_row["image_id"]
        image_width = df_row["image_width"]
        image_height = df_row["image_height"]

        if Config.set_is_tma_rule == 'by_thumbnail_folder':
            if os.path.exists(
                f"{thumbnail_dir}/{image_id}_thumbnail.png"
            ):  # WSI image has thumbnail file as well
                return False
            else:  # TMA image doesn't have thumbnail file -> no thumbnail file exist -> it is TMA image
                return True
        elif Config.set_is_tma_rule == 'by_image_size':
            if image_width > 5000 and image_height > 5000:
                return False
                # if os.path.exists(img_path) and not os.path.exists(thumbnail_img_path):  # handle WSI image
            else:  # handle TMA image
                return True
        else:
            raise ValueError(f"Invalid Config.set_is_tma_rule: {Config.set_is_tma_rule}")

    # prepare test dataframe
    df_test = pd.read_csv(os.path.join(Config.dataset_dir, f"{Config.mode}.csv"))
    # default label
    df_test["label"] = [default_label] * len(df_test) # TODO : need to chagne?

    print(f"Dataset/test size: {len(df_test)}")
    print("df_test.head(): ")
    print(df_test.head())

    df_test["is_tma"] = df_test.apply(lambda row: set_is_tma(Config.image_dir, Config.thumbnail_dir, row), axis=1)

    # Replace 'test_thumbnails' with the actual path to your folder
    return df_test


# setup dataframe
df_test = setup_df()

# # inference and make submission
# if 'tma' in Config.run:
#     inference_tma(df_test)

# if 'wsi' in Config.run:
#     inference_wsi(df_test)

