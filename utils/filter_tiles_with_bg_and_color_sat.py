import datetime
import gc
import glob
import multiprocessing as mproc
import os
import random
import shutil
import time
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple, Union
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvips
from joblib import Parallel, delayed
from PIL import Image
from skimage.exposure import match_histograms
from tqdm import tqdm
from tqdm.auto import tqdm


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
    black_tot, white_tot = np.sum(black_bg_mask), np.sum(white_bg_mask)

    black_ratio = black_tot / np.prod(black_bg_mask.shape)
    white_ratio = white_tot / np.prod(black_bg_mask.shape)
    info_ratio = 1 - (black_tot + white_tot) / np.prod(black_bg_mask.shape)

    return {
        "info_ratio": info_ratio,
        "black_ratio": black_ratio,
        "white_ratio": white_ratio,
    }

def is_valid(img, white_thr, info_thr, color_thr):
    info = get_information_ratio(img, white_thr=white_thr)
    info_ratio = info["info_ratio"]
    black_ratio = info["black_ratio"]
    white_ratio = info["white_ratio"]
    color_saturation = calculate_saturation(img)

    if black_ratio > 0.1:
        return False

    if info_ratio > info_thr and color_saturation > color_thr:
        return True

    return False



def filter_images(idx, img_path, save_path, discard_path, white_thr, info_thr, color_thr):
    img = cv2.imread(img_path)
    if is_valid(img, white_thr, info_thr, color_thr):
        shutil.copy(img_path, save_path)
    else:
        shutil.copy(img_path, discard_path)


def wrapper_filter_images(args):
    return filter_images(*args)


def apply_validation(white_thr, info_thr, color_thr):
    image_paths = glob(
        "/kaggle/working/UBC-OCEAN-root/wsi_tiles_no_mask_1024x1024_stride512_2/train_images/*/*.jpg"
    )

    version = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    root_dir = f"/kaggle/working/UBC-OCEAN-images/wsi_tiles_no_mask_1024x1024_stride512_2_filtered/{version}"

    select_dir = f"{root_dir}/selected"
    discard_dir = f"{root_dir}/discarded"

    os.makedirs(select_dir, exist_ok=True)
    os.makedirs(discard_dir, exist_ok=True)

    combs = []

    for image_path in tqdm(image_paths, total=len(image_paths), desc="make combs"):
        image_id = Path(image_path).parent.stem
        image_name = Path(image_path).name

        cur_select_dir = os.path.join(select_dir, image_id)
        cur_discard_dir = os.path.join(discard_dir, image_id)
        os.makedirs(cur_select_dir, exist_ok=True)
        os.makedirs(cur_discard_dir, exist_ok=True)

        cur_select_path = os.path.join(cur_select_dir, image_name)
        cur_discard_path = os.path.join(cur_discard_dir, image_name)
        combs.append((image_path, cur_select_path, cur_discard_path))

        # if is_validate_tile_without_mask(cv2.imread(image_path), bg_thr, white_threshold):
        #     shutil.copy(image_path, cur_select_path)
        # else:
        #     shutil.copy(image_path, cur_discard_path)

    num_threads = 16
    print(f"found images: {len(combs)}")
    pool = mproc.Pool(num_threads)
    tqdm_bar = tqdm(total=len(combs), desc="filtering images")

    for _ in pool.imap_unordered(
        wrapper_filter_images,
        (
            (idx, img_path, save_path, discard_path, white_thr, info_thr, color_thr)
            for idx, (img_path, save_path, discard_path) in enumerate(combs)
        ),
    ):
        tqdm_bar.update()
    pool.close()
    pool.join()


def run_apply_validation():
    apply_validation(
        white_thr=225,
        info_thr=0.5,
        color_thr=8,
    )


def eda_result():
    ori_df = pd.read_csv("/kaggle/working/UBC-OCEAN/train.csv")
    image_id2label = {str(image_id): label for image_id, label in zip(ori_df['image_id'], ori_df['label'])}

    discarded_image_paths = glob("/kaggle/working/UBC-OCEAN-images/select_or_not/20231206_000112/discarded/*/*.jpg")

    label2cnt = defaultdict(int)
    for discarded_image_path in tqdm(discarded_image_paths, total=len(discarded_image_paths)):
        image_id = Path(discarded_image_path).parent.stem
        label = image_id2label[image_id]
        label2cnt[label] += 1
        # shutil.copy(discarded_image_path, f"/kaggle/working/UBC-OCEAN-images/select_or_not/20231206_000112/discarded/{label}")

    print(label2cnt)


# eda_result()
run_apply_validation()

