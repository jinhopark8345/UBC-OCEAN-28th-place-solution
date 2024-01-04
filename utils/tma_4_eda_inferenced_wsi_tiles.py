import gc
import glob
import multiprocessing as mproc
import os
import pickle
import random
import time
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import pyvips
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm
from tqdm.auto import tqdm
import shutil

pd.set_option("display.max_colwidth", None)


thumbnail_paths = glob("/kaggle/working/UBC-OCEAN/train_thumbnails/*.png")
image_id2thumbnail_paths = {Path(p).name.split("_")[0]: p for p in thumbnail_paths}
train_image_paths = glob("/kaggle/working/UBC-OCEAN/train_images/*.png")
image_id2train_image_paths = {Path(p).name.split("_")[0].split(".")[0]: p for p in train_image_paths}

def overlay_tiles(image, coords, tile_size, color=(0, 0, 255), alpha=0.3):
    """
    Overlay transparent squares on an image.

    :param image: Original image to overlay squares on.
    :param coords: List of coordinates for the top-left corner of each square.
    :param tile_size: Size of each square (width, height).
    :param color: Color of the overlay (B, G, R).
    :param alpha: Transparency of the overlay.
    :return: Image with overlays.
    """
    overlay = image.copy()
    for coord in coords:
        top_left = tuple(coord)
        bottom_right = (top_left[0] + tile_size[0], top_left[1] + tile_size[1])
        cv2.rectangle(overlay, top_left, bottom_right, color, -1)

    overlayed_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    combined = np.concatenate((image, overlayed_image), axis=1)
    return combined


def visualize_inference_result(
    file_paths, image_id, save_dir, tile_stride=512, tile_size=1024
):
    thumbnail_path = image_id2thumbnail_paths[image_id]
    train_image_path = image_id2train_image_paths[image_id]

    thumbnail = cv2.imread(thumbnail_path)
    thumbnail_height, thumbnail_width = thumbnail.shape[:2]
    train_image = pyvips.Image.new_from_file(train_image_path)

    tile_width = tile_height = tile_size

    width_downsize_ratio = thumbnail_width / train_image.width
    height_downsize_ratio = thumbnail_height / train_image.height

    idxs = [
        (y, y + tile_height, x, x + tile_width)
        for y in range(0, train_image.height - tile_size + 1, tile_stride)
        for x in range(0, train_image.width - tile_size + 1, tile_stride)
    ]

    ori_name2coord = {}
    for k, (y, y_, x, x_) in enumerate(idxs):
        ori_name = f"{k:06}_{int(x_ / tile_width)}-{int(y_ / tile_height)}.jpg"
        coord = (x, y)
        ori_name2coord[ori_name] = coord

    coords = [ori_name2coord[Path(file_path).name] for file_path in file_paths]
    downsized_coords = [
        (int(x * width_downsize_ratio), int(y * height_downsize_ratio))
        for x, y in coords
    ]
    downsized_tile_size = (
        int(tile_size * width_downsize_ratio),
        int(tile_size * height_downsize_ratio),
    )

    # Overlay the tiles
    result_image = overlay_tiles(thumbnail, downsized_coords, downsized_tile_size)

    # Save or display the result
    cv2.imwrite(
        os.path.join(save_dir, image_id + ".jpg"), result_image
    )  # Save the image


def visualize_inference_result_overlap_tiles(
    file_paths, image_id, save_dir, tile_stride=512, tile_size=1024
):
    thumbnail_path = image_id2thumbnail_paths[image_id]
    train_image_path = image_id2train_image_paths[image_id]

    thumbnail = cv2.imread(thumbnail_path)
    thumbnail_height, thumbnail_width = thumbnail.shape[:2]
    train_image = pyvips.Image.new_from_file(train_image_path)

    tile_width = tile_height = tile_size

    width_downsize_ratio = thumbnail_width / train_image.width
    height_downsize_ratio = thumbnail_height / train_image.height

    idxs = [
        (y, y + tile_height, x, x + tile_width)
        for y in range(0, train_image.height - tile_size + 1, tile_stride)
        for x in range(0, train_image.width - tile_size + 1, tile_stride)
    ]

    matrix_height = (
        len(list(range(0, train_image.height - tile_size + 1, tile_stride))) + 1
    )
    matrix_width = (
        len(list(range(0, train_image.width - tile_size + 1, tile_stride))) + 1
    )
    matrix = [[0 for _ in range(matrix_width)] for _ in range(matrix_height)]

    ori_name2coord_idx = {}
    for k, (y, y_, x, x_) in enumerate(idxs):
        ori_name = f"{k:06}_{int(x_ / tile_width)}-{int(y_ / tile_height)}.jpg"
        idx = (x // 512, y // 512)
        ori_name2coord_idx[ori_name] = idx

    downsized_tile_size = (
        int(tile_size * width_downsize_ratio),
        int(tile_size * height_downsize_ratio),
    )

    def coord_idx2coord(coord_idx):
        x, y = coord_idx
        return (x * 512, y * 512)

    def coord_idx2downsized_coord(coord_idx, downsized_tile_size):
        x, y = coord_idx
        return (
            int(x * 512 * width_downsize_ratio),
            int(y * 512 * height_downsize_ratio),
        )

    coord_indices = [
        ori_name2coord_idx[Path(file_path).name] for file_path in file_paths
    ]
    # downsized_coords = [coord_idx2downsized_coord(coord_idx, downsized_tile_size) for coord_idx in coord_indices]

    for coord_idx in coord_indices:
        x, y = coord_idx
        matrix[y][x] += 1
        matrix[y + 1][x] += 1
        matrix[y][x + 1] += 1
        matrix[y + 1][x + 1] += 1

    overlap_coord_indices = [
        (x, y)
        for x in range(matrix_width)
        for y in range(matrix_height)
        if matrix[y][x] == 4
    ]
    downsized_overlap_coords = [
        coord_idx2downsized_coord(coord_idx, downsized_tile_size)
        for coord_idx in overlap_coord_indices
    ]

    # Overlay the tiles
    result_image = overlay_tiles(
        thumbnail, downsized_overlap_coords, downsized_tile_size
    )

    # Save or display the result
    cv2.imwrite(
        os.path.join(save_dir, image_id + ".jpg"), result_image
    )  # Save the image

def visualize_inference_result_overlap_tiles_special(
    file_paths, image_id, save_dir, tile_stride=512, tile_size=1024
):
    thumbnail_path = image_id2thumbnail_paths[image_id]
    train_image_path = image_id2train_image_paths[image_id]

    thumbnail = cv2.imread(thumbnail_path)
    thumbnail_height, thumbnail_width = thumbnail.shape[:2]
    train_image = pyvips.Image.new_from_file(train_image_path)

    tile_width = tile_height = tile_size

    width_downsize_ratio = thumbnail_width / train_image.width
    height_downsize_ratio = thumbnail_height / train_image.height

    idxs = [
        (y, y + tile_height, x, x + tile_width)
        for y in range(0, train_image.height - tile_size + 1, tile_stride)
        for x in range(0, train_image.width - tile_size + 1, tile_stride)
    ]

    matrix_height = (
        len(list(range(0, train_image.height - tile_size + 1, tile_stride))) + 1
    )
    matrix_width = (
        len(list(range(0, train_image.width - tile_size + 1, tile_stride))) + 1
    )
    matrix = [[0 for _ in range(matrix_width)] for _ in range(matrix_height)]

    ori_name2coord_idx = {}
    coord_idx2ori_name = {}
    for k, (y, y_, x, x_) in enumerate(idxs):
        ori_name = f"{k:06}_{int(x_ / tile_width)}-{int(y_ / tile_height)}.jpg"
        coord_idx = (x // 512, y // 512)
        ori_name2coord_idx[ori_name] = coord_idx
        coord_idx2ori_name[coord_idx] = ori_name

    downsized_tile_size = (
        int(tile_size * width_downsize_ratio),
        int(tile_size * height_downsize_ratio),
    )

    def coord_idx2coord(coord_idx):
        x, y = coord_idx
        return (x * 512, y * 512)

    def coord_idx2downsized_coord(coord_idx, downsized_tile_size):
        x, y = coord_idx
        return (
            int(x * 512 * width_downsize_ratio),
            int(y * 512 * height_downsize_ratio),
        )

    coord_indices = [
        ori_name2coord_idx[Path(file_path).name] for file_path in file_paths
    ]
    # downsized_coords = [coord_idx2downsized_coord(coord_idx, downsized_tile_size) for coord_idx in coord_indices]

    for coord_idx in coord_indices:
        x, y = coord_idx
        matrix[y][x] += 1
        matrix[y + 1][x] += 1
        matrix[y][x + 1] += 1
        matrix[y + 1][x + 1] += 1

    # overlap_coord_indices = [
    #     (x, y)
    #     for x in range(matrix_width)
    #     for y in range(matrix_height)
    #     if matrix[y][x] == 4
    # ]

    dys = [-1, -1, -1, 0, 1, 1, 1, 0]
    dxs = [-1, 0, 1, 1, 1, 0, -1, -1]
    # dys = [-1, 0, 1, 0]
    # dxs = [0, 1, 0, -1]


    overlap_special_coord_indices = []
    for y in range(1, matrix_height-1):
        for x in range(1, matrix_width-1):
            if matrix[y][x] != 4: continue

            cnt = 0
            for dy, dx in zip(dys, dxs):
                if matrix[y+dy][x+dx] == 4:
                    cnt += 1
            if cnt > 5:
                overlap_special_coord_indices.append((x, y))


    downsized_overlap_coords = [
        coord_idx2downsized_coord(coord_idx, downsized_tile_size)
        for coord_idx in overlap_special_coord_indices
    ]

    # Overlay the tiles
    result_image = overlay_tiles(
        thumbnail, downsized_overlap_coords, downsized_tile_size
    )

    # Save or display the result
    cv2.imwrite(
        os.path.join(save_dir, image_id + ".jpg"), result_image
    )  # Save the image

def select_tiles_from_inference_result_overlap_tiles_special(
    file_paths, image_id, save_dir, tile_stride=512, tile_size=1024
):
    thumbnail_path = image_id2thumbnail_paths[image_id]
    train_image_path = image_id2train_image_paths[image_id]

    thumbnail = cv2.imread(thumbnail_path)
    thumbnail_height, thumbnail_width = thumbnail.shape[:2]
    train_image = pyvips.Image.new_from_file(train_image_path)

    tile_width = tile_height = tile_size

    width_downsize_ratio = thumbnail_width / train_image.width
    height_downsize_ratio = thumbnail_height / train_image.height

    idxs = [
        (y, y + tile_height, x, x + tile_width)
        for y in range(0, train_image.height - tile_size + 1, tile_stride)
        for x in range(0, train_image.width - tile_size + 1, tile_stride)
    ]

    matrix_height = (
        len(list(range(0, train_image.height - tile_size + 1, tile_stride))) + 1
    )
    matrix_width = (
        len(list(range(0, train_image.width - tile_size + 1, tile_stride))) + 1
    )
    matrix = [[0 for _ in range(matrix_width)] for _ in range(matrix_height)]

    ori_name2coord_idx = {}
    coord_idx2ori_name = {}
    for k, (y, y_, x, x_) in enumerate(idxs):
        ori_name = f"{k:06}_{int(x_ / tile_width)}-{int(y_ / tile_height)}.jpg"
        coord_idx = (x // 512, y // 512)
        ori_name2coord_idx[ori_name] = coord_idx
        coord_idx2ori_name[coord_idx] = ori_name

    downsized_tile_size = (
        int(tile_size * width_downsize_ratio),
        int(tile_size * height_downsize_ratio),
    )

    def coord_idx2coord(coord_idx):
        x, y = coord_idx
        return (x * 512, y * 512)

    def coord_idx2downsized_coord(coord_idx, downsized_tile_size):
        x, y = coord_idx
        return (
            int(x * 512 * width_downsize_ratio),
            int(y * 512 * height_downsize_ratio),
        )

    coord_indices = [
        ori_name2coord_idx[Path(file_path).name] for file_path in file_paths
    ]
    # downsized_coords = [coord_idx2downsized_coord(coord_idx, downsized_tile_size) for coord_idx in coord_indices]

    for coord_idx in coord_indices:
        x, y = coord_idx
        matrix[y][x] += 1
        matrix[y + 1][x] += 1
        matrix[y][x + 1] += 1
        matrix[y + 1][x + 1] += 1

    dys = [-1, -1, -1, 0, 1, 1, 1, 0]
    dxs = [-1, 0, 1, 1, 1, 0, -1, -1]
    # dys = [-1, 0, 1, 0]
    # dxs = [0, 1, 0, -1]

    overlap_special_coord_indices = []
    for y in range(1, matrix_height-1):
        for x in range(1, matrix_width-1):
            if matrix[y][x] != 4: continue

            cnt = 0
            for dy, dx in zip(dys, dxs):
                if matrix[y+dy][x+dx] == 4:
                    cnt += 1
            if cnt > 5:
                overlap_special_coord_indices.append((x, y))

    overlap_special_ori_name = [coord_idx2ori_name[coord_idx] for coord_idx in overlap_special_coord_indices]

    # cp the overlap special tiles to the save dir
    os.makedirs(os.path.join(save_dir, image_id), exist_ok=True)
    for ori_name in overlap_special_ori_name:
        # print({})
        from_path = os.path.join("/kaggle/working/UBC-OCEAN-wsi-tiles_without_mask/train_images", image_id, ori_name)
        to_path = os.path.join(save_dir, image_id, ori_name)
        # print(f"{from_path} -> {to_path}")
        shutil.copy(from_path, to_path)


def wrapper_func_for_visualize_inference_result_overlap_tiles_special(args):
    return visualize_inference_result_overlap_tiles_special(*args)

def visualize_inference_result_concurrent():
    save_dir = "/kaggle/working/UBC-OCEAN-images/v2/tma_4_eda_inferenced_wsi_tiles_result_side_by_side_1"
    os.makedirs(save_dir, exist_ok=True)
    with open("/kaggle/working/UBC-OCEAN-wsi-tiles_without_mask_selected_v2/image_id2file_paths.pkl", "rb") as f:
        image_id2file_paths = pickle.load(f)
    # for image_id, file_paths in tqdm(image_id2file_paths.items(), total=len(image_id2file_paths)):
    #     visualize_inference_result_overlap_tiles_special(file_paths, image_id, save_dir)


    # multi processing

    num_threads = 12
    pool = mproc.Pool(num_threads)
    tqdm_bar = tqdm(total=len(image_id2file_paths))
    for _ in pool.imap_unordered(
        wrapper_func_for_visualize_inference_result_overlap_tiles_special,
        (
            (file_paths, image_id, save_dir)
            for image_id, file_paths in image_id2file_paths.items()
        ),
    ):
        tqdm_bar.update()
    pool.close()
    pool.join()


def select_tiles_without_other():
    save_dir = "/kaggle/working/UBC-OCEAN-wsi-tiles_without_mask_selected/train_images"

    # load pickle
    with open("image_id2file_paths.pkl", "rb") as f:
        image_id2file_paths = pickle.load(f)
    os.makedirs(save_dir, exist_ok=True)
    # for image_id, file_paths in tqdm(image_id2file_paths.items(), total=len(image_id2file_paths)):
    #     select_tiles_from_inference_result_overlap_tiles_special(file_paths, image_id, save_dir)
    #     # visualize_inference_result_overlap_tiles_special(file_paths, image_id, save_dir)

    # multi processing
    def wrapper_func(args):
        return select_tiles_from_inference_result_overlap_tiles_special(*args)

    num_threads = 12
    pool = mproc.Pool(num_threads)
    tqdm_bar = tqdm(total=len(image_id2file_paths))
    for _ in pool.imap_unordered(
        wrapper_func,
        (
            (file_paths, image_id, save_dir)
            for image_id, file_paths in image_id2file_paths.items()
        ),
    ):
        tqdm_bar.update()
    pool.close()
    pool.join()

def select_tiles_with_other():
    save_dir = "/kaggle/working/UBC-OCEAN-wsi-tiles_without_mask_selected_with_others/train_images"

    # load pickle
    with open("/kaggle/working/UBC-OCEAN-wsi-tiles_without_mask_selected_with_others/image_id2file_paths.pkl", "rb") as f:
        image_id2file_paths = pickle.load(f)
    os.makedirs(save_dir, exist_ok=True)
    # for image_id, file_paths in tqdm(image_id2file_paths.items(), total=len(image_id2file_paths)):
    #     select_tiles_from_inference_result_overlap_tiles_special(file_paths, image_id, save_dir)
    #     # visualize_inference_result_overlap_tiles_special(file_paths, image_id, save_dir)

    # multi processing
    def wrapper_func(args):
        return select_tiles_from_inference_result_overlap_tiles_special(*args)

    num_threads = 12
    pool = mproc.Pool(num_threads)
    tqdm_bar = tqdm(total=len(image_id2file_paths))
    for _ in pool.imap_unordered(
        wrapper_func,
        (
            (file_paths, image_id, save_dir)
            for image_id, file_paths in image_id2file_paths.items()
        ),
    ):
        tqdm_bar.update()
    pool.close()
    pool.join()


def make_csv_from_selected_tiles():

    ori_df = pd.read_csv("/kaggle/working/UBC-OCEAN/train.csv")
    image_id2label = {str(row["image_id"]): row["label"] for _, row in ori_df.iterrows()}

    root_dir = "/kaggle/working/UBC-OCEAN-wsi-tiles_without_mask_selected/train_images"
    new_csv_path = "/kaggle/working/UBC-OCEAN-wsi-tiles_without_mask_selected/train.csv"

    file_paths = glob(root_dir + "/*/*.jpg")

    rows = []
    for file_path in file_paths:
        image_id = str(Path(file_path).parent.name)
        row = {
            "file_path": file_path,
            "image_id": image_id,
            "label": image_id2label[image_id],
            "width": 512,
            "height": 512,
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(new_csv_path, index=False)

def make_csv_from_selected_tiles_with_mask():
    """ pick df from pred_label == label + pred_label == 'Other' """
    # save pickle

    ori_df = pd.read_csv("/kaggle/working/UBC-OCEAN/train.csv")
    ori_df = ori_df[ori_df["is_tma"] == False]  # filter out tma images
    image_id2label = ori_df.set_index("image_id")["label"].to_dict()
    image_id2label = {str(k): v for k, v in image_id2label.items()}


    inference_result_csv_path = (
        "/kaggle/working/UBC-OCEAN-wsi-tiles_without_mask/inference_result.csv"
    )
    inference_result_df = pd.read_csv(inference_result_csv_path)
    correct_df = inference_result_df[
        inference_result_df["label"] == inference_result_df["pred_label"]
    ]
    other_df = inference_result_df[inference_result_df["label"] != inference_result_df["pred_label"]]
    other_df = other_df[other_df["pred_label"] == 'Other']

    df = pd.concat([correct_df, other_df], ignore_index=True)
    image_id2file_paths = defaultdict(list)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # image_id = row["image_id"]
        file_path = row["file_path"]
        image_id = Path(file_path).parent.name
        file_name = Path(file_path).name
        image_id2file_paths[image_id].append(file_path)
    with open("/kaggle/working/UBC-OCEAN/image_id2file_paths.pkl", "wb") as f:
        pickle.dump(image_id2file_paths, f)

def make_image_id2file_paths_from_selected_tiles_without_mask():
    """ pick df from pred_label == label + pred_label == 'Other' """

    # save pickle
    inference_result_csv_path = "/kaggle/working/UBC-OCEAN-wsi-tiles_without_mask/inference_result_maxvit_07.csv"
    inference_result_df = pd.read_csv(inference_result_csv_path)
    correct_df = inference_result_df[inference_result_df['label'] == inference_result_df['pred_label']]

    other_df = inference_result_df[inference_result_df["pred_label"] == 'Other']

    # df = correct_df
    df = pd.concat([correct_df, other_df], ignore_index=True)
    image_id2file_paths = defaultdict(list)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # image_id = row["image_id"]
        file_path = row["file_path"]
        image_id = Path(file_path).parent.name
        file_name = Path(file_path).name
        image_id2file_paths[image_id].append(file_path)
    with open("/kaggle/working/UBC-OCEAN-wsi-tiles_without_mask_selected_from_maxvit_tiny/image_id2file_paths.pkl", "wb") as f:
        pickle.dump(image_id2file_paths, f)


# make_csv_from_selected_tiles()
# make_csv_from_selected_tiles_with_mask()

# visualize_inference_result_concurrent()
make_image_id2file_paths_from_selected_tiles_without_mask()
