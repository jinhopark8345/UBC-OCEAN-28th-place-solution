import gc
import glob
import multiprocessing as mproc
import os
import pickle
import random
import shutil
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

# pd.set_option("display.max_colwidth", None)


def overlay_tiles(image, tumor_coords, other_coords, tile_size, alpha=0.3):
    """
    Overlay transparent squares on an image.

    :param image: Original image to overlay squares on.
    :param coords: List of coordinates for the top-left corner of each square.
    :param tile_size: Size of each square (width, height).
    :param color: Color of the overlay (B, G, R).
    :param alpha: Transparency of the overlay.
    :return: Image with overlays.
    """
    tumor_color = (0, 0, 255)  # red
    other_color = (0, 255, 0)  # green

    overlay = image.copy()
    for coord in tumor_coords:
        top_left = tuple(coord)
        bottom_right = (top_left[0] + tile_size[0], top_left[1] + tile_size[1])
        cv2.rectangle(overlay, top_left, bottom_right, tumor_color, -1)

    for coord in other_coords:
        top_left = tuple(coord)
        bottom_right = (top_left[0] + tile_size[0], top_left[1] + tile_size[1])
        cv2.rectangle(overlay, top_left, bottom_right, other_color, -1)

    overlayed_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    combined = np.concatenate((image, overlayed_image), axis=1)
    return combined


def filter_by_group_size(H, W, coords, filter_group_size=10):
    matrix = [[0 for _ in range(W)] for _ in range(H)]
    for coord in coords:
        x, y = coord
        matrix[y][x] = 1

    group_size_matrix = np.zeros_like(matrix)
    visited = np.zeros_like(matrix)
    cur_visited = set()
    H, W = len(matrix), len(matrix[0])

    for y in range(H):
        for x in range(W):
            if visited[y][x]:
                continue
            if matrix[y][x] != 1:
                continue

            cur_visited = set()
            stack = [(y, x)]

            while stack:
                y, x = stack.pop()
                if visited[y][x]:
                    continue
                visited[y][x] = 1
                cur_visited.add((y, x))
                # diagonal as well
                for dx, dy in [
                    (1, 0),
                    (-1, 0),
                    (0, 1),
                    (0, -1),
                    (1, 1),
                    (-1, 1),
                    (1, -1),
                    (-1, -1),
                ]:
                    # for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    if x + dx < 0 or x + dx >= W or y + dy < 0 or y + dy >= H:
                        continue
                    if visited[y + dy][x + dx]:
                        continue
                    if matrix[y + dy][x + dx] != 1:
                        continue
                    stack.append((y + dy, x + dx))

            for y, x in cur_visited:
                group_size_matrix[y][x] = len(cur_visited)

    selected_coords = []
    for coord in coords:
        x, y = coord
        if group_size_matrix[y][x] > filter_group_size:
            selected_coords.append(coord)
    return selected_coords


def visualize_and_select_inference_result_overlap_tiles_special(
    label_file_paths,
    image_id2thumbnail_paths,
    image_id2train_image_paths,
    image_id,
    vis_save_dir,
    select_tile_save_dir,
    tile_stride=512,
    tile_size=1024,
    other_valid_neighbor_cnt=2,
    tumor_valid_neighbor_cnt=4,
    tumor_other_ratio=1,
    filter_group_size=4,
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
    other_matrix = [[0 for _ in range(matrix_width)] for _ in range(matrix_height)]
    tumor_matrix = [[0 for _ in range(matrix_width)] for _ in range(matrix_height)]

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

    other_file_paths = [
        file_path for label, file_path in label_file_paths if label == "Other"
    ]
    tumor_file_paths = [
        file_path for label, file_path in label_file_paths if label != "Other"
    ]

    other_coord_indices = [
        ori_name2coord_idx[Path(file_path).name] for file_path in other_file_paths
    ]

    tumor_coord_indices = [
        ori_name2coord_idx[Path(file_path).name] for file_path in tumor_file_paths
    ]
    # downsized_coords = [coord_idx2downsized_coord(coord_idx, downsized_tile_size) for coord_idx in coord_indices]

    for coord_idx in other_coord_indices:
        x, y = coord_idx
        other_matrix[y][x] += 1
        other_matrix[y + 1][x] += 1
        other_matrix[y][x + 1] += 1
        other_matrix[y + 1][x + 1] += 1

    for coord_idx in tumor_coord_indices:
        x, y = coord_idx
        tumor_matrix[y][x] += 1
        tumor_matrix[y + 1][x] += 1
        tumor_matrix[y][x + 1] += 1
        tumor_matrix[y + 1][x + 1] += 1

    dys = [-1, -1, -1, 0, 1, 1, 1, 0]
    dxs = [-1, 0, 1, 1, 1, 0, -1, -1]
    # dys = [-1, 0, 1, 0]
    # dxs = [0, 1, 0, -1]

    other_overlap_special_coord_indices = []
    for y in range(1, matrix_height - 1):
        for x in range(1, matrix_width - 1):
            if other_matrix[y][x] != 4:
                continue

            cnt = 0
            for dy, dx in zip(dys, dxs):
                if other_matrix[y + dy][x + dx] == 4:
                    cnt += 1
            if cnt >= other_valid_neighbor_cnt:
                other_overlap_special_coord_indices.append((x, y))

    tumor_overlap_special_coord_indices = []
    for y in range(1, matrix_height - 1):
        for x in range(1, matrix_width - 1):
            if tumor_matrix[y][x] != 4:
                continue

            cnt = 0
            for dy, dx in zip(dys, dxs):
                if tumor_matrix[y + dy][x + dx] == 4:
                    cnt += 1
            if cnt >= tumor_valid_neighbor_cnt:
                tumor_overlap_special_coord_indices.append((x, y))

    # print(f"{image_id = }")
    # print(f'before: {len(other_overlap_special_coord_indices)=}')
    other_overlap_special_coord_indices = filter_by_group_size(
        matrix_height,
        matrix_width,
        other_overlap_special_coord_indices,
        filter_group_size,
    )
    # print(f'after: {len(other_overlap_special_coord_indices)=}')

    # print(f'before: {len(tumor_overlap_special_coord_indices)=}')
    tumor_overlap_special_coord_indices = filter_by_group_size(
        matrix_height,
        matrix_width,
        tumor_overlap_special_coord_indices,
        filter_group_size,
    )
    # print(f'after: {len(tumor_overlap_special_coord_indices)=}')

    other_downsized_overlap_coords = [
        coord_idx2downsized_coord(coord_idx, downsized_tile_size)
        for coord_idx in other_overlap_special_coord_indices
    ]

    tumor_downsized_overlap_coords = [
        coord_idx2downsized_coord(coord_idx, downsized_tile_size)
        for coord_idx in tumor_overlap_special_coord_indices
    ]

    tumor_tile_ori_names = [
        coord_idx2ori_name[coord_idx]
        for coord_idx in tumor_overlap_special_coord_indices
    ]
    other_tile_ori_names = [
        coord_idx2ori_name[coord_idx]
        for coord_idx in other_overlap_special_coord_indices
    ]

    # TODO : is it better really?
    if (
        not len(tumor_overlap_special_coord_indices)
        > len(other_overlap_special_coord_indices) * tumor_other_ratio
    ):
        print(f"not enough tumor tiles found in: {image_id}")
        return

    # Overlay the tiles
    if vis_save_dir is not None:
        result_image = overlay_tiles(
            thumbnail,
            tumor_downsized_overlap_coords,
            other_downsized_overlap_coords,
            downsized_tile_size,
        )  # red

        # Save or display the result
        cv2.imwrite(
            os.path.join(vis_save_dir, image_id + ".jpg"), result_image
        )  # Save the image

    if select_tile_save_dir is not None:
        # cp the other overlap special tiles to the save dir
        other_overlap_special_ori_name = [
            coord_idx2ori_name[coord_idx]
            for coord_idx in other_overlap_special_coord_indices
        ]
        cur_dir = os.path.join(select_tile_save_dir, "other", image_id)
        os.makedirs(cur_dir, exist_ok=True)
        discard_dir = os.path.join("discard", "other", image_id)
        os.makedirs(discard_dir, exist_ok=True)
        for ori_name in other_overlap_special_ori_name:
            from_path = os.path.join(
                "UBC-OCEAN-wsi-tiles_without_mask/train_images",
                image_id,
                ori_name,
            )

            to_path = os.path.join(cur_dir, ori_name)
            shutil.copy(from_path, to_path)

        # cp the tumor overlap special tiles to the save dir
        tumor_overlap_special_ori_name = [
            coord_idx2ori_name[coord_idx]
            for coord_idx in tumor_overlap_special_coord_indices
        ]
        cur_dir = os.path.join(select_tile_save_dir, "tumor", image_id)
        os.makedirs(cur_dir, exist_ok=True)
        discard_dir = os.path.join(
            "discard", "tumor", image_id
        )
        os.makedirs(discard_dir, exist_ok=True)
        for ori_name in tumor_overlap_special_ori_name:
            # print({})
            from_path = os.path.join(
                "UBC-OCEAN-wsi-tiles_without_mask/train_images",
                image_id,
                ori_name,
            )

            # if not is_validate_tile_without_mask(cv2.imread(from_path), 0.6, 0.8, 220):
            #     to_path = os.path.join(discard_dir, ori_name)
            #     shutil.copy(from_path, to_path)
            #     continue
            to_path = os.path.join(cur_dir, ori_name)
            # print(f"{from_path} -> {to_path}")
            shutil.copy(from_path, to_path)


# multi processing
def visualize_and_select_inference_result_overlap_tiles_special_wrapper_func(args):
    return visualize_and_select_inference_result_overlap_tiles_special(*args)


def visualize_and_select_inference_result_concurrent(
    image_id2label_file_paths_pkl_path: str,
    vis_save_dir: str,
    selected_tile_save_dir: str,
    num_threads: int = 12,
):
    os.makedirs(vis_save_dir, exist_ok=True)
    os.makedirs(selected_tile_save_dir, exist_ok=True)

    # load pickle
    with open(image_id2label_file_paths_pkl_path, "rb") as f:
        image_id2label_file_paths = pickle.load(f)

    thumbnail_paths = glob("/kaggle/input/UBC-OCEAN/train_thumbnails/*.png")
    image_id2thumbnail_paths = {Path(p).name.split("_")[0]: p for p in thumbnail_paths}

    train_image_paths = glob("/kaggle/input/UBC-OCEAN/train_images/*.png")
    image_id2train_image_paths = {
        Path(p).name.split("_")[0].split(".")[0]: p for p in train_image_paths
    }

    # single thread test
    # for image_id, (label_file_paths) in tqdm(image_id2label_file_paths.items(), total=len(image_id2label_file_paths)):
    # #     # visualize_and_select_inference_result_overlap_tiles_special(
    # #     #     label_file_paths, image_id2thumbnail_paths, image_id2train_image_paths, image_id, vis_save_dir, select_tile_save_dir
    # #     # )

    #     visualize_and_select_inference_result_overlap_tiles_special(
    #         label_file_paths, image_id2thumbnail_paths, image_id2train_image_paths, image_id, vis_save_dir, None
    #     )

    # multhi thread
    pool = mproc.Pool(num_threads)
    tqdm_bar = tqdm(total=len(image_id2label_file_paths))
    for _ in pool.imap_unordered(
        visualize_and_select_inference_result_overlap_tiles_special_wrapper_func,
        (
            (
                label_file_paths,
                image_id2thumbnail_paths,
                image_id2train_image_paths,
                image_id,
                vis_save_dir,
                selected_tile_save_dir,
            )
            # (label_file_paths, image_id2thumbnail_paths, image_id2train_image_paths, image_id, vis_save_dir, None)
            for image_id, (label_file_paths) in image_id2label_file_paths.items()
        ),
    ):
        tqdm_bar.update()
    pool.close()
    pool.join()


def make_image_id2label_file_paths_from_selected_tiles(
    inference_result_csv_path: str, save_pkl_path: str
):
    df = pd.read_csv(inference_result_csv_path)
    os.makedirs(os.path.dirname(save_pkl_path), exist_ok=True)

    ori_df = pd.read_csv("/kaggle/input/UBC-OCEAN/train.csv")
    ori_df = ori_df[ori_df["is_tma"] == False]  # filter out tma images
    image_id2label = ori_df.set_index("image_id")["label"].to_dict()
    image_id2label = {str(k): v for k, v in image_id2label.items()}

    thumbnail_paths = glob("/kaggle/input/UBC-OCEAN/train_thumbnails/*.png")
    image_id2thumbnail_paths = {Path(p).name.split("_")[0]: p for p in thumbnail_paths}
    train_image_paths = glob("/kaggle/input/UBC-OCEAN/train_images/*.png")
    image_id2train_image_paths = {
        Path(p).name.split("_")[0].split(".")[0]: p for p in train_image_paths
    }

    image_id2label_file_path = defaultdict(list)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        # image_id = row["image_id"]
        file_path = row["file_path"]
        label = row["label"]
        image_id = Path(file_path).parent.name
        file_name = Path(file_path).name
        image_id2label_file_path[image_id].append((label, file_path))
    with open(save_pkl_path, "wb") as f:
        pickle.dump(image_id2label_file_path, f)


if __name__ == "__main__":

    inference_result_csv_path = "/kaggle/working/UBC-OCEAN-v2/wsi-tiles-without-mask-selected-from-ensembled-tumor-classifier-inference-result-v2"
    pkl_path = "/kaggle/working/UBC-OCEAN-v2/wsi-tiles-without-mask-selected-from-ensembled-tumor-classifier-inference-result-v2/train_ensembled.csv.pkl"
    make_image_id2label_file_paths_from_selected_tiles(inference_result_csv_path, pkl_path)

    vis_save_dir = "/kaggle/working/UBC-OCEAN-v2/wsi-tiles-without-mask-selected-from-ensembled-tumor-classifier-inference-result-v2/vis"
    selected_tile_save_dir = "/kaggle/working/UBC-OCEAN-v2/wsi-tiles-without-mask-selected-from-ensembled-tumor-classifier-inference-result-v2/selected_tiles"

    visualize_and_select_inference_result_concurrent(pkl_path, vis_save_dir, selected_tile_save_dir)
    make_csv_from_selected_tiles_without_mask()
