import datetime
import gc
import glob
import multiprocessing as mproc
import os
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyvips
from joblib import Parallel, delayed
from PIL import Image
from tqdm.auto import tqdm

os.environ["VIPS_CONCURRENCY"] = "2"
# settting higher than 2gb cause memory issue and crash
os.environ["VIPS_DISC_THRESHOLD"] = "2gb"


def is_validate_tile(tile, bg_thr, white_thr=240) -> bool:
    black_bg_mask = np.sum(tile, axis=2) == 0
    white_bg_mask = np.mean(tile, axis=2) > white_thr
    if np.sum(black_bg_mask) + np.sum(white_bg_mask) >= (
        np.prod(black_bg_mask.shape) * bg_thr
    ):
        return False
    return True


def extract_image_tiles_without_mask(
    p_img,
    tile_path,
    discard_path,
    tile_size: Tuple[int] = (1024, 1024),
    tile_stride: int = 512,
    new_size: Tuple[int] = (512, 512),
    bg_thr: float = 0.6,
) -> list:
    im = pyvips.Image.new_from_file(p_img)

    tile_width, tile_height = tile_size
    # https://stackoverflow.com/a/47581978/4521646
    idxs = [
        (y, y + tile_height, x, x + tile_width)
        for y in range(0, im.height - tile_height + 1, tile_stride)
        for x in range(0, im.width - tile_width + 1, tile_stride)
    ]
    files = []
    print(f"total tiles: {len(idxs)}")

    for k, (y, y_, x, x_) in enumerate(idxs):
        # https://libvips.github.io/pyvips/vimage.html#pyvips.Image.crop
        tile = im.crop(
            x, y, min(tile_width, im.width - x), min(tile_height, im.height - y)
        ).numpy()[..., :3]

        # if tile size isn't the same as the expected size, pad it with zeros
        if tile.shape[:2] != (tile_height, tile_width):
            tile_ = tile
            new_shape = (
                (tile_height, tile_width)
                if tile.ndim == 2
                else (tile_height, tile_width, tile.shape[2])
            )
            tile = np.zeros(new_shape, dtype=tile.dtype)
            tile[: tile_.shape[0], : tile_.shape[1], ...] = tile_

        if is_validate_tile(tile, bg_thr):
            p_img_out = os.path.join(
                tile_path, f"{k:06}_{int(x)}-{int(y)}-{tile_width}-{tile_height}.jpg"
            )
            Image.fromarray(tile).resize(new_size, Image.LANCZOS).save(p_img_out)
            files.append(p_img_out)
        else:
            if discard_path:
                p_img_out = os.path.join(
                    discard_path,
                    f"{k:06}_{int(x)}-{int(y)}-{tile_width}-{tile_height}.jpg",
                )
                Image.fromarray(tile).resize(new_size, Image.LANCZOS).save(p_img_out)

    return files, idxs


def extract_image_tiles_based_on_mask_filter_bg(
    img_path: str,
    mask_path: str,
    tile_path: str,
    discard_path: str,
    tile_size: int,
    tile_stride: int,
    new_size: Tuple[int, int],
    red_thr: float,
    other_thr: float,
    bg_thr: float,
) -> list:
    im = pyvips.Image.new_from_file(img_path)
    mask = pyvips.Image.new_from_file(mask_path)

    im_width, im_height = im.width, im.height
    tile_width, tile_height = tile_size
    # https://stackoverflow.com/a/47581978/4521646
    idxs = [
        (y, y + tile_height, x, x + tile_width)
        for y in range(0, im.height - tile_height + 1, tile_stride)
        for x in range(0, im.width - tile_width + 1, tile_stride)
    ]
    files = []
    print(f"total tiles: {len(idxs)}")
    # print(f"center tile size: {center_tile_size}")

    for k, (y, y_, x, x_) in enumerate(idxs):
        # https://libvips.github.io/pyvips/vimage.html#pyvips.Image.crop
        tile = im.crop(
            x, y, min(tile_width, im.width - x), min(tile_height, im.height - y)
        ).numpy()[..., :3]
        mask_tile = mask.crop(
            x, y, min(tile_width, mask.width - x), min(tile_height, mask.height - y)
        ).numpy()[..., :3]

        # if tile size isn't the same as the expected size, pad it with zeros
        if tile.shape[:2] != (tile_height, tile_width):
            tile_ = tile
            new_shape = (
                (tile_height, tile_width)
                if tile.ndim == 2
                else (tile_height, tile_width, tile.shape[2])
            )
            tile = np.zeros(new_shape, dtype=tile.dtype)
            tile[: tile_.shape[0], : tile_.shape[1], ...] = tile_

        tile_color = None

        red_mask = (
            (mask_tile[..., 0] > 200)
            & (mask_tile[..., 1] < 50)
            & (mask_tile[..., 2] < 50)
        )

        red_ratio, other_ratio = 0, 0

        red_ratio = np.sum(red_mask) / np.prod(red_mask.shape)
        if red_ratio > red_thr:
            tile_color = "red"

        if tile_color is None:
            green_mask = (
                (mask_tile[..., 0] < 50)
                & (mask_tile[..., 1] > 200)
                & (mask_tile[..., 2] < 50)
            )
            blue_mask = (
                (mask_tile[..., 0] < 50)
                & (mask_tile[..., 1] < 50)
                & (mask_tile[..., 2] > 200)
            )

            other_ratio = (np.sum(green_mask) + np.sum(blue_mask)) / np.prod(
                red_mask.shape
            )
            if other_ratio > other_thr:
                tile_color = "other"

        if tile_color == "red" or (tile_color == "other" and red_ratio < 0.02):
            if is_validate_tile(
                tile, bg_thr
            ):  # even if it's red, check if it's valid tile
                p_img_out = os.path.join(
                    tile_path,
                    f"{tile_color}_{k:06}_{int(x)}-{int(y)}-{tile_width}-{tile_height}_red_{red_ratio:.4f}_other_{other_ratio:.4f}_.jpg",
                )
                Image.fromarray(tile).resize(new_size, Image.LANCZOS).save(p_img_out)
                files.append(p_img_out)
            else:  # even if it has color, too much background -> discard
                p_img_out = os.path.join(
                    tile_path,
                    f"{tile_color}_{k:06}_{int(x)}-{int(y)}-{tile_width}-{tile_height}_red_{red_ratio:.4f}_other_{other_ratio:.4f}_.jpg",
                )
                Image.fromarray(tile).resize(new_size, Image.LANCZOS).save(p_img_out)
    return files, idxs


def extract_tiles(
    idx: int,
    img_path: str,
    mask_path: str,
    tile_image_name_folder: str,
    discard_image_name_folder: Optional[str],
    tile_size: Tuple[int, int],
    tile_stride: int,
    new_size: Tuple[int, int],
) -> None:
    print(f"processing #{idx}: {img_path}")

    image_name, _ = os.path.splitext(os.path.basename(img_path))
    tile_image_name_folder = os.path.join(tile_image_name_folder, image_name)
    discard_image_name_folder = os.path.join(discard_image_name_folder, image_name)

    os.makedirs(tile_image_name_folder, exist_ok=True)
    os.makedirs(discard_image_name_folder, exist_ok=True)

    if mask_path is None:
        # without mask images
        tiles, _ = extract_image_tiles_without_mask(
            p_img=img_path,
            tile_path=tile_image_name_folder,
            discard_path=discard_image_name_folder,
            tile_size=tile_size,
            tile_stride=tile_stride,
            new_size=new_size,
            bg_thr=0.6,
        )
    else:
        # with mask images
        tiles, _ = extract_image_tiles_based_on_mask_filter_bg(
            img_path=img_path,
            mask_path=mask_path,
            tile_path=tile_image_name_folder,
            discard_path=discard_image_name_folder,
            tile_size=tile_size,
            tile_stride=tile_stride,
            new_size=new_size,
            red_thr=0.1,
            other_thr=0.7,
            bg_thr=0.6,
        )

    gc.collect()
    time.sleep(1)
    print(f"processing done #{idx}: {img_path}")


def wrapper_extract_tiles(args):
    return extract_tiles(*args)


def extract_tiles_from_wsi_images_with_mask(
    image_dir: str,
    mask_dir: str,
    tile_size: Tuple[int, int],
    stride: int,
    resize: Tuple[int, int],
    num_threads: int = 2,
    save_root: str = "/kaggle/working/",
    dataset_prefix: str = "train_tiles_based_on_mask_tile",
    version: Optional[str] = None,
):
    if version is None:
        version = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

    tile_size_str = "x".join([str(e) for e in tile_size])
    resize_str = "x".join([str(e) for e in resize])
    dataset_name = f"{dataset_prefix}-tile_size{tile_size_str}-stride{stride}-resize{resize_str}_v{version}"
    dataset_path = os.path.join(save_root, dataset_name)
    discard_path = dataset_path + "_discard"

    if not os.path.exists(mask_dir):
        assert False, f"mask_dir does not exist: {mask_dir}"

    if not os.path.exists(image_dir):
        assert False, f"image_dir does not exist: {image_dir}"

    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(discard_path, exist_ok=True)

    image_paths = glob.glob(os.path.join(image_dir, "*.png"))
    mask_paths = glob.glob(os.path.join(mask_dir, "*.png"))
    mask_ids = set([os.path.splitext(os.path.basename(p))[0] for p in mask_paths])

    image_mask_path_list = []
    for image_path in image_paths:
        name, _ = os.path.splitext(os.path.basename(image_path))
        if name in mask_ids:
            image_mask_path_list.append(
                (image_path, os.path.join(mask_dir, name + ".png"))
            )

    print(f"found images: {len(image_mask_path_list)}")

    pool = mproc.Pool(num_threads)
    tqdm_bar = tqdm(total=len(image_mask_path_list), desc="extracting tiles...")
    for _ in pool.imap_unordered(
        wrapper_extract_tiles,
        (
            (
                idx,
                img_path,
                mask_path,
                dataset_path,
                discard_path,
                tile_size,
                stride,
                resize,
            )
            for idx, (img_path, mask_path) in enumerate(image_mask_path_list)
        ),
    ):
        tqdm_bar.update()
    pool.close()
    pool.join()


def extract_tiles_from_wsi_images_without_mask(
    tile_size: Tuple[int, int],
    stride: int,
    resize: Tuple[int, int],
    exclude_image_ids: Optional[set] = None,
    version: Optional[str] = None,
    num_threads=2,
    save_root: str = "/kaggle/working/",
    dataset_prefix: str = "train_tiles_without_mask_tile",
):
    """extract tiles excluding images with masks"""
    image_dir = "/kaggle/input/UBC-OCEAN/train_images"

    if version is None:
        version = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    tile_size_str = "x".join([str(e) for e in tile_size])
    resize_str = "x".join([str(e) for e in resize])
    dataset_name = f"{dataset_prefix}-tile_size{tile_size_str}-stride{stride}-resize{resize_str}_v{version}"
    dataset_path = os.path.join(save_root, dataset_name)
    discard_path = dataset_path + "_discard"

    if not os.path.exists(image_dir):
        assert False, f"image_dir does not exist: {image_dir}"

    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(discard_path, exist_ok=True)

    df = pd.read_csv("/kaggle/input/UBC-OCEAN/train.csv")
    df["file_path"] = df["image_id"].apply(
        lambda x: os.path.join(image_dir, str(x) + ".png")
    )

    # filter out TMA images -> use only WSI images
    wsi_df = df[df["is_tma"] == False]

    if exclude_image_ids:
        no_mask_wsi_df = wsi_df[~wsi_df["image_id"].isin(mask_provided_image_ids)]
        image_paths = no_mask_wsi_df["file_path"].tolist()
    else:
        image_paths = wsi_df["file_path"].tolist()

    print(f"found images: {len(image_paths)}")

    pool = mproc.Pool(num_threads)
    tqdm_bar = tqdm(total=len(image_paths))
    for _ in pool.imap_unordered(
        wrapper_extract_tiles,
        (
            (
                idx,
                img_path,
                None,
                dataset_path,
                discard_path,
                tile_size,
                stride,
                resize,
            )
            for idx, img_path in enumerate(image_paths)
        ),
    ):
        tqdm_bar.update()
    pool.close()
    pool.join()


def extract_tiles_from_wsi_thumbnail_with_mask(
    image_dir: str,
    mask_dir: str,
    tile_size: Tuple[int, int],
    stride: int,
    resize: Tuple[int, int],
    num_threads: int = 2,
    save_root: str = "/kaggle/working/",
    dataset_prefix: str = "train_thumbnail_tiles_based_on_mask_tile",
    version: Optional[str] = None,
):
    if version is None:
        version = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

    tile_size_str = "x".join([str(e) for e in tile_size])
    resize_str = "x".join([str(e) for e in resize])
    dataset_name = f"{dataset_prefix}-tile_size{tile_size_str}-stride{stride}-resize{resize_str}_v{version}"
    dataset_path = os.path.join(save_root, dataset_name)
    discard_path = dataset_path + "_discard"

    if not os.path.exists(mask_dir):
        assert False, f"mask_dir does not exist: {mask_dir}"

    if not os.path.exists(image_dir):
        assert False, f"image_dir does not exist: {image_dir}"

    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(discard_path, exist_ok=True)

    image_paths = glob.glob(os.path.join(image_dir, "*.png"))
    mask_paths = glob.glob(os.path.join(mask_dir, "*.png"))
    mask_ids = set([os.path.splitext(os.path.basename(p))[0] for p in mask_paths])

    image_mask_path_list = []
    for image_path in image_paths:
        image_id = Path(image_path).name.split("_")[0]
        if image_id in mask_ids:
            image_mask_path_list.append(
                (image_path, os.path.join(mask_dir, image_id + ".png"))
            )

    print(f"found images: {len(image_mask_path_list)}")

    pool = mproc.Pool(num_threads)
    tqdm_bar = tqdm(total=len(image_mask_path_list), desc="extracting tiles...")
    for _ in pool.imap_unordered(
        wrapper_extract_tiles,
        (
            (
                idx,
                img_path,
                mask_path,
                dataset_path,
                discard_path,
                tile_size,
                stride,
                resize,
            )
            for idx, (img_path, mask_path) in enumerate(image_mask_path_list)
        ),
    ):
        tqdm_bar.update()
    pool.close()
    pool.join()



if __name__ == "__main__":
    # even if you have many threads, if you don't have enoguh ram, it will crash

    # image_dir = "/kaggle/input/UBC-OCEAN/train_images"
    # mask_dir = "/kaggle/working/UBC-OCEAN-supplemental_masks"
    # extract_tiles_from_wsi_images_with_mask(image_dir, mask_dir, tile_size=(1024, 1024), stride=512, resize=(512, 512))
    # extract_tiles_from_wsi_images_with_mask(image_dir, mask_dir, tile_size=(1792, 1792), stride=512, new_size=(512, 512))
    # extract_tiles_from_wsi_images_with_mask(image_dir, mask_dir, tile_size=(3072, 3072), stride=1536, new_size=(512, 512))

    # mask_provided_image_ids = set([int(Path(f).stem) for f in os.listdir("/kaggle/working/UBC-OCEAN-supplemental_masks")])
    # extract_tiles_from_wsi_images_without_mask(tile_size=(1024, 1024), stride=512, resize=(512, 512), exclude_image_ids=mask_provided_image_ids)

    mask_dir = "/kaggle/working/UBC-OCEAN-compressed_supplemental_masks"
    image_dir = "/kaggle/input/UBC-OCEAN/train_thumbnails"
    extract_tiles_from_wsi_thumbnail_with_mask(
        image_dir,
        mask_dir,
        tile_size=(200, 200),
        stride=100,
        resize=(200, 200),
        dataset_prefix="train_thumbnail_based_on_mask",
    )
