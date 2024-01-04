import gc
import multiprocessing as mproc
import os
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = 10000000000
from concurrent.futures import ThreadPoolExecutor


def compress_supplementary_mask():
    mask_path = "/kaggle/working/UBC-OCEAN-supplemental_masks"
    compressed_mask_path = "/kaggle/working/UBC-OCEAN-compressed_supplemental_masks"
    train_images_path = "/kaggle/input/UBC-OCEAN/train_images"

    mask_image_ids = set(os.listdir(mask_path))
    train_image_ids = set(os.listdir(train_images_path))

    os.makedirs(compressed_mask_path, exist_ok=True)

    def resize_mask(mask_image_path, new_mask_image_path):
        with Image.open(mask_image_path) as mask_image:
            width, height = mask_image.size
            resize_ratio = width / 3000
            resized_image = mask_image.resize(
                (3000, int(height / resize_ratio)), Image.NEAREST
            )
            resized_image.save(new_mask_image_path)
            resized_image.close()

    for mask_image_id in mask_image_ids:
        mask_image_path = os.path.join(mask_path, mask_image_id)
        new_mask_image_path = os.path.join(compressed_mask_path, mask_image_id)
        resize_mask(mask_image_path, new_mask_image_path)

def supplementary_mask_eda():
    """
    to check distribution of supplementary mask images' labels
    """
    mask_path = "/kaggle/working/UBC-OCEAN-supplemental_masks"
    mask_image_ids = os.listdir(mask_path)
    mask_image_ids = [
        int(mask_image_id.split(".")[0]) for mask_image_id in mask_image_ids
    ]

    ori_train_csv_path = "/kaggle/input/UBC-OCEAN/train.csv"
    ori_train_df = pd.read_csv(ori_train_csv_path)
    ori_train_df = ori_train_df[ori_train_df["image_id"].isin(mask_image_ids)]


def overlay_mask_on_image(original_image, mask_image, alpha):
    # Resize mask image to match the size of the original image
    mask_image_resized = cv2.resize(
        mask_image, (original_image.shape[1], original_image.shape[0])
    )

    # Ensure that the alpha value is between 0 and 1
    alpha = max(0, min(alpha, 1))

    # Blend the images using the alpha value
    overlaid_image = cv2.addWeighted(
        original_image, 1 - alpha, mask_image_resized, alpha, 0
    )

    # Concatenate original and overlaid images horizontally
    side_by_side = cv2.hconcat([original_image, overlaid_image])

    # return overlaid_image
    return side_by_side


def run_overlay_mask_on_image(idx, comp_mask_image_path, save_path):
    mask_image_id = Path(comp_mask_image_path).stem
    train_thumbnail_image_path = f"/kaggle/input/UBC-OCEAN/train_thumbnails/{mask_image_id}_thumbnail.png"
    assert os.path.exists(train_thumbnail_image_path)

    mask_image = cv2.imread(comp_mask_image_path)
    train_thumbnail_image = cv2.imread(train_thumbnail_image_path)

    assert mask_image.size == train_thumbnail_image.size

    overlap_image_save_path = os.path.join(save_path, mask_image_id + ".jpg")

    cv2.imwrite(
        overlap_image_save_path,
        overlay_mask_on_image(train_thumbnail_image, mask_image, alpha=0.3),
    )


def run_overlay_mask_on_image_wrapper(args):
    return run_overlay_mask_on_image(*args)


def run_overlay_mask_on_images():
    compressed_mask_image_paths = glob("/kaggle/working/UBC-OCEAN-compressed_supplemental_masks/*.png")
    train_thumbnail_image_paths = glob("/kaggle/input/UBC-OCEAN/train_thumbnails/*.png")
    save_path = "/kaggle/working/mask_overlap_images"

    mask_image_ids = [
        Path(mask_image_path).stem for mask_image_path in compressed_mask_image_paths
    ]
    os.makedirs(save_path, exist_ok=True)

    num_threads = 16
    pool = mproc.Pool(num_threads)
    tqdm_bar = tqdm(total=len(compressed_mask_image_paths))
    for _ in pool.imap_unordered(
        run_overlay_mask_on_image_wrapper,
        (
            (idx, img_path, save_path)
            for idx, img_path in enumerate(compressed_mask_image_paths)
        ),
    ):
        tqdm_bar.update()
    pool.close()
    pool.join()


if __name__ == "__main__":
    # supplementary_mask_eda()
    run_overlay_mask_on_images()
