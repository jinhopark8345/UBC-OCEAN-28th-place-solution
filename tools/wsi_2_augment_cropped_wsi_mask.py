import cv2
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from albumentations import (Compose, HorizontalFlip, RandomCrop, Resize,
                            Rotate, VerticalFlip)
import albumentations as A
import time
import random
import multiprocessing as mproc
from pathlib import Path
from glob import glob
from PIL import Image

labels = ["CC", "EC", "HGSC", "LGSC", "MC", "Other"]

def generate_microsecond_seed():
    """
    Generate a random seed based on the current time in microseconds.
    """
    microseconds = int(time.time() * 1e6)
    seed = microseconds % (2**32)  # Ensure that the seed is within the valid range
    return seed

def generate_random_wsi_thumbnail_height(mean=2005, std=826, min_val=530, max_val=7842):
    # Generate a single random height
    random_height = np.random.normal(mean, std)

    # Clip the value to be within the specified range
    random_height = np.clip(random_height, min_val, max_val)

    return int(random_height)

def overlay_image_helper(background, cropped_image, x, y):

    # Get the coordinates for placing the image on the background
    # x, y = row['x'], row['y']
    output_size = background.shape[:2]
    h, w = cropped_image.shape[:2]

    # Clip the image if it exceeds the boundaries of the background
    image_to_overlay = cropped_image[:min(h, output_size[0]-y), :min(w, output_size[1]-x), :]

    # Overlay the clipped image on the background
    background[y:y+image_to_overlay.shape[0], x:x+image_to_overlay.shape[1]] = image_to_overlay

    return background

def wrapper_func(args):
    return overlay_image(*args)

def gen_image_multi(df, save_dir, num_other=10, num_non_other=3, num_images=100):
    os.makedirs(save_dir, exist_ok=True)

    img_color_mean = [0.8721593659261734, 0.7799686061900686, 0.8644588534918227]
    img_color_std = [0.08258995918115268, 0.10991684444009092, 0.06839816226731532]

    image_aug = A.Compose(
            [
                A.RandomScale(scale_limit=(0.5, 2), p=0.7),  # Random resizing
                A.ShiftScaleRotate(
                    shift_limit=0.2, scale_limit=0.15, rotate_limit=90, p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ],
            p=0.7,
    )

    num_threads = 12


    # overlay_image(0, df, save_dir, image_aug, num_other=num_other, num_non_other=num_non_other)

    pool = mproc.Pool(num_threads)
    tqdm_bar = tqdm(total=num_images)
    for _ in pool.imap_unordered(
        wrapper_func,
        (
            (idx, df, save_dir, image_aug, 30)
            for idx in range(num_images)
        ),
    ):
        tqdm_bar.update()
    pool.close()
    pool.join()


def overlay_image(idx, df, save_dir, image_aug, num_other=10, num_non_other=3):
    """
    1. randomly set output height
    2. randomly pick 10 other cropped images
        randomly pick x, y coordinates for placing the image on the background
    3. pick label from 6 labels (including other) (if it's other -> don't need to add anything)
        4. randomly pick x, y coordinates for placing the image on the background

    """
    os.makedirs(save_dir, exist_ok=True)

    output_width = 3000
    output_height = generate_random_wsi_thumbnail_height()
    output_size = (output_height, output_width)

    other_df = df[df['label'] == 'Other']

    # Filter the dataframe for 'Other' label images
    other_df = other_df.sample(n=num_other, random_state=generate_microsecond_seed())


    # Initialize a black background
    background = np.zeros(output_size + (3,), dtype=np.uint8)

    applied_num_other_cnt = 0

    # Overlay 'Other' label images on the black background
    for _, row in other_df.iterrows():
        crooped_image = cv2.imread(row['file_path'])
        crooped_image = cv2.resize(crooped_image, (row['w'], row['h']))
        crooped_image = cv2.cvtColor(crooped_image, cv2.COLOR_BGR2RGB)

        # breakpoint()
        # Apply augmentations using albumentations
        augmented = image_aug(image=crooped_image)
        crooped_image = augmented['image']


        # Randomly pick x, y coordinates for placing the image on the background
        if output_size[1] - row['w'] <= 0 or output_size[0] - row['h'] <= 0:
            continue # some situations we have to skip
        x = np.random.randint(0, output_size[1] - row['w'])
        y = np.random.randint(0, output_size[0] - row['h'])

        background = overlay_image_helper(background, crooped_image, x, y)
        applied_num_other_cnt += 1

    if applied_num_other_cnt == 0 or applied_num_other_cnt < num_other // 2:
        return

    label_idx = random.randint(0, 5)
    rand_select_label = labels[label_idx]
    if rand_select_label != 'Other':
        # Overlay non-'Other' label images on the black background
        # Filter the dataframe for non-'Other' label images
        label_df = df[df['label'] == rand_select_label].sample(n=num_non_other, random_state=generate_microsecond_seed())
        applied_num_non_other_cnt = 0

        # non_other_images = non_other_df.sample(n=num_non_other, random_state=generate_microsecond_seed())
        for _, row in label_df.iterrows():
            crooped_image = cv2.imread(row['file_path'])
            crooped_image = cv2.resize(crooped_image, (row['w'], row['h']))
            crooped_image = cv2.cvtColor(crooped_image, cv2.COLOR_BGR2RGB)

            # Randomly pick x, y coordinates for placing the image on the background
            if output_size[1] - row['w'] <= 0 or output_size[0] - row['h'] <= 0:
                continue # some situations we have to skip
            x = np.random.randint(0, output_size[1] - row['w'])
            y = np.random.randint(0, output_size[0] - row['h'])

            background = overlay_image_helper(background, crooped_image, x, y)
            applied_num_non_other_cnt += 1
        if applied_num_non_other_cnt == 0:
            return

    save_path = os.path.join(save_dir, f"{rand_select_label}_{idx}.jpg")
    cv2.imwrite(save_path, background)
