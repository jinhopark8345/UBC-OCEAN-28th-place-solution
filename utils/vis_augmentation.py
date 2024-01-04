# maxvit_base_tf_512.in21k_ft_in1k
import argparse
import copy
import datetime
import gc
import glob
import itertools
import json
import math
import multiprocessing as mproc
import os
import random
import re
import time
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple, Union

# Albumentations for augmentations
import albumentations as A
import cv2
# Utils
import joblib
import matplotlib.pyplot as plt
# For data manipulation
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sn
# For Image Models
import timm
# Pytorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import yaml
# from adan_pytorch import Adan
from albumentations import (Compose, HorizontalFlip, RandomCrop, Resize,
                            Rotate, VerticalFlip)
from albumentations.pytorch import ToTensorV2
# For colored terminal text
from colorama import Back, Fore, Style
# from lion_pytorch import Lion
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from overrides import overrides
from PIL import Image
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint, ModelSummary,
                                         TQDMProgressBar)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.plugins import CheckpointIO
from pytorch_lightning.utilities import rank_zero_only
from sconf import Config
# Training Function
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
# Sklearn Imports
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.cuda import amp
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch_optimizer import AdaBound, RAdam, Yogi
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from tqdm.auto import tqdm

# from torch_optimizer import AdaBound, RAdam, Yogi












def calculate_color_std_mean(img_path):
    # Load an image using OpenCV
    image = cv2.imread(img_path)  # Replace with the path to your image

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to a 2D array of pixels (height * width, channels)
    pixels = image_rgb.reshape(-1, 3)

    # Calculate the mean and standard deviation for each channel
    mean_values = pixels.mean(axis=0)
    std_values = pixels.std(axis=0)

    print("Mean Values (R, G, B):", mean_values)
    print("Std Deviation Values (R, G, B):", std_values)


def vis_aug(idx, original_image_path, image_aug, save_dir):
    """
    Visualize the original and augmented images side by side using OpenCV.
    """
    # Convert images to BGR format (OpenCV uses BGR by default)
    # original_image = Image.open(original_image_path)
    original_image = cv2.imread(original_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    augmented_image = image_aug(image=original_image)["image"]
    original_image_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)

    # Concatenate images horizontally
    result_image = np.concatenate([original_image_bgr, augmented_image_bgr], axis=1)

    save_path = os.path.join(save_dir, f"{idx}.jpg")
    cv2.imwrite(save_path, result_image)

def wrapper_func(args):
    return vis_aug(*args)


# # Create a random image for demonstration
# random_image = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)


# Define the image augmentation
# img_color_mean = [0.485, 0.456, 0.406]
# img_color_std = [0.229, 0.224, 0.225]
# img_color_mean = [0.8721593659261734, 0.7799686061900686, 0.8644588534918227]
# img_color_std = [0.08258995918115268, 0.10991684444009092, 0.06839816226731532]

# Mean Values (R, G, B): [202.3164444  167.75648117 207.67457199]
# Std Deviation Values (R, G, B): [27.44015397 24.72714561 20.39544822]
img_color_mean = [202.3164444, 167.75648117, 207.67457199]
img_color_mean = [e / 255 for e in img_color_mean]
img_color_std = [27.44015397, 24.72714561, 20.39544822]
img_color_std = [e / 255 for e in img_color_std]

# image_aug = A.Compose(
#     [
#         A.Normalize(
#             mean=img_color_mean,
#             std=img_color_std,
#             max_pixel_value=255.0,
#             p=1.0,
#         ),
#         # ToTensorV2(),
#     ],
#     p=1.0,
# )

class UBCDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.file_names = df["file_path"].values
        self.labels = df["label"].values
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.file_names[index]
        breakpoint()
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[index]

        if self.transforms:
            img = self.transforms(image=img)["image"]
        return_dict = {"image": img}
        return return_dict

img_height = 512
img_width = 512
image_transform = A.Compose(
            [
                # A.Resize(img_height, img_width),
                A.ShiftScaleRotate(
                    shift_limit=0.2, scale_limit=0.15, rotate_limit=90, p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,  # Increased limit for more pronounced color shifts
                    sat_shift_limit=0.25,  # Slightly higher saturation shift
                    val_shift_limit=0.25,  # Slightly higher value shift
                    p=0.5,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(
                        -0.15,
                        0.15,
                    ),  # Slightly wider range for brightness
                    contrast_limit=(
                        -0.15,
                        0.15,
                    ),  # Slightly wider range for contrast
                    p=0.5,
                ),
                # A.RandomResizedCrop(  # Random crop and resize
                #     height=img_height,
                #     width=img_width,
                #     scale=(0.8, 1.0),
                #     ratio=(0.9, 1.1),
                #     p=0.5,
                # ),
                # A.ChannelShuffle(
                #     p=0.1
                # ),  # Randomly shuffle channels to change color perspectives
                A.CLAHE(
                    clip_limit=2, tile_grid_size=(8, 8), p=0.3
                ),  # Apply CLAHE for enhancing contrast
                # A.Normalize(
                #     mean=[0.485, 0.456, 0.406],
                #     std=[0.229, 0.224, 0.225],
                #     # mean = img_color_mean,
                #     # std = img_color_std,
                #     max_pixel_value=255.0,
                #     p=1.0,
                # ),
                # ToTensorV2(),
            ],
            p=1.0,
)



# Apply the augmentation
# augmented_image = image_aug(image=random_image)["image"]

# Visualize the original and augmented images
# vis_aug(random_image, augmented_image, title="Image Augmentation")

# save_dir = "/kaggle/working/UBC-OCEAN-images/mask_crop_aug9"
# df = pd.read_csv("/kaggle/working/UBC-OCEAN-wsi-mask-cropped/train.csv").sample(frac=1).reset_index(drop=True)

df = pd.read_csv("/kaggle/working/UBC-OCEAN-masked_filtered_tiles-1024px-scale-0-50/train.csv").sample(n=1000).reset_index(drop=True)
save_dir = "/kaggle/working/UBC-OCEAN-images/tile_aug2"
os.makedirs(save_dir, exist_ok=True)
for idx, row in tqdm(df.iterrows(), total=len(df)):
    vis_aug(idx, row['file_path'], image_transform, save_dir)
# calculate_color_std_mean(df['file_path'][0])

# dataset = UBCDataset(df, image_transform)
# for idx in range(len(dataset)):
#     tmp = dataset[idx]['image']
#     breakpoint()



    # vis_aug(idx, UBC_dataset[idx]['image'], image_aug, save_dir)
# num_threads = 12
# pool = mproc.Pool(num_threads)
# tqdm_bar = tqdm(total=len(df))
# for _ in pool.imap_unordered(
#     wrapper_func,
#     (
#         (idx, row['file_path'], image_aug, save_dir)
#         for idx, row in enumerate(df.iterrows())
#     ),
# ):
#     tqdm_bar.update()
# pool.close()
# pool.join()


# imgs_path = df['file_path'].values

# rgb_values = np.concatenate(
#     [Image.open(img_path).getdata() for img_path in imgs_path],
#     axis=0
# ) / 255.

# # rgb_values.shape == (n, 3),
# # wh
# # and 3 are the 3 channels: R, G, B.

# # Each value is in the interval [0; 1]

# mu_rgb = np.mean(rgb_values, axis=0)  # mu_rgb.shape == (3,)
# std_rgb = np.std(rgb_values, axis=0)  # std_rgb.shape == (3,)

# print(f"{ mu_rgb, std_rgb = }")
