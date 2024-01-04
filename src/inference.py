import copy
import gc
import glob
import math
import os
import random
import shutil
import time
import warnings
from collections import defaultdict, Counter
from pprint import pprint
from typing import Dict, List, Tuple, Union, Optional

# Albumentations for augmentations
import albumentations as A
import cv2

# Utils
import joblib

# For data manipulation
import numpy as np
import pandas as pd
import pyvips

# For Image Models
import timm

# Pytorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from albumentations.pytorch import ToTensorV2

# For colored terminal text
from colorama import Back, Fore, Style
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.model_selection import StratifiedKFold

# Sklearn Imports
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

os.environ["VIPS_CONCURRENCY"] = "4"
os.environ["VIPS_DISC_THRESHOLD"] = "15gb"



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
    tma_tile_stride = 256
    # tma_tile_stride = 256
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
    run = 'tma' # 'tma', 'wsi', 'tma_wsi'

    dataset_dir = "/kaggle/input/UBC-OCEAN/"
    image_dir = os.path.join(dataset_dir, f"{mode}_images")
    thumbnail_dir = os.path.join(dataset_dir, f"{mode}_thumbnails")

    tile_folder = "./tmp_tiles"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    num_workers = 4

    tma_config = TMAConfig
    wsi_config = WSIConfig

# sanity check
assert os.path.exists(Config.tma_config.tma_model_weight), "Please check if the TMA model weight exists"
assert os.path.exists(Config.wsi_config.tumor_classifier_model_weight), "Please check if the WSI tumor classifier model weight exists"
assert os.path.exists(Config.wsi_config.tumor_subtype_classifier_model_weight), "Please check if the WSI tumor subtype classifier model weight exists"

if Config.debug:
    print("-------------------------- DEBUG MODE IS ON --------------------------")
    Config.mode = "train"
    Config.image_dir = os.path.join(Config.dataset_dir, f"{Config.mode}_images")
    Config.thumbnail_dir = os.path.join(
        Config.dataset_dir, f"{Config.mode}_thumbnails"
    )

def is_valid_thumbnail_tile(tile, black_bg_thr, white_bg_thr, white_thr) -> bool:
    black_bg_mask = np.sum(tile, axis=2) == 0
    if np.sum(black_bg_mask) >= (np.prod(black_bg_mask.shape) * black_bg_thr):
        return False

    white_bg_mask = np.mean(tile, axis=2) > white_thr
    if np.sum(white_bg_mask) >= (np.prod(white_bg_mask.shape) * white_bg_thr):
        return False

    return True

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

def center_crop_and_extract_tiles(
    img_path,
    center_crop_size=2048,
    tile_size: int = 2048,
    scale: float = 0.5,
    drop_thr: float = 0.6,
    white_thr: int = 240,
    stride: int = 512,  # Add stride parameter with a default value
) -> list[Image]:
    image_id, _ = os.path.splitext(os.path.basename(img_path))
    img = pyvips.Image.new_from_file(img_path)

    # Calculate the cropping box
    # subtract crop_size / 2 from center
    img_width, img_height = img.width, img.height
    new_size = (int(tile_size * scale), int(tile_size * scale))

    images = []
    # when img is small then crop size
    if img_width < tile_size or img_height < tile_size:
        cur_tile = Image.open(img_path).resize(new_size, Image.LANCZOS)
        images.append(cur_tile)
        # Image.open(img_path).resize(new_size, Image.LANCZOS).save(new_img_path)
        return images

    # Calculate the cropping box
    # subtract crop_size / 2 from center
    left = max(0, int((img_width - center_crop_size) / 2))
    top = max(0, int((img_height - center_crop_size) / 2))
    img = img.crop(left, top, min(center_crop_size, img_width), min(center_crop_size, img_height))
    img_width, img_height = min(center_crop_size, img_width), min(center_crop_size, img_height)

    tile_width = tile_height = tile_size

    idxs = [
        (y, y + tile_height, x, x + tile_width)
        for y in range(
            0, img.height - tile_height + 1, stride
        )  # Include stride in the y range
        for x in range(
            0, img.width - tile_width + 1, stride
        )  # Include stride in the x range
    ]
    # random.shuffle(idxs)
    images = []
    for y, y_, x, x_ in idxs:
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
            images.append(Image.fromarray(tile).resize(new_size, Image.LANCZOS))
        except:
            print(f"center_crop_and_extract_tiles, Error in extracting tile from {img_path} at {x=}, {y=}, {tile_width=}, {tile_height=}")


    return images

def extract_tiles_from_wsi_thumbnail(
    img_path,
    tile_size: int = 200,
    tile_stride: int = 100,
    tile_scale: float = 1.0,
    white_thr: int = 240,
    black_bg_thr: float = 0.6,
    white_bg_thr: float = 0.8,
) -> list[Image]:


    image_id, _ = os.path.splitext(os.path.basename(img_path))
    img = pyvips.Image.new_from_file(img_path)

    # Calculate the cropping box
    # subtract crop_size / 2 from center
    img_width, img_height = img.width, img.height
    new_size = (int(tile_size * tile_scale), int(tile_size * tile_scale))
    tile_width = tile_height = tile_size

    idxs = [
        (y, y + tile_height, x, x + tile_width)
        for y in range(
            0, img.height - tile_height + 1, tile_stride
        )  # Include stride in the y range
        for x in range(
            0, img.width - tile_width + 1, tile_stride
        )  # Include stride in the x range
    ]
    # random.shuffle(idxs)
    images = []
    image_infos = []

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

            if is_valid_thumbnail_tile(tile, black_bg_thr, white_bg_thr, white_thr):
                images.append(Image.fromarray(tile).resize(new_size, Image.LANCZOS))
                image_infos.append(f"{image_id}_xywh_{x}_{y}_{tile_width}_{tile_height}")

        except:
            print(f"extract_tiles_from_wsi_thumbnail, Error in extracting tile from {img_path, img_width, img_height} at {x=}, {y=}, {tile_width=}, {tile_height=}")
    return images, image_infos

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


color_mean = [0.80584254, 0.67628579, 0.81574082]
color_std = [0.09395851, 0.11092248, 0.08021936]

VALID_TRANSFORM = {
    "tma": T.Compose(
        [
            T.Resize(512),
            T.ToTensor(),
            T.Normalize(color_mean, color_std),
        ]
    ),
    "wsi_thumbnail": T.Compose(
        [
            T.Resize(200),
            T.ToTensor(),
            T.Normalize(color_mean, color_std),
        ]
    ),
}

class TilesDataset(Dataset):
    def __init__(self, images: List[Image] , transforms):
        self.transforms = transforms
        self.images = images

    def __getitem__(self, idx: int) -> tuple:
        img = self.images[idx]
        if self.transforms:
            img = self.transforms(img)

        return img

    def __len__(self) -> int:
        return len(self.images)


class CancerSubtypeClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super(CancerSubtypeClassifier, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = timm.create_model(self.model_name, num_classes=self.num_classes)

    def forward(self, images):
        logits = self.model(images)
        return logits

    def load_weight(self, ckpt_path: str):
        weight = torch.load(ckpt_path)['state_dict']
        for key in weight.copy().keys():
            val = weight.pop(key)
            new_key = key[6:]
            weight[new_key] = val

        if 'loss_fct.weight' in weight:
            weight.pop("loss_fct.weight")

        self.load_state_dict(weight)
        # tma_model.load_state_dict(weight)
        # tma_model.to(Config.device)

# GeM pooling
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class TumorClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super(TumorClassifier, self).__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.model = timm.create_model(self.model_name)

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.linear = nn.Linear(in_features, self.num_labels)


    def forward(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        logits = self.linear(pooled_features)

        return logits

    def load_weight(self, ckpt_path: str):
        weight = torch.load(ckpt_path)['state_dict']
        for key in weight.copy().keys():
            val = weight.pop(key)
            new_key = key[6:]
            weight[new_key] = val
        if 'loss_fct.weight' in weight:
            weight.pop("loss_fct.weight")

        self.load_state_dict(weight)

def inference_tma(df: pd.DataFrame):
    """
    1. get tiles tma image
    2. predict tiles with tma model
    3. collect the result from step 2 and make final decision
    """

    tma_config = Config.tma_config

    tma_model = CancerSubtypeClassifier(tma_config.tma_model_name, tma_config.num_classes)
    if tma_config.tma_model_weight:
        tma_model.load_weight(tma_config.tma_model_weight)
    tma_model.eval()
    tma_model = tma_model.cuda()

    for i, row in df.iterrows():
        if not row['is_tma']: continue
        row = dict(row)

        image_id = str(row["image_id"])
        img_path = os.path.join(Config.image_dir, f"{image_id}.png")

        tiles = center_crop_and_extract_tiles(
            img_path,
            center_crop_size=tma_config.tma_center_crop_size ,
            tile_size=tma_config.tma_tile_size,
            scale=tma_config.tma_tile_scale,
            drop_thr=tma_config.tma_tile_drop_thr,
            stride=tma_config.tma_tile_stride,
            white_thr=tma_config.tma_tile_white_thr,
        )

        dataset = TilesDataset(tiles, transforms=VALID_TRANSFORM['tma'])
        print(f"found tiles: {len(dataset)}")
        if not len(dataset):
            print(f"seem no tiles were cut for `{img_path}`")
            continue

        dataloader = DataLoader(dataset, batch_size=Config.batch_size, num_workers=Config.num_workers, shuffle=False)

        # iterate over images and collect predictions
        preds = []
        for imgs in dataloader:
            # print(f"{imgs.shape}")
            with torch.no_grad():
                pred = tma_model(imgs.cuda())
            preds += pred.cpu().numpy().tolist()

        if tma_config.tma_decision_rule == 'vote':
            pred_labels = torch.argmax(F.softmax(torch.tensor(preds), dim=1), dim=1).tolist()
            counter = Counter(pred_labels)
            print_order = [k for k, v in counter.most_common()]
            print_counter = [(tma_config.tma_labels[k], counter[k]) for k in print_order]
            print(f'tma result - counter: {print_counter}')

            # all tiles been predicted as same label, then use that label
            final_label = counter.most_common(1)[0][0]
            final_label = tma_config.tma_labels[final_label]
            df.at[i, 'label'] = final_label

        elif tma_config.tma_decision_rule == 'sum':
            # print(f"Sum contrinution from all tiles: {np.sum(preds, axis=0)}")
            # print(f"Max contribution over all tiles: {np.max(preds, axis=0)}")

            # decide label
            pred_label = np.argmax(np.sum(preds, axis=0))
            final_label = tma_config.tma_labels[pred_label]
            df.at[i, 'label'] = final_label
        else:
            raise ValueError(f"unknown decision rule: {tma_config.tma_decision_rule}")

        print(f"image_id: {image_id}, predicted label : {final_label}")

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


def inference_wsi(df: pd.DataFrame):
    """
    1. get tiles from wsi thumbnail image
    2. predict tiles with is_tumor model
    3. if we find tumor tile from step 2, then we extract tiles from original wsi image
    4. predict tiles from step 3 with tma model
    5. collect the result from step 4 and make final decision
    """


    wsi_config = Config.wsi_config

    # load tumor classifier models for 1st phase
    tumor_classifier = TumorClassifier(wsi_config.tumor_classifier, wsi_config.tumors_classifier_num_classes)
    if wsi_config.tumor_classifier_model_weight:
        tumor_classifier.load_weight(wsi_config.tumor_classifier_model_weight)
    tumor_classifier.eval()
    tumor_classifier = tumor_classifier.cuda()

    # load cancer subtype classifier models for 2nd phase
    tumor_subtype_classifier = CancerSubtypeClassifier(
        wsi_config.tumor_subtype_classifier,
        wsi_config.tumor_subtype_classifier_num_classes
    )
    if wsi_config.tumor_subtype_classifier_model_weight:
        tumor_subtype_classifier.load_weight(wsi_config.tumor_subtype_classifier_model_weight)
    tumor_subtype_classifier.eval()
    tumor_subtype_classifier = tumor_subtype_classifier.cuda()

    for i, row in df.iterrows():
        if row['is_tma']: continue # want to skip tma images, and learn WSI only

        image_id = str(row["image_id"])
        image_width = row["image_width"]
        image_height = row["image_height"]
        wsi_thumbnail_image_path = os.path.join(Config.thumbnail_dir, f"{image_id}_thumbnail.png") # use thumbnail image for WSI
        wsi_image_path = os.path.join(Config.image_dir, f"{image_id}.png")
        thumbnail_width, thumbnail_height = Image.open(wsi_thumbnail_image_path).size

        thumbnail_tile_images, thumbnail_infos = extract_tiles_from_wsi_thumbnail(
            img_path=wsi_thumbnail_image_path,
            tile_size=wsi_config.thumbnail_tile_size,
            tile_stride=wsi_config.thumbnail_tile_stride,
            tile_scale=wsi_config.thumbnail_tile_scale,
            white_thr=wsi_config.thumbnail_tile_white_thr,
            black_bg_thr=wsi_config.thumbnail_tile_black_bg_thr,
            white_bg_thr=wsi_config.thumbnail_tile_white_bg_thr,
        )

        dataset = TilesDataset(thumbnail_tile_images, transforms=VALID_TRANSFORM['wsi_thumbnail'])

        print(f"1st phase - found {len(dataset)} tiles from wsi thumbnail images")
        if not len(dataset):
            print(f"1st phase - seem no tiles were cut for `{wsi_thumbnail_image_path}`")
            submission.append(row)
            continue

        dataloader = DataLoader(dataset, batch_size=Config.batch_size, num_workers=Config.num_workers, shuffle=False)

        # iterate over images and collect predictions
        preds = []
        for imgs in dataloader:
            # print(f"{imgs.shape}")
            with torch.no_grad():
                pred = tumor_classifier(imgs.cuda())
            preds += pred.cpu().numpy().tolist()

        pred_labels = np.argmax(preds, axis=1)

        if np.sum(pred_labels == 0) == 0:
            # 0 -> tumor label index, if np.sum(pred_labels == 0) is 0 that means no tumor tile were found
            print(f"1st phase - seem no tumor tile were found for `{wsi_thumbnail_image_path}`")
            df.at[i, 'label'] = "Other"
            continue

        # second phase, get tiles from original WSI image and predict with TMA model

        # image_width -> thumbnail_width
        ori_width2thumbnail_width_ratio = image_width / thumbnail_width
        # image_height -> thumbnail_height
        ori_height2thumbnail_height_ratio = image_height / thumbnail_height

        pred_label_and_thumbnail_tile_info_list = list(zip(pred_labels, thumbnail_infos))
        thumbnail_tumor_tile_info_list = [tile_info for label, tile_info in pred_label_and_thumbnail_tile_info_list if label == 0]

        tumor_tile_info_list = []
        tmp_tumor_tile_info_list = []
        for thumbnail_tile_info in thumbnail_tumor_tile_info_list:
            thumbnail_tile_x, thumbnail_tile_y, thumbnail_tile_w, thumbnail_tile_h = list(map(int, thumbnail_tile_info.split("_")[-4:]))

            wsi_tile_x = int(thumbnail_tile_x * ori_width2thumbnail_width_ratio)
            wsi_tile_y = int(thumbnail_tile_y * ori_height2thumbnail_height_ratio)
            wsi_tile_w = int(thumbnail_tile_w * ori_width2thumbnail_width_ratio)
            wsi_tile_h = int(thumbnail_tile_h * ori_height2thumbnail_height_ratio)
            tumor_tile_info_list.append((wsi_tile_x, wsi_tile_y, wsi_tile_w, wsi_tile_h))
            # thumbnail_tumor_tile_info_list.append((thumbnail_tile_x, thumbnail_tile_y, thumbnail_tile_w, thumbnail_tile_h))
            tmp_tumor_tile_info_list.append((thumbnail_tile_x, thumbnail_tile_y))

        print(f"1st phase - found {len(tumor_tile_info_list)} tumor tiles from wsi thumbnail images")

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

        dataset = TilesDataset(wsi_tumor_tiles, transforms=VALID_TRANSFORM['tma'])
        if len(dataset) == 0:
            # 0 -> tumor label index, if np.sum(pred_labels == 0) is 0 that means no tumor tile were found
            print(f"2nd phase - weird situation, some tiles were found in thumbnail but not in original WSI image, {image_id}")
            df.at[i, 'label'] = "Other"
            continue
        print(f"2nd phase - extracted {len(dataset)} tiles from wsi tumor tiles")

        dataloader = DataLoader(dataset, batch_size=Config.batch_size, num_workers=Config.num_workers, shuffle=False)

        # iterate over images and collect predictions
        preds = []
        for imgs in dataloader:
            # print(f"{imgs.shape}")
            with torch.no_grad():
                pred = tumor_subtype_classifier(imgs.cuda())
            preds += pred.cpu().numpy().tolist()

        pred_labels = torch.argmax(F.softmax(torch.tensor(preds), dim=1), dim=1).tolist()
        # pred_labels = [wsi_config.tma_labels[pred_label_index] for pred_label_index in pred_label_indices]

        counter = Counter(pred_labels)
        print_order = [k for k, v in counter.most_common()]
        print_counter = [(wsi_config.tumor_subtype_classifier_classes[k], counter[k]) for k in print_order]
        print(f'2nd phase - counter: {print_counter}')

        # all tiles been predicted as same label, then use that label
        if len(counter) == 1:
            final_label = counter.most_common(1)[0][0]
            df.at[i, 'label'] = wsi_config.tumor_subtype_classifier_classes[final_label]
        else:
            # otherwise, use the label with highest count unless it's 'Other
            candidates = counter.most_common(2)
            first_cand, second_cand = candidates[0][0], candidates[1][0]
            final_label = first_cand if first_cand != wsi_config.tumor_subtype_class2idx['Other'] else second_cand
            df.at[i, 'label'] = wsi_config.tumor_subtype_classifier_classes[final_label]

        image_id = int(image_id)
        final_label = wsi_config.tumor_subtype_classifier_classes[final_label]
        print(f"2nd phase - {image_id, final_label = }")


        if i % 10 == 0 and i != 0:
            gc.collect()


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

# inference and make submission
if 'tma' in Config.run:
    inference_tma(df_test)

if 'wsi' in Config.run:
    inference_wsi(df_test)

print(df_test.head())
df_test[["image_id", "label"]].to_csv("submission.csv", index=False)
# ! head submission.csv
