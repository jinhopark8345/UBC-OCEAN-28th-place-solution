import argparse
import copy
import gc
import glob
import math
import os
import random
import shutil
import time
import warnings
from collections import defaultdict
from pprint import pprint
from typing import Dict, List, Tuple, Union

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
from sklearn.preprocessing import LabelEncoder
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

# pd.set_option("display.max_colwidth", None)
# pd.set_option("display.max_rows", None)

os.environ["VIPS_CONCURRENCY"] = "4"
os.environ["VIPS_DISC_THRESHOLD"] = "15gb"


class Config:
    debug = False
    mode = "test"
    tma_model_name = "maxvit_tiny_tf_512.in1k"
    tma_model_weight = ""
    num_classes = 6
    tma_labels = ["CC", "EC", "HGSC", "LGSC", "MC", "Other"]
    # tma_labels = ["tumor", "normal"]
    label2id = {label: i for i, label in enumerate(tma_labels)}
    id2label = {v: k for k, v in label2id.items()}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 180
    num_workers = 8

    inference_result_csv_path = "inference_result_tumor_non_tumor.csv"


parser = argparse.ArgumentParser()
parser.add_argument("--tma_model_weight", type=str, default=None)
parser.add_argument("--inference_result_csv_path", type=str, default=None)
args, left_argv = parser.parse_known_args()

if args.tma_model_weight:
    Config.tma_model_weight = args.tma_model_weight
if args.inference_result_csv_path:
    Config.inference_result_csv_path = args.inference_result_csv_path


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
}

class UBCDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.file_names = df["file_path"].values
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = Image.open(img_path)
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # label = self.labels[index]

        if self.transforms:
            img = self.transforms(img)
        return_dict = {"image": img}
        return return_dict

class CancerSubtypeClassifier(nn.Module):
    def __init__(self, cfg):
        super(CancerSubtypeClassifier, self).__init__()
        self.cfg = cfg
        self.model = timm.create_model(
            self.cfg.tma_model_name,
            pretrained=True,
            num_classes=cfg.num_classes,
        )

    def forward(self, images, labels=None):
        logits = self.model(images)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        # return logits, loss
        return logits, loss

    def load_weight(self, ckpt_path: str):
        weight = torch.load(ckpt_path)["state_dict"]
        for key in weight.copy().keys():
            val = weight.pop(key)
            new_key = key[6:]
            weight[new_key] = val

        if "loss_fct.weight" in weight:
            weight.pop("loss_fct.weight")

        self.load_state_dict(weight)


tma_model = CancerSubtypeClassifier(Config)
if Config.tma_model_weight:
    tma_model.load_weight(Config.tma_model_weight)

tma_model.eval()
tma_model = tma_model.cuda()
tma_model.half()

df = pd.read_csv("train_is_validate_logic_applied_without_inference_result.csv")

# df = df[:400] # for debugging
print(f"Dataset/test size: {len(df)}")
print("df_test.head(): ")
print(df.head())

test_dataset = UBCDataset(df, transforms=VALID_TRANSFORM["tma"])
test_loader = DataLoader(
    test_dataset,
    batch_size=Config.batch_size,
    num_workers=Config.num_workers,
    shuffle=False,
    pin_memory=False,
)


preds = []
with torch.no_grad():
    bar = tqdm(enumerate(test_loader), total=len(test_loader))
    for step, data in bar:
        images = data["image"].to(Config.device, dtype=torch.float)
        logits, _ = tma_model(images.half(), None)
        pred_labels = torch.argmax(F.softmax(logits, dim=1), dim=1).tolist()
        # _, predicted = torch.max(model.softmax(logits), 1)
        preds.append(pred_labels)

preds = np.concatenate(preds).flatten()

df["pred_label"] = [Config.id2label[pred_id] for pred_id in preds]
print(f"prediction result : {df}")

df.to_csv(Config.inference_result_csv_path, index=False)
