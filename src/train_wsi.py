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
from albumentations.pytorch import ToTensorV2

# For colored terminal text
from PIL import Image
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.plugins import CheckpointIO
from pytorch_lightning.utilities import rank_zero_only
from sconf import Config

# Training Function
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

# Sklearn Imports
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

def generate_microsecond_seed():
    """
    Generate a random seed based on the current time in microseconds.
    """
    microseconds = int(time.time() * 1e6)
    seed = microseconds % (2**32)  # Ensure that the seed is within the valid range
    return seed


def linear_scheduler(optimizer, warmup_steps, training_steps, last_epoch=-1):
    """linear_scheduler with warmup from huggingface"""

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(training_steps - current_step)
            / float(max(1, training_steps - warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def cosine_scheduler(
    optimizer, warmup_steps, training_steps, cycles=0.5, last_epoch=-1
):
    """Cosine LR scheduler with warmup from huggingface"""

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = current_step - warmup_steps
        progress /= max(1, training_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * cycles * 2 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def multistep_scheduler(optimizer, warmup_steps, milestones, gamma=0.1, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # calculate a warmup ratio
            return current_step / max(1, warmup_steps)
        else:
            # calculate a multistep lr scaling ratio
            idx = np.searchsorted(milestones, current_step)
            return gamma**idx

    return LambdaLR(optimizer, lr_lambda, last_epoch)


@rank_zero_only
def save_config_file(config, data_transforms, save_path):
    if not Path(save_path).exists():
        os.makedirs(save_path)
    config_save_path = Path(save_path) / "config.yaml"
    train_data_transform_save_path = Path(save_path) / "train_data_transform.json"
    valid_data_transform_save_path = Path(save_path) / "valid_data_transform.json"
    A.save(data_transforms["train"], train_data_transform_save_path)
    A.save(data_transforms["valid"], valid_data_transform_save_path)

    print(config.dumps())
    with open(config_save_path, "w") as f:
        f.write(config.dumps(modified_color=None, quote_str=True))
        print(f"Config is saved at {config_save_path}")


# Class weights
def compute_class_weights(df, label_column):
    """
    Compute class weights based on the inverse of class frequencies.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        label_column (str): Name of the column containing class labels.

    Returns:
        class_weights (dict): Dictionary containing weights for each class.
    """
    # Get the total number of samples
    total_samples = len(df)
    # Get the number of classes
    num_classes = df[label_column].nunique()
    # Get the count of each class
    class_counts = df[label_column].value_counts().to_dict()
    # Compute class weights
    class_weights = {
        class_label: total_samples / (num_classes * count)
        for class_label, count in class_counts.items()
    }

    return class_weights


class UBCDataset(Dataset):
    def __init__(self, df, label2idx, transforms=None):
        self.df = df
        self.file_names = df["file_path"].values
        self.labels = df["label"].values
        self.transforms = transforms
        self.label2idx = label2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label_idx = self.label2idx[self.labels[index]]

        if self.transforms:
            img = self.transforms(image=img)["image"]
        return_dict = {"image": img, "label": torch.tensor(label_idx, dtype=torch.long)}
        return return_dict


class UBCDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_batch_size = self.cfg.train.train_batch_size
        self.val_batch_size = self.cfg.train.val_batch_size
        self.train_dataset = None
        self.valid_dataset = None
        self.num_classes = None

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.cfg.train.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.cfg.train.num_workers,
            shuffle=False,
            pin_memory=True,
        )


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
    def __init__(self, cfg):
        super(TumorClassifier, self).__init__()
        self.cfg = cfg
        self.model = timm.create_model(self.cfg.model.name, pretrained=self.cfg.model.pretrained)

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.linear = nn.Linear(in_features, self.cfg.model.num_classes)
        self.loss_fct = nn.CrossEntropyLoss(
            weight=cfg.class_weights_tensor.to(self.cfg.train.accelerator)
        )


    def forward(self, images, labels=None):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        logits = self.linear(pooled_features)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return logits, loss

    def load_weight(self, ckpt_path: str):
        weight = torch.load(ckpt_path)['state_dict']
        for key in weight.copy().keys():
            val = weight.pop(key)
            new_key = key[6:]
            weight[new_key] = val
        self.load_state_dict(weight)



class UBCModelModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = None

        self.optimizer_types = {
            "sgd": SGD,
            "adam": Adam,
            "adamw": AdamW,
        }
        # self.arch = self.model.pretrained_cfg.get("architecture")
        self.num_classes = self.cfg.model.num_classes
        self.train_accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_f1_score = MulticlassF1Score(num_classes=self.num_classes)
        self.val_step_outputs = []

    def training_step(self, batch, batch_idx, *args):
        images = batch["image"]
        labels = batch["label"]

        logits, loss = self.model(images=images, labels=labels)

        # softmax along the class dimension (dim = 1, column dimension)
        pred_labels = torch.argmax(F.softmax(logits, dim=1), dim=1)
        self.log("train_loss", loss, logger=True, prog_bar=True)
        # self.log("train_acc", self.train_accuracy(pred_labels, labels), logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, *args):
        images = batch["image"]
        labels = batch["label"]
        logits, loss = self.model(images=images, labels=labels)
        pred_labels = torch.argmax(F.softmax(logits, dim=1), dim=1)

        step_out = {
            "pred_labels": pred_labels,
            "labels": labels,
        }
        self.val_step_outputs.append(step_out)
        self.log("val_loss", loss, logger=True, prog_bar=True)

        return step_out

    def on_validation_epoch_end(self):
        pred_labels = torch.cat(
            [step_out["pred_labels"] for step_out in self.val_step_outputs], dim=0
        )
        labels = torch.cat(
            [step_out["labels"] for step_out in self.val_step_outputs], dim=0
        )
        self.log(
            "val_acc",
            self.val_accuracy(pred_labels, labels),
            logger=True,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            "val_f1",
            self.val_f1_score(pred_labels, labels),
            logger=True,
            sync_dist=True,
            prog_bar=True,
        )

        self.val_step_outputs.clear()

    def _get_optimizer(self):
        opt_cfg = self.cfg.optimizer
        method = opt_cfg.method.lower()

        if method not in self.optimizer_types:
            raise ValueError(f"Unknown optimizer method={method}")

        kwargs = dict(opt_cfg.params)
        kwargs["params"] = self.model.parameters()
        optimizer = self.optimizer_types[method](**kwargs)

        return optimizer

    def _get_lr_scheduler(self, optimizer):
        # scheduler = torch.optim.lr_scheduler.CyclicLR(
        #     optimizer,
        #     base_lr=self.learn_rate,
        #     max_lr=self.learn_rate * 10,
        #     step_size_up=10,
        #     cycle_momentum=False,
        #     mode="triangular2",
        #     verbose=True,
        # )

        # return scheduler

        # cfg_train = self.cfg.train
        lr_schedule_method = self.cfg.optimizer.lr_schedule.method
        lr_schedule_params = self.cfg.optimizer.lr_schedule.params
        train_cfg = self.cfg.train

        if lr_schedule_method is None:
            scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1)
        elif lr_schedule_method == "step":
            scheduler = multistep_scheduler(optimizer, **lr_schedule_params)
        elif lr_schedule_method == "cosine":
            total_samples = train_cfg.max_epochs * train_cfg.num_samples_per_epoch
            total_batch_size = train_cfg.train_batch_size * self.trainer.world_size
            max_iter = total_samples / total_batch_size
            scheduler = cosine_scheduler(
                optimizer, training_steps=max_iter, **lr_schedule_params
            )
        elif lr_schedule_method == "linear":
            total_samples = train_cfg.max_epochs * train_cfg.num_samples_per_epoch
            total_batch_size = train_cfg.train_batch_size * self.trainer.world_size
            max_iter = total_samples / total_batch_size
            scheduler = linear_scheduler(
                optimizer, training_steps=max_iter, **lr_schedule_params
            )
        elif lr_schedule_method == "cyclic_lr":
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=self.cfg.optimizer.params.lr,
                max_lr=self.cfg.optimizer.params.lr * 10,
                # step_size_up=10,
                cycle_momentum=False,
                mode="triangular2",
                verbose=True,
            )
        else:
            raise ValueError(f"Unknown lr_schedule_method={lr_schedule_method}")

        return scheduler

    def configure_optimizers(self):
        optimizer = self._get_optimizer()
        scheduler = self._get_lr_scheduler(optimizer)
        scheduler = {
            "scheduler": scheduler,
            "name": "learning_rate",
            "interval": "step",
        }
        return [optimizer], [scheduler]

def setup_dataframe(cfg):

    def fetch_csvs(csv_paths):
        if isinstance(csv_paths, list):
            csvs = []
            for csv_path in csv_paths:
                csvs.append(pd.read_csv(csv_path).reset_index(drop=True))
            df = pd.concat(csvs).reset_index(drop=True)
            # if cfg.data.fold value is none we use all additional data for training
            # (fold -> used for validation, others -> used for training)
            # df['kfold'].fillna(cfg.data.fold+1)
            return df

        elif isinstance(cfg.data.train_csv, str):
            df = pd.read_csv(cfg.data.train_csv)
            return df
        else:
            raise ValueError(f"Unknown type for cfg.data.train_csv={cfg.data.train_csv}")

    df_train = None
    df_valid = None

    # case 1 : both train & val set predefined
    if cfg.data.get('train_csv') and cfg.data.get('valid_csv'):
        df_train = fetch_csvs(cfg.data.train_csv)
        df_valid = fetch_csvs(cfg.data.valid_csv)


    elif cfg.data.get('train_csv') and not cfg.data.get('valid_csv'):
        df = fetch_csvs(cfg.data.train_csv)
        if cfg.data.get('create_new_fold'): # Create Folds
            skf = StratifiedKFold(n_splits=cfg.data.n_fold)
            for fold, (_, val_) in enumerate(skf.split(X=df, y=df.label)):
                df.loc[val_, "kfold"] = int(fold)
            df_train = df[df.kfold != cfg.data.fold].reset_index(drop=True)
            df_valid = df[df.kfold == cfg.data.fold].reset_index(drop=True)
        else:
            try:
                df_train = df[df.kfold != cfg.data.fold].reset_index(drop=True)
                df_valid = df[df.kfold == cfg.data.fold].reset_index(drop=True)
            except:
                raise Exception("No kfold column in the train.csv or valid_csv is not defined")


    if cfg.debug:
        df_train = df_train.sample(n=min(200, len(df_train)), random_state=cfg.seed).reset_index(drop=True)
        df_valid = df_valid.sample(n=min(300, len(df_valid)), random_state=cfg.seed).reset_index(drop=True)


    print(f"Dataset/train size: {len(df_train)}")
    print(f"Dataset/validation size: {len(df_valid)}")
    print(df_train.head())
    return df_train, df_valid

def train(cfg):
    pl.seed_everything(cfg.seed)

    df_train, df_valid = setup_dataframe(cfg)

    # save the dataframe
    if cfg.data.save_dataframe:
        save_train_csv_path = Path(cfg.result_path) / cfg.exp_name / cfg.exp_version / "train.csv"
        save_valid_csv_path = Path(cfg.result_path) / cfg.exp_name / cfg.exp_version / "valid.csv"
        df_train.to_csv(save_train_csv_path, index=False)
        df_valid.to_csv(save_valid_csv_path, index=False)

    # set class weights
    class_weights: Dict[str, float] = compute_class_weights(df_train, "label")
    # class_weights = {cfg.data.labels[idx]:weight for idx, weight in class_weights.items()}
    print(f"Class weights\n: {class_weights}")
    if "Other" not in class_weights:
        class_weights['Other'] = 1.0
    class_weights: List[float] = [class_weights[label] for label in cfg.data.labels]
    if cfg.debug:
        class_weights = [1,1,1,1,1,1]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    cfg.class_weights_tensor = (
        class_weights_tensor  # need this for loss function in model forward)
    )

    labels = cfg.data.labels
    label2label_idx = {label: idx for idx, label in enumerate(labels)}
    train_dataset = UBCDataset(df_train, label2label_idx, transforms=cfg.data_transforms["train"])
    valid_dataset = UBCDataset(df_valid, label2label_idx, transforms=cfg.data_transforms["valid"])

    print(f"training number of images: {len(train_dataset)}")
    print(f"validation number of images: {len(valid_dataset)}")

    data_module = UBCDataModule(cfg)

    data_module.train_dataset = train_dataset
    data_module.valid_dataset = valid_dataset
    data_module.num_classes = len(labels)

    cfg.train.num_samples_per_epoch = len(train_dataset)
    model_module = UBCModelModule(cfg)
    model_module.model = TumorClassifier(cfg)
    if cfg.model.checkpoint_path:
        model_module.model.load_weight(cfg.model.checkpoint_path)

    # csv_logger = pl.loggers.CSVLogger(save_dir=cfg.log_dir, name=model_module.arch)
    tensorboard_logger = TensorBoardLogger(
        save_dir=cfg.result_path,
        name=cfg.exp_name,
        version=cfg.exp_version,
        sub_dir="logs",
        default_hp_metric=False,
    )

    lr_callback = LearningRateMonitor(logging_interval="step")
    # early_stop_callback = EarlyStopping(
    #     monitor="val_loss", min_delta=0.00, patience=5, verbose=True, mode="min"
    # )
    model_summary_callback = ModelSummary(max_depth=4)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath=Path(cfg.result_path) / cfg.exp_name / cfg.exp_version / "ckpts",
        filename=f"{cfg.model.name}" + "-{epoch:02d}-{step}-{val_acc:.4f}",
        save_top_k=30,  # if you save more than 1 model,
        # then checkpoint and huggingface model are not guaranteed to be matching
        # because we are saving with huggingface model with save_pretrained method
        # in "on_save_checkpoint" method in "BROSModelPLModule"
        mode="max",
        save_weights_only=True,
    )

    # ==============================

    trainer = pl.Trainer(
        # fast_dev_run=True,
        accelerator=cfg.train.accelerator,
        logger=[tensorboard_logger],
        precision=cfg.model.precision,  # is it the best?
        # accumulate_grad_batches=14,
        val_check_interval=cfg.train.val_check_interval,
        callbacks=[
            checkpoint_callback,
            lr_callback,
            model_summary_callback,
            # early_stop_callback,
        ],
        max_epochs=cfg.train.max_epochs,
        num_sanity_val_steps=3,
        gradient_clip_val=cfg.train.clip_gradient_value,
        gradient_clip_algorithm=cfg.train.clip_gradient_algorithm,
    )

    # trainer.fit(model=model_module, datamodule=data_module, ckpt_path=cfg.model.checkpoint_path)
    trainer.fit(model=model_module, datamodule=data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exp_version", type=str, required=False)
    args, left_argv = parser.parse_known_args()

    config = Config(args.config)
    config.argv_update(left_argv)

    config.exp_name = os.path.basename(args.config).split(".")[0]
    config.exp_version = (
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not args.exp_version
        else args.exp_version
    )

    img_height, img_width = config.model.input_image_size

    color_mean = [0.80584254, 0.67628579, 0.81574082]
    color_std = [0.09395851, 0.11092248, 0.08021936]

    # below is imagenet mean and std default
    # color_mean = [0.485, 0.456, 0.406]
    # color_std = [0.229, 0.224, 0.225]


    # Augmentations
    data_transforms = {
        "train": A.Compose(
            [
                A.Resize(img_height, img_width),
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
                A.RandomResizedCrop(  # Random crop and resize
                    height=img_height,
                    width=img_width,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    p=0.5,
                ),
                A.ChannelShuffle(
                    p=0.1
                ),  # Randomly shuffle channels to change color perspectives
                A.CLAHE(
                    clip_limit=2, tile_grid_size=(8, 8), p=0.3
                ),  # Apply CLAHE for enhancing contrast
                A.Normalize(
                    mean=color_mean,
                    std=color_std,
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ],
            p=1.0,
        ),
        "valid": A.Compose(
            [
                A.Resize(img_height, img_width),
                A.Normalize(
                    mean=color_mean,
                    std=color_std,
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ],
            p=1.0,
        ),
    }

    save_config_file(
        config=config,
        data_transforms=data_transforms,
        save_path=Path(config.result_path) / config.exp_name / config.exp_version,
    )

    config.data_transforms = data_transforms
    train(config)
