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
from collections import defaultdict, Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple, Union
import shutil

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
from pytorch_lightning.callbacks import ProgressBar

from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger

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


def pick_most_common(pred_labels,except_other=False, other_idx=5):
    counter = Counter(pred_labels)

    # all tiles been predicted as same label, then use that label
    if len(counter) == 1:
        final_label = counter.most_common(1)[0][0]
    else: # otherwise, use the label with highest count unless it's 'Other
        candidates = counter.most_common(2)
        first_cand, second_cand = candidates[0][0], candidates[1][0]

        if except_other:
            final_label = first_cand if first_cand != other_idx else second_cand
        else:
            final_label = first_cand
    return final_label

    # df.at[i, 'label'] = wsi_config.tumor_subtype_classifier_classes[final_label]

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
def compute_class_weights(dfs, label_column):
    """
    Compute class weights based on the inverse of class frequencies.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        label_column (str): Name of the column containing class labels.

    Returns:
        class_weights (dict): Dictionary containing weights for each class.
    """
    if isinstance(dfs, list):
        dfs = [d['df'] for d in dfs]
        dfs = pd.concat(dfs).reset_index(drop=True)
    # Get the total number of samples
    total_samples = len(dfs)
    # Get the number of classes
    num_classes = dfs[label_column].nunique()
    # Get the count of each class
    class_counts = dfs[label_column].value_counts().to_dict()
    # Compute class weights
    class_weights = {
        class_label: total_samples / (num_classes * count)
        for class_label, count in class_counts.items()
    }

    return class_weights


class ProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self.enable = True
        # self.config = config

    def disable(self):
        self.enable = False

    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        # items["exp_name"] = f"{self.config.get('exp_name', '')}"
        # items["exp_version"] = f"{self.config.get('exp_version', '')}"
        return items

class UBCDataset(Dataset):
    def __init__(self, df, label2idx, transforms=None, use_labels=True):
        self.df = df
        self.file_names = df["file_path"].values
        self.image_id = df['image_id'].values
        self.use_labels = use_labels

        if self.use_labels:
            self.labels = df["label"].values
        self.transforms = transforms
        self.label2idx = label2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.file_names[index]
        image_id = self.image_id[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        if self.transforms:
            img = self.transforms(image=img)["image"]

        return_dict = {
            "image": img,
            "image_id": torch.tensor(image_id, dtype=torch.long),
        }
        if self.use_labels:
            label_idx = self.label2idx[self.labels[index]]
            return_dict["label"] = torch.tensor(label_idx, dtype=torch.long)
        return return_dict


class UBCDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_batch_size = self.cfg.train.train_batch_size
        self.val_batch_size = self.cfg.train.val_batch_size
        self.train_datasets = None
        self.valid_datasets = None
        self.num_classes = None

    def train_dataloader(self):
        if isinstance(self.train_datasets, list):
            loaders = []
            for train_dataset in self.train_datasets:
                loaders.append(
                    DataLoader(
                        dataset=train_dataset,
                        batch_size=self.train_batch_size,
                        num_workers=self.cfg.train.num_workers,
                        shuffle=True,
                        pin_memory=True,
                    )
                )
        else:
            loaders = DataLoader(
                dataset=self.train_datasets,
                batch_size=self.train_batch_size,
                num_workers=self.cfg.train.num_workers,
                shuffle=True,
                pin_memory=True,
            )
        return loaders

    def val_dataloader(self):
        if isinstance(self.valid_datasets, list):
            loaders = []
            for valid_dataset in self.valid_datasets:
                loaders.append(
                    DataLoader(
                        dataset=valid_dataset,
                        batch_size=self.val_batch_size,
                        num_workers=self.cfg.train.num_workers,
                        shuffle=False,
                        pin_memory=True,
                    )
                )
        else:
            loaders = DataLoader(
                dataset=self.valid_datasets,
                batch_size=self.val_batch_size,
                num_workers=self.cfg.train.num_workers,
                shuffle=False,
                pin_memory=True,
            )

        return loaders

class CancerSubtypeClassifier(nn.Module):
    def __init__(self, cfg):
        super(CancerSubtypeClassifier, self).__init__()
        self.cfg = cfg
        self.model = timm.create_model(
            self.cfg.model.name,
            pretrained=self.cfg.model.pretrained,
            num_classes=self.cfg.model.num_classes,
        )

        self.loss_fct = nn.CrossEntropyLoss(
            weight=cfg.class_weights_tensor.to(self.cfg.train.accelerator)
        )

    def forward(self, images, labels=None):
        logits = self.model(images)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        # return logits, loss
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
        self.num_classes = self.cfg.model.num_classes
        self.train_accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_f1_score = MulticlassF1Score(num_classes=self.num_classes)
        self.val_accuracy2 = MulticlassAccuracy(num_classes=self.num_classes, average=None)

        self.val_step_outputs = defaultdict(list)
        self.train_dataset_idx2dataset_name = None
        self.valid_dataset_idx2dataset_name = None
        self.image_id2class_idx = None

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        images = batch["image"]
        labels = batch["label"]

        logits, loss = self.model(images=images, labels=labels)

        # softmax along the class dimension (dim = 1, column dimension)
        pred_labels = torch.argmax(F.softmax(logits, dim=1), dim=1)
        self.log("train_loss", loss, logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        dataset_name = self.valid_dataset_idx2dataset_name[dataloader_idx]

        if "by_image_id" in dataset_name:
            # special case : val acc per image_id
            # if "by_image_id" is in dataset_name -> individual image label is not always correct
            images = batch['image']
            image_id = batch['image_id']
            logits, _ = self.model(images=images)
            pred_labels = torch.argmax(F.softmax(logits, dim=1), dim=1)
            step_out = {
                "pred_labels": pred_labels,
                "image_id": image_id,
            }
        else: # default, val acc per image
            images = batch["image"]
            labels = batch["label"]
            logits, loss = self.model(images=images, labels=labels)
            pred_labels = torch.argmax(F.softmax(logits, dim=1), dim=1)
            step_out = {
                "pred_labels": pred_labels,
                "labels": labels,
            }
            self.log(f"{dataset_name}_val_loss", loss, logger=True, prog_bar=False, add_dataloader_idx=False)
        self.val_step_outputs[dataset_name].append(step_out)

        return step_out

    def on_validation_epoch_end(self):
        for dataset_name, val_step_output in self.val_step_outputs.items():

            if 'by_image_id' in dataset_name: # val acc per image id
                pred_labels = torch.cat([step_out["pred_labels"] for step_out in val_step_output], dim=0)
                image_ids = torch.cat([step_out["image_id"] for step_out in val_step_output], dim=0)
                image_id2pred_labels = defaultdict(list)
                for image_id, pred_label in zip(image_ids, pred_labels):
                    image_id2pred_labels[image_id.item()].append(pred_label.item())

                if "tumor_only" in dataset_name:
                    image_id2pred_final_label = {
                        image_id: pick_most_common(pred_labels, except_other=True) for image_id, pred_labels in image_id2pred_labels.items()
                    }
                else:
                    image_id2pred_final_label = {
                        image_id: pick_most_common(pred_labels, except_other=False) for image_id, pred_labels in image_id2pred_labels.items()
                    }

                labels, preds = [], []

                for image_id, pred_final_label in image_id2pred_final_label.items():
                    labels.append(self.image_id2class_idx[image_id])
                    preds.append(pred_final_label)

                labels = torch.tensor(labels, dtype=torch.long).to(self.device)
                preds = torch.tensor(preds, dtype=torch.long).to(self.device)

                self.log(
                    f"{dataset_name}_val_acc",
                    self.val_accuracy(preds, labels),
                    logger=True,
                    sync_dist=True,
                    prog_bar=True,
                    add_dataloader_idx=False,
                )
                self.log(
                    f"{dataset_name}_val_f1",
                    self.val_f1_score(preds, labels),
                    logger=True,
                    sync_dist=True,
                    prog_bar=False,
                    add_dataloader_idx=False,
                )
                print(f"{dataset_name} : {self.val_accuracy(preds, labels)}")
                print(f"{dataset_name} : {self.val_accuracy2(preds, labels)}")


            else: # default case : val_acc per image
                pred_labels = torch.cat([step_out["pred_labels"] for step_out in val_step_output], dim=0)
                labels = torch.cat([step_out["labels"] for step_out in val_step_output], dim=0)

                self.log(
                    f"{dataset_name}_val_acc",
                    self.val_accuracy(pred_labels, labels),
                    logger=True,
                    sync_dist=True,
                    prog_bar=True,
                    add_dataloader_idx=False,
                )
                self.log(
                    f"{dataset_name}_val_f1",
                    self.val_f1_score(pred_labels, labels),
                    logger=True,
                    sync_dist=True,
                    prog_bar=False,
                    add_dataloader_idx=False,
                )
                print(f"{dataset_name} : {self.val_accuracy(pred_labels, labels)}")

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

    elif isinstance(csv_paths, str):
        df = pd.read_csv(csv_paths)
        return df
    else:
        raise ValueError(f"Unknown type for cfg.data.train_csv={csv_paths}")

def parse_dataset_info(cfg):
    train_dataset_info = cfg.data.get('train')
    valid_dataset_info = cfg.data.get('valid')

    if not train_dataset_info or not valid_dataset_info:
        raise ValueError("train_csv or valid_csv is not defined")

    train_datasets = []
    n_train_images = 0

    for dataset in train_dataset_info:
        dataset_name, csv_path = dataset.dataset_name, dataset.csv_path
        df = pd.read_csv(csv_path).reset_index(drop=True)

        if cfg.debug:
            df = df.sample(n=min(200, len(df)), random_state=cfg.seed).reset_index(drop=True)

        train_datasets.append({
            "name": dataset_name,
            "df": df,
            "csv_path": csv_path,
        })

        n_train_images += len(df)
        print(f"Train Dataset `{dataset_name}` size: {len(df)}")
        print(df.head())
    print(f"training number of images: {n_train_images}")

    valid_datasets = []
    n_valid_images = 0
    for dataset in valid_dataset_info:
        dataset_name, csv_path = dataset.dataset_name, dataset.csv_path
        df = pd.read_csv(csv_path).reset_index(drop=True)

        if cfg.debug:
            df = df.sample(n=min(200, len(df)), random_state=cfg.seed).reset_index(drop=True)

        valid_datasets.append({
            "name": dataset_name,
            "df": df,
            "csv_path": csv_path,
        })

        n_valid_images += len(df)
        print(f"Validation Dataset `{dataset_name}` size: {len(df)}")
        print(df.head())

    print(f"validation number of images: {n_valid_images}")
    cfg.train.num_samples_per_epoch = n_train_images

    # save the dataframe
    if cfg.data.save_dataframe:
        save_df_dir = os.path.join(cfg.result_path, cfg.exp_name, cfg.exp_version)
        os.makedirs(save_df_dir, exist_ok=True)
        for idx, df_train in enumerate(train_datasets):
            csv_name = Path(df_train['csv_path']).name
            shutil.copy(df_train['csv_path'], os.path.join(save_df_dir, f"train_{idx}_{csv_name}.csv"))
        for idx, df_valid in enumerate(valid_datasets):
            csv_name = Path(df_valid['csv_path']).name
            shutil.copy(df_valid['csv_path'], os.path.join(save_df_dir, f"valid_{idx}_{csv_name}.csv"))

    return train_datasets, valid_datasets

def train(cfg):
    pl.seed_everything(cfg.seed)
    train_dicts, valid_dicts = parse_dataset_info(cfg)

    # set class weights
    class_weights: Dict[str, float] = compute_class_weights(train_dicts, "label")
    # class_weights = {cfg.data.labels[idx]:weight for idx, weight in class_weights.items()}
    print(f"Class weights\n: {class_weights}")
    if "Other" not in class_weights:
        class_weights['Other'] = 1.0
    class_weights: List[float] = [class_weights[label] for label in cfg.data.labels]

    # if cfg.debug:
    #     class_weights = [1,1,1,1,1,1]

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    cfg.class_weights_tensor = (
        class_weights_tensor  # need this for loss function in model forward)
    )

    class2class_idx = {label: idx for idx, label in enumerate(cfg.data.labels)}

    # train_dataset = UBCDataset(df_trains, label2label_idx, transforms=cfg.data_transforms["train"])
    train_datasets = [UBCDataset(ele['df'], class2class_idx, transforms=cfg.data_transforms["train"]) for ele in train_dicts]
    # valid_dataset = UBCDataset(df_valids, label2label_idx, transforms=cfg.data_transforms["valid"])
    valid_datasets = []
    for ele in valid_dicts:
        dataset_name = ele['name']
        if 'by_image_id' in dataset_name:
            # not going to use labels
            dataset = UBCDataset(ele['df'], class2class_idx, transforms=cfg.data_transforms["valid"], use_labels=False)
        else:
            # using labels
            dataset = UBCDataset(ele['df'], class2class_idx, transforms=cfg.data_transforms["valid"])
        valid_datasets.append(dataset)

    data_module = UBCDataModule(cfg)


    # merge train datasets to use single dataloader for training
    if cfg.data.merge_train_datasets:
        data_module.train_datasets = torch.utils.data.ConcatDataset(train_datasets)
    else:
        raise NotImplementedError("Not implemented yet")

    # we are not merging validation datasets because we want to know the model's performance on each dataset
    data_module.valid_datasets = valid_datasets
    data_module.num_classes = len(cfg.data.labels)

    model_module = UBCModelModule(cfg)

    ori_df = pd.read_csv(cfg.data.original_csv_path)
    model_module.image_id2class_idx = {image_id: class2class_idx[label] for image_id, label in zip(ori_df['image_id'], ori_df['label'])}
    model_module.train_dataset_idx2dataset_name = {idx: df_train['name'] for idx, df_train in enumerate(train_dicts)}
    model_module.valid_dataset_idx2dataset_name = {idx: df_valid['name'] for idx, df_valid in enumerate(valid_dicts)}
    model_module.model = CancerSubtypeClassifier(cfg)
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
    csv_logger = CSVLogger(
        save_dir=cfg.result_path,
        name=cfg.exp_name,
        version=cfg.exp_version,
        # sub_dir="logs",
    )

    lr_callback = LearningRateMonitor(logging_interval="step")
    model_summary_callback = ModelSummary(max_depth=4)
    checkpoint_filename = f"{cfg.model.name}" + "-{epoch:02d}-{step}"
    for idx, ele in enumerate(valid_dicts[:4]):
        dataset_name = ele['name']
        checkpoint_filename += "-{" + dataset_name + "_val_acc:.4f}"
    # for dataset_name in model_module.df_trains:
    checkpoint_callback = ModelCheckpoint(
        monitor=f"{valid_dicts[0]['name']}_val_acc",
        dirpath=Path(cfg.result_path) / cfg.exp_name / cfg.exp_version / "ckpts",
        filename=checkpoint_filename,
        save_top_k=30,  # if you save more than 1 model,
        # then checkpoint and huggingface model are not guaranteed to be matching
        # because we are saving with huggingface model with save_pretrained method
        # in "on_save_checkpoint" method in "BROSModelPLModule"
        mode="max",
        save_weights_only=True,
    )


    progress_bar = ProgressBar()

    # ==============================

    trainer = pl.Trainer(
        # fast_dev_run=True,
        accelerator=cfg.train.accelerator,
        logger=[tensorboard_logger, csv_logger],
        precision=cfg.model.precision,  # is it the best?
        # accumulate_grad_batches=14,
        val_check_interval=cfg.train.val_check_interval,
        callbacks=[
            checkpoint_callback,
            lr_callback,
            model_summary_callback,
            progress_bar
        ],
        max_epochs=cfg.train.max_epochs,
        num_sanity_val_steps=3,
        gradient_clip_val=cfg.train.clip_gradient_value,
        gradient_clip_algorithm=cfg.train.clip_gradient_algorithm,
    )

    # trainer.fit(model=model_module, datamodule=data_module, ckpt_path=cfg.model.checkpoint_path)
    trainer.fit(model=model_module, datamodule=data_module)
    # trainer.validate(model=model_module, datamodule=data_module) # if you want to run validation step only


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exp_version", type=str, required=False)
    parser.add_argument("--checkpoint_path", type=str, required=False)
    args, left_argv = parser.parse_known_args()

    config = Config(args.config)
    config.argv_update(left_argv)

    config.exp_name = os.path.basename(args.config).split(".")[0]
    config.exp_version = (
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not args.exp_version
        else args.exp_version
    )

    if args.checkpoint_path:
        config.model.checkpoint_path = args.checkpoint_path

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
