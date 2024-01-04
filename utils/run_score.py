import os
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sconf import Config


def compare_with_answer_wsi_only(inference_result_csv_path: str, answer_csv_path: str):
    inference_df = pd.read_csv(inference_result_csv_path)
    answer_df = pd.read_csv(answer_csv_path)

    wsi_df = answer_df[answer_df["is_tma"] == False]
    wsi_image_ids = set(wsi_df["image_id"].unique())

    inference_df = inference_df[inference_df["image_id"].isin(wsi_image_ids)]
    answer_df = answer_df[answer_df["image_id"].isin(wsi_image_ids)]

    print(answer_df["label"].value_counts())

    df = answer_df
    df["pred_label"] = inference_df["label"]
    df["correct"] = df["label"] == df["pred_label"]

    print("wrong answer, label count")
    print(df[df["correct"] == False]["label"].value_counts())


def compare_with_answer(
    inference_result_csv_path: str, answer_csv_path: str, mode: str
):
    inference_df = pd.read_csv(inference_result_csv_path)
    answer_df = pd.read_csv(answer_csv_path)

    target_image_ids = set()

    if "tma" in mode:
        tma_df = answer_df[answer_df["is_tma"] == True]
        tma_image_ids = set(tma_df["image_id"].unique())
        target_image_ids |= tma_image_ids

    if "wsi" in mode:
        wsi_df = answer_df[answer_df["is_tma"] == False]
        wsi_image_ids = set(wsi_df["image_id"].unique())
        target_image_ids |= wsi_image_ids

    inference_df = inference_df[inference_df["image_id"].isin(target_image_ids)]
    answer_df = answer_df[answer_df["image_id"].isin(target_image_ids)]

    df = answer_df
    df["pred_label"] = inference_df["label"]
    df["correct"] = df["label"] == df["pred_label"]

    print("original df")
    print(df["label"].value_counts())

    print("wrong answer, label count")
    print(df[df["correct"] == False]["label"].value_counts())
    print(df[df["correct"] == False])


def compare_with_answer_external(inference_result_csv_path: str, answer_csv_path: str):
    inference_df = pd.read_csv(inference_result_csv_path)
    answer_df = pd.read_csv(answer_csv_path)

    df = answer_df
    df["pred_label"] = inference_df["label"]
    df["correct"] = df["label"] == df["pred_label"]

    print("wrong answer, label count")
    print(df[df["correct"] == False]["label"].value_counts())

compare_with_answer(
    "./submission.csv",
    "/kaggle/input/UBC-OCEAN/train.csv",
    'tma'
)
