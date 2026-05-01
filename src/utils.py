"""
src/utils.py
Shared utilities for the Medical VQA Classification project.

Extracted from eda.ipynb so all team members can import
without copy-pasting code across notebooks.

Usage:
    import sys, os
    sys.path.append(os.path.abspath(".."))   # if running from notebooks/
    from src.utils import (
        load_binary_metadata,
        get_dataloaders,
        MedicalVQABinaryDataset,
        IMAGE_TRANSFORM,
        compute_metrics,
        print_metrics,
    )
"""

import io
import pandas as pd
import numpy as np

from datasets import load_dataset
from PIL import Image

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
import torchvision.transforms as transforms

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------

def load_medical_vqa_dataset(dataset_name="robailleo/medical-vision-llm-dataset"):
    """
    Loads the Medical Vision LLM dataset from Hugging Face.

    Returns:
        DatasetDict with 'train' and 'validation' splits.
    """
    dataset = load_dataset(dataset_name)
    return dataset


def convert_splits_to_dataframe(dataset):
    """
    Converts train and validation splits into pandas DataFrames
    and adds a 'split' column.

    Returns:
        train_df, val_df, full_df
    """
    train_df = dataset["train"].to_pandas()
    val_df = dataset["validation"].to_pandas()

    train_df["split"] = "train"
    val_df["split"] = "validation"

    full_df = pd.concat([train_df, val_df], ignore_index=True)

    return train_df, val_df, full_df


def load_binary_metadata(csv_path="binary_vqa_metadata.csv"):
    """
    Loads the pre-exported binary VQA metadata CSV (940 rows).
    Returns train_df and val_df already split.

    Args:
        csv_path: path to binary_vqa_metadata.csv

    Returns:
        train_df, val_df
    """
    df = pd.read_csv(csv_path)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "validation"].reset_index(drop=True)
    return train_df, val_df


# ---------------------------------------------------------------------------
# 2. COLUMN / TEXT HELPERS
# ---------------------------------------------------------------------------

def first_existing_column(df, candidates):
    """
    Returns the first column name from candidates that exists in df.
    Useful for handling naming variations across dataset versions.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_text(x):
    """Converts text to lowercase and strips whitespace."""
    if x is None:
        return None
    return str(x).strip().lower()


def infer_question_type(answer):
    """
    Returns 'binary' if answer is yes/no, else 'open_ended'.
    """
    ans = normalize_text(answer)
    if ans in {"yes", "no"}:
        return "binary"
    return "open_ended"


def infer_modality_from_row(row, modality_col=None):
    """
    Infers image modality (X-ray, CT, MRI, etc.) from an existing
    modality column or by scanning the question/answer text.
    """
    if modality_col and modality_col in row and pd.notna(row[modality_col]):
        return str(row[modality_col]).strip()

    text_bits = []
    for key in ["question", "caption", "answer", "report", "context"]:
        if key in row and pd.notna(row[key]):
            text_bits.append(str(row[key]).lower())
    text = " ".join(text_bits)

    mapping = {
        "xray": "X-ray",
        "x-ray": "X-ray",
        "radiograph": "X-ray",
        "computed tomography": "CT",
        "ct ": "CT",
        " mri": "MRI",
        "magnetic resonance": "MRI",
        "ultrasound": "Ultrasound",
        "sonography": "Ultrasound",
    }
    for keyword, modality in mapping.items():
        if keyword in text:
            return modality
    return "Unknown"


# ---------------------------------------------------------------------------
# 3. IMAGE HELPERS
# ---------------------------------------------------------------------------

def open_image_from_dataset_value(image_value):
    """
    Opens a PIL Image from various HuggingFace / pandas formats:
    - PIL Image directly
    - dict with 'bytes' key
    - dict with 'path' key
    """
    if isinstance(image_value, Image.Image):
        return image_value
    if isinstance(image_value, dict) and image_value.get("bytes") is not None:
        return Image.open(io.BytesIO(image_value["bytes"]))
    if isinstance(image_value, dict) and image_value.get("path") is not None:
        return Image.open(image_value["path"])
    return None


# Standard preprocessing pipeline — ResNet/ImageNet compatible (224x224, RGB, normalized)
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------------------------------------------------------------------
# 4. PYTORCH DATASET
# ---------------------------------------------------------------------------

class MedicalVQABinaryDataset(TorchDataset):
    """
    PyTorch Dataset for binary Medical VQA classification.

    Each __getitem__ returns a dict:
        {
            "image":    FloatTensor (3, 224, 224),
            "question": str,
            "label":    LongTensor (scalar 0 or 1)
        }

    Args:
        dataframe:    pandas DataFrame with image, question, and label columns
        image_col:    column name holding the image data
        question_col: column name holding the question text
        label_col:    column name holding the binary label (default: "label")
        transform:    torchvision transform (defaults to IMAGE_TRANSFORM)
    """
    def __init__(self, dataframe, image_col, question_col,
                 label_col="label", transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_col = image_col
        self.question_col = question_col
        self.label_col = label_col
        self.transform = transform if transform is not None else IMAGE_TRANSFORM

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        img = open_image_from_dataset_value(row[self.image_col])
        if img is None:
            raise ValueError(f"Could not open image at index {idx}")

        img = self.transform(img)

        question = str(row[self.question_col])
        label = torch.tensor(row[self.label_col], dtype=torch.long)

        return {"image": img, "question": question, "label": label}


# ---------------------------------------------------------------------------
# 5. DATALOADER FACTORY
# ---------------------------------------------------------------------------

def get_dataloaders(train_df, val_df, image_col, question_col,
                    label_col="label", batch_size=16, transform=None):
    """
    Builds train and validation DataLoaders from DataFrames.

    Args:
        train_df, val_df: DataFrames from convert_splits_to_dataframe()
        image_col:        column holding image data
        question_col:     column holding question text
        label_col:        column holding binary label
        batch_size:       default 16
        transform:        optional custom transform (uses IMAGE_TRANSFORM if None)

    Returns:
        train_loader, val_loader
    """
    train_dataset = MedicalVQABinaryDataset(
        train_df, image_col, question_col, label_col, transform
    )
    val_dataset = MedicalVQABinaryDataset(
        val_df, image_col, question_col, label_col, transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# 6. EVALUATION HELPERS
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Computes accuracy, F1, and optionally AUC-ROC.

    Args:
        y_true:  list/array of true binary labels
        y_pred:  list/array of predicted binary labels
        y_prob:  list/array of predicted probabilities for positive class
                 (needed for AUC-ROC; pass None to skip)

    Returns:
        dict with keys: accuracy, f1, auc_roc (or None)
    """
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1":       f1_score(y_true, y_pred, zero_division=0),
        "auc_roc":  roc_auc_score(y_true, y_prob) if y_prob is not None else None,
    }
    return results


def print_metrics(metrics: dict, model_name: str = "Model"):
    """Pretty-prints the metrics dict returned by compute_metrics()."""
    print(f"\n{'='*40}")
    print(f"  {model_name}")
    print(f"{'='*40}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1 Score : {metrics['f1']:.4f}")
    if metrics.get("auc_roc") is not None:
        print(f"  AUC-ROC  : {metrics['auc_roc']:.4f}")
    print(f"{'='*40}\n")
