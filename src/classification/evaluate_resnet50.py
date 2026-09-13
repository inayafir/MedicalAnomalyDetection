from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "classification"
    / "resnet50"
)

VAL_CSV = DATA_DIR / "val.csv"

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "vinbigdata"
    / "train"
)

# IMPORTANT:
# This is the checkpoint produced by train_resnet50.py
# for the 15-class MULTI-LABEL model.
MODEL_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "classification"
    / "resnet50_15class_multilabel"
)

MODEL_PATH = MODEL_DIR / "best_model.pth"

OUTPUT_DIR = MODEL_DIR / "evaluation"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 15

CLASS_NAMES = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Normal",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
]

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0

# Same threshold used during training
THRESHOLD = 0.5

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DATASET
# ============================================================

class VinBigDataDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[index]

        image_id = str(
            row["image_id"]
        )

        image_path = (
            IMAGE_DIR
            / f"{image_id}.png"
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found:\n{image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:

            image = self.transform(
                image
            )

        # ----------------------------------------------------
        # Multi-label target
        # ----------------------------------------------------

        labels = []

        for class_name in CLASS_NAMES:

            labels.append(
                float(row[class_name])
            )

        labels = torch.tensor(
            labels,
            dtype=torch.float32
        )

        return image, labels


# ============================================================
# VALIDATION TRANSFORM
# ============================================================

val_transform = transforms.Compose(
    [

        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        ),

    ]
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print(
        "\nLoading ResNet-50 architecture..."
    )

    model = models.resnet50(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    print(
        f"Loading checkpoint:\n"
        f"{MODEL_PATH}"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        "15-class multi-label ResNet-50 "
        "checkpoint loaded successfully."
    )

    if isinstance(
        checkpoint,
        dict
    ):

        if "epoch" in checkpoint:

            print(
                f"Checkpoint epoch : "
                f"{checkpoint['epoch']}"
            )

        if "val_loss" in checkpoint:

            print(
                f"Checkpoint val loss : "
                f"{checkpoint['val_loss']:.4f}"
            )

        if "val_f1" in checkpoint:

            print(
                f"Checkpoint val F1 : "
                f"{checkpoint['val_f1']:.4f}"
            )

        if "val_auc" in checkpoint:

            print(
                f"Checkpoint val AUC : "
                f"{checkpoint['val_auc']:.4f}"
            )

    return model, checkpoint


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader
):

    model.eval()

    all_labels = []
    all_probabilities = []

    print(
        "\nRunning 15-class multi-label evaluation..."
    )

    with torch.no_grad():

        for batch_index, (
            images,
            labels
        ) in enumerate(loader):

            images = images.to(
                DEVICE
            )

            outputs = model(
                images
            )

            # ------------------------------------------------
            # Multi-label probabilities
            # ------------------------------------------------

            probabilities = torch.sigmoid(
                outputs
            )

            all_labels.append(
                labels.numpy()
            )

            all_probabilities.append(
                probabilities
                .cpu()
                .numpy()
            )

            if (
                (batch_index + 1) % 50
                == 0
            ):

                print(
                    f"  Batch "
                    f"{batch_index + 1}/"
                    f"{len(loader)}"
                )

    labels = np.concatenate(
        all_labels,
        axis=0
    )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0
    )

    # --------------------------------------------------------
    # Convert probabilities to binary predictions
    # --------------------------------------------------------

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    return (
        labels,
        predictions,
        probabilities
    )


# ============================================================
# OVERALL METRICS
# ============================================================

def calculate_metrics(
    labels,
    predictions,
    probabilities
):

    precision = precision_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------------
    # Macro AUC
    # --------------------------------------------------------

    auc_scores = []

    for class_id in range(
        NUM_CLASSES
    ):

        y_true = labels[:, class_id]
        y_score = probabilities[:, class_id]

        # AUC is undefined if validation data contains
        # only one class for a particular label.
        if len(np.unique(y_true)) < 2:
            continue

        auc_scores.append(
            roc_auc_score(
                y_true,
                y_score
            )
        )

    macro_auc = (
        float(np.mean(auc_scores))
        if auc_scores
        else float("nan")
    )

    return {

        "macro_precision": float(
            precision
        ),

        "macro_recall": float(
            recall
        ),

        "macro_f1": float(
            f1
        ),

        "macro_auc_roc": macro_auc,

        "threshold": THRESHOLD,

    }


# ============================================================
# PER-CLASS METRICS
# ============================================================

def calculate_per_class_metrics(
    labels,
    predictions,
    probabilities
):

    results = {}

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        y_true = labels[:, class_id]

        y_pred = predictions[:, class_id]

        y_prob = probabilities[:, class_id]

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        if len(
            np.unique(y_true)
        ) >= 2:

            auc = roc_auc_score(
                y_true,
                y_prob
            )

        else:

            auc = float("nan")

        results[class_name] = {

            "class_id": class_id,

            "precision": float(
                precision
            ),

            "recall": float(
                recall
            ),

            "f1": float(
                f1
            ),

            "auc_roc": float(
                auc
            ),

            "support": int(
                y_true.sum()
            ),

        }

    return results


# ============================================================
# SAVE PER-CLASS REPORT
# ============================================================

def save_classification_report(
    labels,
    predictions
):

    report = classification_report(
        labels,
        predictions,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    output_path = (
        OUTPUT_DIR
        / "classification_report.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            report
        )

    return (
        report,
        output_path
    )


# ============================================================
# SAVE CONFUSION MATRICES
# ============================================================

def save_confusion_matrices(
    labels,
    predictions
):

    matrices = {}

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        y_true = labels[:, class_id]

        y_pred = predictions[:, class_id]

        tn = int(
            np.sum(
                (y_true == 0)
                & (y_pred == 0)
            )
        )

        fp = int(
            np.sum(
                (y_true == 0)
                & (y_pred == 1)
            )
        )

        fn = int(
            np.sum(
                (y_true == 1)
                & (y_pred == 0)
            )
        )

        tp = int(
            np.sum(
                (y_true == 1)
                & (y_pred == 1)
            )
        )

        matrices[class_name] = {

            "true_negative": tn,

            "false_positive": fp,

            "false_negative": fn,

            "true_positive": tp,

        }

    output_path = (
        OUTPUT_DIR
        / "confusion_matrices.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            matrices,
            file,
            indent=4
        )

    return output_path


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(
    metrics,
    per_class_metrics,
    checkpoint,
    num_validation_images
):

    summary = {

        "model": "ResNet-50",

        "task": "15-class multi-label classification",

        "num_classes": NUM_CLASSES,

        "class_names": CLASS_NAMES,

        "threshold": THRESHOLD,

        "checkpoint": str(
            MODEL_PATH
        ),

        "validation_csv": str(
            VAL_CSV
        ),

        "image_directory": str(
            IMAGE_DIR
        ),

        "device": str(
            DEVICE
        ),

        "num_validation_images":
            num_validation_images,

        "overall_metrics":
            metrics,

        "per_class_metrics":
            per_class_metrics,

    }

    if isinstance(
        checkpoint,
        dict
    ):

        for key in [
            "epoch",
            "val_loss",
            "val_f1",
            "val_auc"
        ]:

            if key in checkpoint:

                summary[
                    f"checkpoint_{key}"
                ] = checkpoint[key]

    output_path = (
        OUTPUT_DIR
        / "evaluation_summary.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 65
    )

    print(
        "RESNET-50 15-CLASS MULTI-LABEL EVALUATION"
    )

    print(
        "=" * 65
    )

    print(
        f"Device       : {DEVICE}"
    )

    print(
        f"Classes      : {NUM_CLASSES}"
    )

    print(
        f"Threshold    : {THRESHOLD}"
    )

    print(
        f"Validation   : {VAL_CSV}"
    )

    print(
        f"Model        : {MODEL_PATH}"
    )

    print(
        f"Output       : {OUTPUT_DIR}"
    )


    # ========================================================
    # CHECK FILES
    # ========================================================

    if not VAL_CSV.exists():

        raise FileNotFoundError(
            f"\nValidation CSV not found:\n{VAL_CSV}"
        )

    if not IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"\nImage directory not found:\n{IMAGE_DIR}"
        )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nMulti-label checkpoint not found:\n{MODEL_PATH}"
        )


    # ========================================================
    # LOAD VALIDATION CSV
    # ========================================================

    print(
        "\nLoading validation CSV..."
    )

    val_df = pd.read_csv(
        VAL_CSV
    )

    print(
        f"Validation images : {len(val_df)}"
    )


    # ========================================================
    # VERIFY LABEL COLUMNS
    # ========================================================

    required_columns = (
        ["image_id"]
        + CLASS_NAMES
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in val_df.columns
    ]

    if missing_columns:

        raise ValueError(
            "\nMissing required columns:\n"
            + "\n".join(
                missing_columns
            )
        )

    print(
        "\nAll 15 multi-label columns verified."
    )


    # ========================================================
    # VERIFY LABEL VALUES
    # ========================================================

    for class_name in CLASS_NAMES:

        unique_values = sorted(
            val_df[class_name]
            .dropna()
            .unique()
            .tolist()
        )

        if not set(unique_values).issubset(
            {0, 1}
        ):

            raise ValueError(
                f"Invalid values in "
                f"'{class_name}': "
                f"{unique_values}"
            )

    print(
        "All labels verified as binary 0/1."
    )


    # ========================================================
    # CHECK IMAGES
    # ========================================================

    print(
        "\nChecking validation images..."
    )

    missing_images = []

    for image_id in val_df[
        "image_id"
    ]:

        image_path = (
            IMAGE_DIR
            / f"{image_id}.png"
        )

        if not image_path.exists():

            missing_images.append(
                str(image_path)
            )

    if missing_images:

        raise FileNotFoundError(
            f"\nMissing validation images: "
            f"{len(missing_images)}\n"
            f"First missing image:\n"
            f"{missing_images[0]}"
        )

    print(
        f"Images found: {len(val_df)}"
    )

    print(
        "Images missing: 0"
    )


    # ========================================================
    # LABEL DISTRIBUTION
    # ========================================================

    print(
        "\nValidation label distribution:"
    )

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        count = int(
            val_df[class_name].sum()
        )

        print(
            f"{class_id:2d} "
            f"{class_name:<22} "
            f"{count}"
        )


    # ========================================================
    # DATASET
    # ========================================================

    print(
        "\nCreating validation dataset..."
    )

    val_dataset = VinBigDataDataset(
        val_df,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    model, checkpoint = load_model()


    # ========================================================
    # EVALUATE
    # ========================================================

    (
        labels,
        predictions,
        probabilities
    ) = evaluate(
        model,
        val_loader
    )


    # ========================================================
    # METRICS
    # ========================================================

    metrics = calculate_metrics(
        labels,
        predictions,
        probabilities
    )

    per_class_metrics = (
        calculate_per_class_metrics(
            labels,
            predictions,
            probabilities
        )
    )


    # ========================================================
    # SAVE REPORTS
    # ========================================================

    (
        report,
        report_path
    ) = save_classification_report(
        labels,
        predictions
    )

    confusion_path = save_confusion_matrices(
        labels,
        predictions
    )

    summary_path = save_summary(
        metrics,
        per_class_metrics,
        checkpoint,
        len(val_df)
    )


    # ========================================================
    # PRINT OVERALL RESULTS
    # ========================================================

    print(
        "\n" + "=" * 65
    )

    print(
        "RESNET-50 MULTI-LABEL EVALUATION RESULTS"
    )

    print(
        "=" * 65
    )

    print(
        f"Macro Precision : "
        f"{metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall    : "
        f"{metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1        : "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Macro AUC-ROC   : "
        f"{metrics['macro_auc_roc']:.4f}"
    )


    # ========================================================
    # PER-CLASS RESULTS
    # ========================================================

    print(
        "\n" + "=" * 65
    )

    print(
        "PER-CLASS RESULTS"
    )

    print(
        "=" * 65
    )

    print(
        f"{'Class':<22}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'AUC':>12}"
    )

    print(
        "-" * 70
    )

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        result = per_class_metrics[
            class_name
        ]

        auc_value = result["auc_roc"]

        auc_text = (
            f"{auc_value:.4f}"
            if not np.isnan(auc_value)
            else "N/A"
        )

        print(
            f"{class_name:<22}"
            f"{result['precision']:>12.4f}"
            f"{result['recall']:>12.4f}"
            f"{result['f1']:>12.4f}"
            f"{auc_text:>12}"
        )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print(
        "\n" + "=" * 65
    )

    print(
        "CLASSIFICATION REPORT"
    )

    print(
        "=" * 65
    )

    print(
        report
    )


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print(
        "=" * 65
    )

    print(
        "15-CLASS MULTI-LABEL RESNET-50 EVALUATION COMPLETED"
    )

    print(
        "=" * 65
    )

    print(
        f"Results directory : {OUTPUT_DIR}"
    )

    print(
        f"Summary file      : {summary_path}"
    )

    print(
        f"Classification report : {report_path}"
    )

    print(
        f"Confusion matrices     : {confusion_path}"
    )

    print(
        "\nNo training was performed."
    )

    print(
        "The existing 15-class MULTI-LABEL "
        "checkpoint was evaluated."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()