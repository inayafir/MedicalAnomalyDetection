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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Classification dataset
DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "classification"
    / "resnet50"
)

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"

# Original VinBigData images
IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "vinbigdata"
    / "train"
)

# Trained model
MODEL_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "classification"
    / "resnet50"
)

MODEL_PATH = MODEL_DIR / "best_model.pth"

# Evaluation output
OUTPUT_DIR = (
    MODEL_DIR
    / "evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 5

CLASS_NAMES = [
    "Normal",
    "Cardiomegaly",
    "Pleural effusion",
    "Lung Opacity",
    "Pulmonary fibrosis",
]

IMAGE_SIZE = 224

BATCH_SIZE = 16

NUM_WORKERS = 0

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

        label = int(
            row["class_id"]
        )

        image_path = (
            IMAGE_DIR
            / f"{image_id}.png"
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found:\n"
                f"{image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:

            image = self.transform(
                image
            )

        return image, label


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

    # Do NOT download ImageNet weights again.
    # The trained checkpoint contains our trained model weights.

    model = models.resnet50(
        weights=None
    )

    # Replace ImageNet's 1000-class output
    # with our 5 project classes.

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

    # The training script saves:
    #
    # {
    #     "epoch": ...,
    #     "model_state_dict": ...,
    #     "optimizer_state_dict": ...,
    #     ...
    # }

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        # Fallback for a plain state_dict.
        model.load_state_dict(
            checkpoint
        )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        "ResNet-50 checkpoint loaded successfully."
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

    all_predictions = []

    all_probabilities = []

    print(
        "\nRunning validation evaluation..."
    )

    with torch.no_grad():

        for batch_index, (
            images,
            labels
        ) in enumerate(loader):

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            # Forward pass
            outputs = model(
                images
            )

            # Convert logits to probabilities
            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            # Select highest-probability class
            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                labels
                .cpu()
                .numpy()
                .tolist()
            )

            all_predictions.extend(
                predictions
                .cpu()
                .numpy()
                .tolist()
            )

            all_probabilities.extend(
                probabilities
                .cpu()
                .numpy()
                .tolist()
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

    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities)
    )


# ============================================================
# OVERALL METRICS
# ============================================================

def calculate_metrics(
    labels,
    predictions,
    probabilities
):

    accuracy = accuracy_score(
        labels,
        predictions
    )

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

    # Multiclass one-vs-rest AUC
    try:

        auc = roc_auc_score(
            labels,
            probabilities,
            multi_class="ovr",
            average="macro"
        )

    except ValueError:

        auc = float("nan")

    return {
        "accuracy": float(
            accuracy
        ),
        "macro_precision": float(
            precision
        ),
        "macro_recall": float(
            recall
        ),
        "macro_f1": float(
            f1
        ),
        "macro_auc_roc": float(
            auc
        ),
    }


# ============================================================
# PER-CLASS METRICS
# ============================================================

def calculate_per_class_metrics(
    labels,
    predictions
):

    precision = precision_score(
        labels,
        predictions,
        labels=list(
            range(NUM_CLASSES)
        ),
        average=None,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        labels=list(
            range(NUM_CLASSES)
        ),
        average=None,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        labels=list(
            range(NUM_CLASSES)
        ),
        average=None,
        zero_division=0
    )

    results = {}

    for class_id in range(
        NUM_CLASSES
    ):

        results[
            CLASS_NAMES[class_id]
        ] = {
            "class_id": class_id,
            "precision": float(
                precision[class_id]
            ),
            "recall": float(
                recall[class_id]
            ),
            "f1": float(
                f1[class_id]
            ),
        }

    return results


# ============================================================
# CONFUSION MATRIX
# ============================================================

def create_confusion_matrix(
    labels,
    predictions
):

    return confusion_matrix(
        labels,
        predictions,
        labels=list(
            range(NUM_CLASSES)
        )
    )


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    cm
):

    output_path = (
        OUTPUT_DIR
        / "confusion_matrix.npy"
    )

    np.save(
        output_path,
        cm
    )

    return output_path


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

def save_classification_report(
    labels,
    predictions
):

    report = classification_report(
        labels,
        predictions,
        labels=list(
            range(NUM_CLASSES)
        ),
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
# SAVE SUMMARY JSON
# ============================================================

def save_summary(
    metrics,
    per_class_metrics,
    checkpoint
):

    summary = {

        "model": "ResNet-50",

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

        "num_validation_images": None,

        "num_classes": NUM_CLASSES,

        "class_names": CLASS_NAMES,

        "overall_metrics": metrics,

        "per_class_metrics": per_class_metrics,

    }

    # Save training checkpoint metadata
    if isinstance(
        checkpoint,
        dict
    ):

        if "epoch" in checkpoint:

            summary[
                "checkpoint_epoch"
            ] = checkpoint[
                "epoch"
            ]

        if "val_accuracy" in checkpoint:

            summary[
                "checkpoint_val_accuracy"
            ] = checkpoint[
                "val_accuracy"
            ]

        if "val_f1" in checkpoint:

            summary[
                "checkpoint_val_f1"
            ] = checkpoint[
                "val_f1"
            ]

        if "val_auc" in checkpoint:

            summary[
                "checkpoint_val_auc"
            ] = checkpoint[
                "val_auc"
            ]

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

    print("=" * 60)

    print(
        "RESNET-50 CLASSIFICATION EVALUATION"
    )

    print("=" * 60)

    print(
        f"Device       : {DEVICE}"
    )

    print(
        f"Validation   : {VAL_CSV}"
    )

    print(
        f"Images       : {IMAGE_DIR}"
    )

    print(
        f"Model        : {MODEL_PATH}"
    )

    print(
        f"Output       : {OUTPUT_DIR}"
    )

    # ========================================================
    # CHECK REQUIRED FILES
    # ========================================================

    if not VAL_CSV.exists():

        raise FileNotFoundError(
            f"\nValidation CSV not found:\n"
            f"{VAL_CSV}"
        )

    if not IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"\nImage directory not found:\n"
            f"{IMAGE_DIR}"
        )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nResNet-50 checkpoint not found:\n"
            f"{MODEL_PATH}"
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
        f"Validation images : "
        f"{len(val_df)}"
    )

    print(
        "\nCSV columns:"
    )

    print(
        val_df.columns.tolist()
    )

    # Your actual CSV format is:
    #
    # image_id
    # project_class
    # class_id

    required_columns = {
        "image_id",
        "project_class",
        "class_id",
    }

    missing_columns = (
        required_columns
        - set(val_df.columns)
    )

    if missing_columns:

        raise ValueError(
            "\nMissing required columns: "
            f"{sorted(missing_columns)}"
        )

    # ========================================================
    # CHECK IMAGE FILES
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

        print(
            f"Missing images: "
            f"{len(missing_images)}"
        )

        print(
            "\nFirst missing image:"
        )

        print(
            missing_images[0]
        )

        raise FileNotFoundError(
            "\nSome validation images "
            "could not be found."
        )

    print(
        f"Images found: "
        f"{len(val_df)}"
    )

    print(
        "Images missing: 0"
    )

    # ========================================================
    # VALIDATION DATASET
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
    # RUN EVALUATION
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
    # CALCULATE METRICS
    # ========================================================

    metrics = calculate_metrics(
        labels,
        predictions,
        probabilities
    )

    per_class_metrics = (
        calculate_per_class_metrics(
            labels,
            predictions
        )
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = create_confusion_matrix(
        labels,
        predictions
    )

    cm_path = save_confusion_matrix(
        cm
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    (
        report,
        report_path
    ) = save_classification_report(
        labels,
        predictions
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_path = save_summary(
        metrics,
        per_class_metrics,
        checkpoint
    )

    # Update number of validation images
    with open(
        summary_path,
        "r",
        encoding="utf-8"
    ) as file:

        summary = json.load(
            file
        )

    summary[
        "num_validation_images"
    ] = len(val_df)

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    # ========================================================
    # PRINT OVERALL RESULTS
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "RESNET-50 EVALUATION RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"Accuracy        : "
        f"{metrics['accuracy']:.4f}"
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
    # PRINT PER-CLASS RESULTS
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "PER-CLASS RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"{'Class':<22}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
    )

    print(
        "-" * 58
    )

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        result = per_class_metrics[
            class_name
        ]

        print(
            f"{class_name:<22}"
            f"{result['precision']:>12.4f}"
            f"{result['recall']:>12.4f}"
            f"{result['f1']:>12.4f}"
        )

    # ========================================================
    # PRINT CLASSIFICATION REPORT
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "CLASSIFICATION REPORT"
    )

    print(
        "=" * 60
    )

    print(
        report
    )

    # ========================================================
    # PRINT CONFUSION MATRIX
    # ========================================================

    print(
        "=" * 60
    )

    print(
        "CONFUSION MATRIX"
    )

    print(
        "=" * 60
    )

    print(
        "Rows    = Actual"
    )

    print(
        "Columns = Predicted"
    )

    print()

    print(
        "             "
        + " ".join(
            f"{i:^7}"
            for i in range(NUM_CLASSES)
        )
    )

    for class_id in range(
        NUM_CLASSES
    ):

        print(
            f"{class_id} "
            f"{CLASS_NAMES[class_id]:<18}"
            + " ".join(
                f"{value:^7}"
                for value in cm[class_id]
            )
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "EVALUATION COMPLETED"
    )

    print(
        "=" * 60
    )

    print(
        f"Results directory : "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Summary file      : "
        f"{summary_path}"
    )

    print(
        f"Classification report : "
        f"{report_path}"
    )

    print(
        f"Confusion matrix      : "
        f"{cm_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()