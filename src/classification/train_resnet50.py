from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
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

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "vinbigdata"
    / "train"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "classification"
    / "resnet50_15class_multilabel"
)

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

BATCH_SIZE = 8

NUM_EPOCHS = 5

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# OUTPUT FILES
# ============================================================

BEST_MODEL_PATH = (
    OUTPUT_DIR / "best_model.pth"
)

FINAL_MODEL_PATH = (
    OUTPUT_DIR / "final_model.pth"
)

HISTORY_PATH = (
    OUTPUT_DIR / "training_history.json"
)

CLASS_NAMES_PATH = (
    OUTPUT_DIR / "class_names.json"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=RANDOM_SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET
# ============================================================

class VinBigDataDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None
    ):

        self.df = (
            dataframe
            .reset_index(drop=True)
        )

        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

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
                f"Image not found: "
                f"{image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:

            image = self.transform(image)

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
# IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose(
    [

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=7
        ),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15
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


val_transform = transforms.Compose(
    [

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
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
# LOAD DATA
# ============================================================

def load_data():

    print("\nLoading training CSV...")

    train_df = pd.read_csv(
        TRAIN_CSV
    )

    print("Loading validation CSV...")

    val_df = pd.read_csv(
        VAL_CSV
    )

    print(
        f"Training images   : "
        f"{len(train_df)}"
    )

    print(
        f"Validation images : "
        f"{len(val_df)}"
    )

    # --------------------------------------------------------
    # Verify required columns
    # --------------------------------------------------------

    required_columns = (
        ["image_id"]
        + CLASS_NAMES
    )

    missing_train = [
        col
        for col in required_columns
        if col not in train_df.columns
    ]

    missing_val = [
        col
        for col in required_columns
        if col not in val_df.columns
    ]

    if missing_train:

        raise ValueError(
            "Training CSV is missing columns: "
            f"{missing_train}"
        )

    if missing_val:

        raise ValueError(
            "Validation CSV is missing columns: "
            f"{missing_val}"
        )

    print(
        "\nMulti-label CSV verified."
    )

    return train_df, val_df


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    print(
        "\nLoading pretrained ResNet-50..."
    )

    try:

        weights = (
            models.ResNet50_Weights.DEFAULT
        )

        model = models.resnet50(
            weights=weights
        )

        print(
            "Pretrained ImageNet weights loaded."
        )

    except Exception as error:

        print(
            "\nWARNING: Could not load "
            "pretrained weights."
        )

        print(
            f"Reason: {error}"
        )

        model = models.resnet50(
            weights=None
        )

    input_features = (
        model.fc.in_features
    )

    model.fc = nn.Linear(
        input_features,
        NUM_CLASSES
    )

    return model


# ============================================================
# CALCULATE POSITIVE WEIGHTS
# ============================================================

def calculate_pos_weights(
    train_df
):

    print(
        "\nCalculating positive class weights..."
    )

    positive_counts = (
        train_df[CLASS_NAMES]
        .sum()
        .values
    )

    total_samples = len(
        train_df
    )

    pos_weights = []

    for count in positive_counts:

        count = float(count)

        if count > 0:

            weight = (
                total_samples - count
            ) / count

        else:

            weight = 1.0

        # Avoid excessively huge weights
        weight = min(
            weight,
            20.0
        )

        pos_weights.append(
            weight
        )

    pos_weights = torch.tensor(
        pos_weights,
        dtype=torch.float32
    )

    print("\nPositive weights:")

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{i:2d} "
            f"{class_name:<25} "
            f"{pos_weights[i].item():.4f}"
        )

    return pos_weights


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    total = 0

    for batch_index, (
        images,
        labels
    ) in enumerate(loader):

        images = images.to(
            device
        )

        labels = labels.to(
            device
        )

        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        batch_size = (
            images.size(0)
        )

        running_loss += (
            loss.item()
            * batch_size
        )

        total += batch_size

        if (
            (batch_index + 1) % 100
            == 0
        ):

            print(
                f"  Batch "
                f"{batch_index + 1}/"
                f"{len(loader)}"
            )

    return (
        running_loss / total
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    total = 0

    all_labels = []

    all_probabilities = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            batch_size = (
                images.size(0)
            )

            running_loss += (
                loss.item()
                * batch_size
            )

            total += batch_size

            probabilities = (
                torch.sigmoid(
                    outputs
                )
            )

            all_labels.append(
                labels
                .cpu()
                .numpy()
            )

            all_probabilities.append(
                probabilities
                .cpu()
                .numpy()
            )

    epoch_loss = (
        running_loss / total
    )

    all_labels = np.concatenate(
        all_labels,
        axis=0
    )

    all_probabilities = np.concatenate(
        all_probabilities,
        axis=0
    )

    # --------------------------------------------------------
    # Threshold predictions
    # --------------------------------------------------------

    predictions = (
        all_probabilities >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # Macro F1
    # --------------------------------------------------------

    epoch_f1 = f1_score(
        all_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------

    class_aucs = []

    for class_id in range(
        NUM_CLASSES
    ):

        y_true = (
            all_labels[:, class_id]
        )

        y_score = (
            all_probabilities[:, class_id]
        )

        # AUC requires both positive
        # and negative examples.
        if (
            len(np.unique(y_true))
            > 1
        ):

            auc = roc_auc_score(
                y_true,
                y_score
            )

            class_aucs.append(
                auc
            )

    if class_aucs:

        epoch_auc = float(
            np.mean(class_aucs)
        )

    else:

        epoch_auc = float("nan")

    return (
        epoch_loss,
        epoch_f1,
        epoch_auc,
        all_labels,
        all_probabilities
    )


# ============================================================
# PRINT PER-CLASS METRICS
# ============================================================

def print_per_class_metrics(
    labels,
    probabilities
):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    print(
        "\n" + "=" * 70
    )

    print(
        "PER-CLASS VALIDATION METRICS"
    )

    print(
        "=" * 70
    )

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        y_true = (
            labels[:, class_id]
        )

        y_pred = (
            predictions[:, class_id]
        )

        y_score = (
            probabilities[:, class_id]
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        if len(
            np.unique(y_true)
        ) > 1:

            auc = roc_auc_score(
                y_true,
                y_score
            )

        else:

            auc = float("nan")

        print(
            f"{class_id:2d} "
            f"{class_name:<25} "
            f"F1={f1:.4f} "
            f"AUC={auc:.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed()

    print(
        "=" * 70
    )

    print(
        "RESNET-50 15-CLASS MULTI-LABEL TRAINING"
    )

    print(
        "=" * 70
    )

    print(
        f"Device     : {DEVICE}"
    )

    print(
        f"Classes    : {NUM_CLASSES}"
    )

    print(
        f"Epochs     : {NUM_EPOCHS}"
    )

    print(
        f"Batch size : {BATCH_SIZE}"
    )

    print(
        f"Image size : {IMAGE_SIZE}"
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    train_df, val_df = load_data()

    # ========================================================
    # DATASETS
    # ========================================================

    print(
        "\nCreating datasets..."
    )

    train_dataset = (
        VinBigDataDataset(
            train_df,
            transform=train_transform
        )
    )

    val_dataset = (
        VinBigDataDataset(
            val_df,
            transform=val_transform
        )
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(
        f"Training batches   : "
        f"{len(train_loader)}"
    )

    print(
        f"Validation batches : "
        f"{len(val_loader)}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = create_model()

    model = model.to(
        DEVICE
    )

    print(
        "\nResNet-50 model ready."
    )

    # ========================================================
    # LOSS
    # ========================================================

    pos_weights = (
        calculate_pos_weights(
            train_df
        )
    )

    pos_weights = (
        pos_weights.to(
            DEVICE
        )
    )

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weights
        )
    )

    print(
        "\nUsing BCEWithLogitsLoss "
        "for multi-label classification."
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=1
        )
    )

    # ========================================================
    # TRAINING
    # ========================================================

    best_val_loss = float(
        "inf"
    )

    history = []

    final_labels = None

    final_probabilities = None

    print(
        "\n" + "=" * 70
    )

    print(
        "STARTING MULTI-LABEL TRAINING"
    )

    print(
        "=" * 70
    )

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        print(
            "\n" + "=" * 70
        )

        print(
            f"Epoch {epoch}/{NUM_EPOCHS}"
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_loss = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                DEVICE
            )
        )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        (
            val_loss,
            val_f1,
            val_auc,
            val_labels,
            val_probabilities
        ) = validate(
            model,
            val_loader,
            criterion,
            DEVICE
        )

        final_labels = (
            val_labels
        )

        final_probabilities = (
            val_probabilities
        )

        scheduler.step(
            val_loss
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        print(
            "\nEpoch results:"
        )

        print(
            f"Train loss : "
            f"{train_loss:.4f}"
        )

        print(
            f"Val loss   : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val F1     : "
            f"{val_f1:.4f}"
        )

        print(
            f"Val AUC    : "
            f"{val_auc:.4f}"
        )

        print(
            f"LR         : "
            f"{current_lr:.6f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1": val_f1,
                "val_auc": (
                    None
                    if np.isnan(val_auc)
                    else val_auc
                ),
                "learning_rate": current_lr,
            }
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = (
                val_loss
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "val_loss":
                        val_loss,
                    "val_f1":
                        val_f1,
                    "val_auc":
                        val_auc,
                    "class_names":
                        CLASS_NAMES,
                    "num_classes":
                        NUM_CLASSES,
                    "image_size":
                        IMAGE_SIZE,
                },
                BEST_MODEL_PATH
            )

            print(
                "\nBest model saved:"
            )

            print(
                BEST_MODEL_PATH
            )

    # ========================================================
    # FINAL PER-CLASS METRICS
    # ========================================================

    print_per_class_metrics(
        final_labels,
        final_probabilities
    )

    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
            "class_names":
                CLASS_NAMES,
            "num_classes":
                NUM_CLASSES,
            "image_size":
                IMAGE_SIZE,
        },
        FINAL_MODEL_PATH
    )

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )

    # ========================================================
    # SAVE CLASS NAMES
    # ========================================================

    with open(
        CLASS_NAMES_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                str(i): name
                for i, name
                in enumerate(
                    CLASS_NAMES
                )
            },
            file,
            indent=4
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "MULTI-LABEL RESNET-50 TRAINING COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"Best validation loss : "
        f"{best_val_loss:.4f}"
    )

    print(
        f"\nBest model:"
    )

    print(
        BEST_MODEL_PATH
    )

    print(
        f"\nFinal model:"
    )

    print(
        FINAL_MODEL_PATH
    )

    print(
        f"\nTraining history:"
    )

    print(
        HISTORY_PATH
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()