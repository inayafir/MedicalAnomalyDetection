from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
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
    / "resnet50"
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

# CPU training is slow, so keep this manageable.
BATCH_SIZE = 8

NUM_EPOCHS = 5

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

RANDOM_SEED = 42


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# OUTPUT FILES
# ============================================================

BEST_MODEL_PATH = (
    OUTPUT_DIR
    / "best_model.pth"
)

FINAL_MODEL_PATH = (
    OUTPUT_DIR
    / "final_model.pth"
)

HISTORY_PATH = (
    OUTPUT_DIR
    / "training_history.json"
)

CONFUSION_MATRIX_PATH = (
    OUTPUT_DIR
    / "confusion_matrix.npy"
)

CLASS_NAMES_PATH = (
    OUTPUT_DIR
    / "class_names.json"
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

    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[index]

        image_id = str(
            row["image_id"]
        )

        # IMPORTANT:
        # train.csv contains "class_id"
        label = int(
            row["class_id"]
        )

        image_path = (
            IMAGE_DIR
            / f"{image_id}.png"
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found: {image_path}"
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
# IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose(
    [

        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
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
# LOAD DATA
# ============================================================

def load_data():

    print(
        "\nLoading training CSV..."
    )

    train_df = pd.read_csv(
        TRAIN_CSV
    )

    print(
        "Loading validation CSV..."
    )

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
    # Verify expected columns
    # --------------------------------------------------------

    required_columns = {
        "image_id",
        "project_class",
        "class_id",
    }

    missing_train = (
        required_columns
        - set(train_df.columns)
    )

    missing_val = (
        required_columns
        - set(val_df.columns)
    )

    if missing_train:

        raise ValueError(
            "Training CSV is missing "
            f"columns: {missing_train}"
        )

    if missing_val:

        raise ValueError(
            "Validation CSV is missing "
            f"columns: {missing_val}"
        )

    print(
        "\nCSV columns verified:"
    )

    print(
        train_df.columns.tolist()
    )

    return train_df, val_df


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def print_class_distribution(
    train_df,
    val_df
):

    print(
        "\n" + "=" * 60
    )

    print(
        "CLASS DISTRIBUTION"
    )

    print(
        "=" * 60
    )

    print(
        "\nTraining:"
    )

    train_counts = (
        train_df["class_id"]
        .value_counts()
        .sort_index()
    )

    for class_id in range(
        NUM_CLASSES
    ):

        count = int(
            train_counts.get(
                class_id,
                0
            )
        )

        print(
            f"{class_id} "
            f"{CLASS_NAMES[class_id]:<22} "
            f"{count}"
        )

    print(
        "\nValidation:"
    )

    val_counts = (
        val_df["class_id"]
        .value_counts()
        .sort_index()
    )

    for class_id in range(
        NUM_CLASSES
    ):

        count = int(
            val_counts.get(
                class_id,
                0
            )
        )

        print(
            f"{class_id} "
            f"{CLASS_NAMES[class_id]:<22} "
            f"{count}"
        )


# ============================================================
# WEIGHTED RANDOM SAMPLER
# ============================================================

def create_weighted_sampler(
    train_df
):

    print(
        "\nCreating WeightedRandomSampler..."
    )

    # IMPORTANT:
    # The CSV column is "class_id".
    class_counts = (
        train_df["class_id"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nTraining class counts:"
    )

    print(
        class_counts
    )

    # Inverse-frequency weighting

    class_weights = {}

    for class_id in range(
        NUM_CLASSES
    ):

        count = class_counts.get(
            class_id,
            0
        )

        if count > 0:

            class_weights[
                class_id
            ] = 1.0 / float(count)

        else:

            class_weights[
                class_id
            ] = 0.0

    sample_weights = []

    for label in train_df[
        "class_id"
    ]:

        sample_weights.append(
            class_weights[
                int(label)
            ]
        )

    sample_weights = torch.tensor(
        sample_weights,
        dtype=torch.double
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(
            sample_weights
        ),
        replacement=True
    )

    print(
        "Weighted sampling enabled."
    )

    return sampler


# ============================================================
# CREATE RESNET-50
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
            "\nWARNING:"
        )

        print(
            "Could not load pretrained "
            "ResNet-50 weights."
        )

        print(
            f"Reason: {error}"
        )

        print(
            "Using ResNet-50 without "
            "pretrained weights."
        )

        model = models.resnet50(
            weights=None
        )

    # --------------------------------------------------------
    # Replace ImageNet classifier
    #
    # Original:
    # 2048 -> 1000
    #
    # Ours:
    # 2048 -> 5
    # --------------------------------------------------------

    input_features = (
        model.fc.in_features
    )

    model.fc = nn.Linear(
        input_features,
        NUM_CLASSES
    )

    return model


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

    all_labels = []

    all_predictions = []

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

        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(
            images
        )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            outputs,
            labels
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Update weights
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        batch_size = (
            images.size(0)
        )

        running_loss += (
            loss.item()
            * batch_size
        )

        total += batch_size

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        all_labels.extend(
            labels
            .detach()
            .cpu()
            .numpy()
        )

        all_predictions.extend(
            predictions
            .detach()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (batch_index + 1) % 100
            == 0
        ):

            print(
                f"  Batch "
                f"{batch_index + 1}/"
                f"{len(loader)}"
            )

    epoch_loss = (
        running_loss
        / total
    )

    epoch_accuracy = (
        accuracy_score(
            all_labels,
            all_predictions
        )
    )

    epoch_f1 = (
        f1_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0
        )
    )

    return (
        epoch_loss,
        epoch_accuracy,
        epoch_f1
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

    all_predictions = []

    all_probabilities = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            outputs = model(
                images
            )

            # ------------------------------------------------
            # Validation loss
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Probabilities
            # ------------------------------------------------

            probabilities = (
                torch.softmax(
                    outputs,
                    dim=1
                )
            )

            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            all_labels.extend(
                labels
                .cpu()
                .numpy()
            )

            all_predictions.extend(
                predictions
                .cpu()
                .numpy()
            )

            all_probabilities.extend(
                probabilities
                .cpu()
                .numpy()
            )

    epoch_loss = (
        running_loss
        / total
    )

    epoch_accuracy = (
        accuracy_score(
            all_labels,
            all_predictions
        )
    )

    epoch_f1 = (
        f1_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # AUC-ROC
    # --------------------------------------------------------

    probabilities = np.array(
        all_probabilities
    )

    labels_array = np.array(
        all_labels
    )

    try:

        epoch_auc = (
            roc_auc_score(
                labels_array,
                probabilities,
                multi_class="ovr",
                average="macro"
            )
        )

    except ValueError:

        epoch_auc = float(
            "nan"
        )

    return (
        epoch_loss,
        epoch_accuracy,
        epoch_f1,
        epoch_auc,
        all_labels,
        all_predictions
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed()

    print(
        "=" * 60
    )

    print(
        "RESNET-50 CLASSIFICATION TRAINING"
    )

    print(
        "=" * 60
    )

    print(
        f"Device       : {DEVICE}"
    )

    print(
        f"Epochs       : {NUM_EPOCHS}"
    )

    print(
        f"Batch size   : {BATCH_SIZE}"
    )

    print(
        f"Image size   : {IMAGE_SIZE}"
    )

    print(
        f"Train CSV    : {TRAIN_CSV}"
    )

    print(
        f"Validation   : {VAL_CSV}"
    )

    print(
        f"Output       : {OUTPUT_DIR}"
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    train_df, val_df = load_data()

    print_class_distribution(
        train_df,
        val_df
    )

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
    # SAMPLER
    # ========================================================

    sampler = (
        create_weighted_sampler(
            train_df
        )
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = create_model()

    model = model.to(
        DEVICE
    )

    print(
        "ResNet-50 model ready."
    )

    # ========================================================
    # CLASS-WEIGHTED LOSS
    # ========================================================

    print(
        "\nCreating class-weighted loss..."
    )

    class_counts = (
        train_df["class_id"]
        .value_counts()
        .sort_index()
    )

    class_weights = []

    for class_id in range(
        NUM_CLASSES
    ):

        count = class_counts.get(
            class_id,
            0
        )

        if count > 0:

            class_weights.append(
                1.0 / float(count)
            )

        else:

            class_weights.append(
                0.0
            )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32
    )

    # Normalize average weight to 1

    nonzero_weights = (
        class_weights[
            class_weights > 0
        ]
    )

    if len(nonzero_weights) > 0:

        class_weights = (
            class_weights
            / nonzero_weights.mean()
        )

    class_weights = (
        class_weights.to(
            DEVICE
        )
    )

    print(
        "\nClass weights:"
    )

    for class_id in range(
        NUM_CLASSES
    ):

        print(
            f"{class_id} - "
            f"{CLASS_NAMES[class_id]}: "
            f"{class_weights[class_id].item():.4f}"
        )

    criterion = (
        nn.CrossEntropyLoss(
            weight=class_weights
        )
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ========================================================
    # LR SCHEDULER
    # ========================================================

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=1
        )
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    best_val_loss = float(
        "inf"
    )

    best_val_accuracy = 0.0

    history = []

    final_labels = None

    final_predictions = None

    print(
        "\n" + "=" * 60
    )

    print(
        "STARTING TRAINING"
    )

    print(
        "=" * 60
    )

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        print(
            "\n" + "=" * 60
        )

        print(
            f"Epoch {epoch}/{NUM_EPOCHS}"
        )

        print(
            "=" * 60
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        (
            train_loss,
            train_accuracy,
            train_f1
        ) = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE
        )

        # ----------------------------------------------------
        # VALIDATE
        # ----------------------------------------------------

        (
            val_loss,
            val_accuracy,
            val_f1,
            val_auc,
            val_labels,
            val_predictions
        ) = validate(
            model,
            val_loader,
            criterion,
            DEVICE
        )

        final_labels = (
            val_labels
        )

        final_predictions = (
            val_predictions
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

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
            f"Train loss     : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train accuracy : "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Train F1       : "
            f"{train_f1:.4f}"
        )

        print(
            f"Val loss       : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val accuracy   : "
            f"{val_accuracy:.4f}"
        )

        print(
            f"Val F1         : "
            f"{val_f1:.4f}"
        )

        print(
            f"Val AUC-ROC    : "
            f"{val_auc:.4f}"
        )

        print(
            f"Learning rate  : "
            f"{current_lr:.6f}"
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
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
        # Save best checkpoint
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = (
                val_loss
            )

            best_val_accuracy = (
                val_accuracy
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

                    "val_accuracy":
                        val_accuracy,

                    "val_f1":
                        val_f1,

                    "val_auc":
                        val_auc,

                    "class_names":
                        CLASS_NAMES,

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
    # CONFUSION MATRIX
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "VALIDATION CONFUSION MATRIX"
    )

    print(
        "=" * 60
    )

    cm = confusion_matrix(
        final_labels,
        final_predictions,
        labels=list(
            range(NUM_CLASSES)
        )
    )

    print(cm)

    np.save(
        CONFUSION_MATRIX_PATH,
        cm
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
    # SAVE TRAINING HISTORY
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
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "RESNET-50 TRAINING COMPLETED"
    )

    print(
        "=" * 60
    )

    print(
        f"Best validation loss : "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Best validation accuracy : "
        f"{best_val_accuracy:.4f}"
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

    print(
        f"\nConfusion matrix:"
    )

    print(
        CONFUSION_MATRIX_PATH
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()