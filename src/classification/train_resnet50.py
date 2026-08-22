from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image

from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler,
)

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
)

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
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_PATH = (
    OUTPUT_DIR
    / "resnet50_best.pth"
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 5

IMAGE_SIZE = 224

BATCH_SIZE = 8

NUM_WORKERS = 0

# We already completed a 1-epoch pipeline test.
# Now we perform proper training.
EPOCHS = 5

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
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(
    RANDOM_SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Normal",
    "Cardiomegaly",
    "Pleural effusion",
    "Lung Opacity",
    "Pulmonary fibrosis",
]


# ============================================================
# DATASET
# ============================================================

class ChestXrayDataset(Dataset):

    def __init__(
        self,
        csv_file,
        transform=None
    ):

        self.data = pd.read_csv(
            csv_file
        )

        self.transform = transform


    def __len__(self):

        return len(
            self.data
        )


    def __getitem__(
        self,
        index
    ):

        row = self.data.iloc[
            index
        ]

        image_id = row[
            "image_id"
        ]

        label = int(
            row["class_id"]
        )

        image_path = (
            IMAGE_DIR
            / f"{image_id}.png"
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

# Training transformations.
#
# These introduce small variations so that the model
# does not simply memorize the training images.

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
            degrees=5
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


# Validation images must NOT receive random
# augmentation.

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
# LOAD DATASETS
# ============================================================

print("=" * 60)
print("RESNET-50 TRAINING")
print("=" * 60)

print(
    f"Device : {DEVICE}"
)

print(
    f"Epochs : {EPOCHS}"
)

print(
    f"Batch size : {BATCH_SIZE}"
)


train_csv = (
    DATA_DIR
    / "train.csv"
)

val_csv = (
    DATA_DIR
    / "val.csv"
)


print(
    "\nLoading datasets..."
)


train_dataset = ChestXrayDataset(
    train_csv,
    transform=train_transform
)

val_dataset = ChestXrayDataset(
    val_csv,
    transform=val_transform
)


print(
    f"Training images   : "
    f"{len(train_dataset)}"
)

print(
    f"Validation images : "
    f"{len(val_dataset)}"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

train_labels = (
    train_dataset.data[
        "class_id"
    ].tolist()
)


class_counts = (
    train_dataset.data[
        "class_id"
    ]
    .value_counts()
    .sort_index()
)


print(
    "\nTraining class distribution:"
)


for class_id in range(
    NUM_CLASSES
):

    count = int(
        class_counts.get(
            class_id,
            0
        )
    )

    print(
        f"{class_id} - "
        f"{CLASS_NAMES[class_id]}: "
        f"{count}"
    )


# ============================================================
# WEIGHTED RANDOM SAMPLER
# ============================================================

print(
    "\nCreating WeightedRandomSampler..."
)


# Inverse-frequency weighting:
#
# Frequent class
#       ↓
# smaller weight
#
# Rare class
#       ↓
# larger weight

class_sample_weights = {}

for class_id in range(
    NUM_CLASSES
):

    count = class_counts.get(
        class_id,
        0
    )

    if count > 0:

        class_sample_weights[
            class_id
        ] = 1.0 / count

    else:

        class_sample_weights[
            class_id
        ] = 0.0


sample_weights = [
    class_sample_weights[
        label
    ]
    for label in train_labels
]


sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(
        sample_weights
    ),
    replacement=True
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# CREATE RESNET-50
# ============================================================

print(
    "\nLoading pretrained ResNet-50..."
)


weights = (
    models.ResNet50_Weights.DEFAULT
)


model = models.resnet50(
    weights=weights
)


# ImageNet ResNet-50 originally predicts
# 1000 ImageNet classes.
#
# We replace the final layer with:
#
# 2048 → 5
#
# Our five classes are:
#
# 0 Normal
# 1 Cardiomegaly
# 2 Pleural effusion
# 3 Lung Opacity
# 4 Pulmonary fibrosis

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)


model = model.to(
    DEVICE
)


print(
    "ResNet-50 loaded successfully."
)


# ============================================================
# CLASS-WEIGHTED LOSS
# ============================================================

print(
    "\nCreating class-weighted loss..."
)


counts_tensor = torch.tensor(
    [
        float(
            class_counts.get(
                class_id,
                0
            )
        )
        for class_id in range(
            NUM_CLASSES
        )
    ],
    dtype=torch.float32
)


# Inverse-frequency class weights.

class_weights_tensor = (
    1.0 / counts_tensor
)


# Normalize the weights so their
# average value is approximately 1.

class_weights_tensor = (
    class_weights_tensor
    / class_weights_tensor.mean()
)


class_weights_tensor = (
    class_weights_tensor.to(
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
        f"{class_weights_tensor[class_id].item():.4f}"
    )


criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=1
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    all_predictions = []

    all_labels = []


    for batch_index, (
        images,
        labels
    ) in enumerate(
        train_loader
    ):

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        # ----------------------------------------------------
        # Clear previous gradients
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(
            images
        )


        # ----------------------------------------------------
        # Calculate loss
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
        # Update model weights
        # ----------------------------------------------------

        optimizer.step()


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        running_loss += (
            loss.item()
            * images.size(0)
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        all_predictions.extend(
            predictions
            .detach()
            .cpu()
            .numpy()
        )

        all_labels.extend(
            labels
            .detach()
            .cpu()
            .numpy()
        )


        if (
            (batch_index + 1) % 100
            == 0
        ):

            print(
                f"  Batch "
                f"{batch_index + 1}/"
                f"{len(train_loader)}"
            )


    epoch_loss = (
        running_loss
        / len(train_dataset)
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
# VALIDATION FUNCTION
# ============================================================

def validate():

    model.eval()

    running_loss = 0.0

    all_predictions = []

    all_labels = []

    all_probabilities = []


    with torch.no_grad():

        for images, labels in (
            val_loader
        ):

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            # Forward pass.

            outputs = model(
                images
            )


            # Validation loss.

            loss = criterion(
                outputs,
                labels
            )


            running_loss += (
                loss.item()
                * images.size(0)
            )


            # Convert logits into probabilities.

            probabilities = (
                torch.softmax(
                    outputs,
                    dim=1
                )
            )


            predictions = (
                torch.argmax(
                    outputs,
                    dim=1
                )
            )


            all_predictions.extend(
                predictions
                .cpu()
                .numpy()
            )

            all_labels.extend(
                labels
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
        / len(val_dataset)
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


    probabilities = np.array(
        all_probabilities
    )


    labels_array = np.array(
        all_labels
    )


    # AUC-ROC requires probabilities
    # for every class.

    try:

        epoch_auc = roc_auc_score(
            labels_array,
            probabilities,
            multi_class="ovr",
            average="macro"
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
# TRAINING LOOP
# ============================================================

best_val_loss = float(
    "inf"
)


print(
    "\nStarting training..."
)


for epoch in range(
    1,
    EPOCHS + 1
):

    print(
        "\n" + "=" * 60
    )

    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    print(
        "=" * 60
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    (
        train_loss,
        train_accuracy,
        train_f1
    ) = train_one_epoch()


    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    (
        val_loss,
        val_accuracy,
        val_f1,
        val_auc,
        val_labels,
        val_predictions
    ) = validate()


    # --------------------------------------------------------
    # Update learning rate
    # --------------------------------------------------------

    scheduler.step(
        val_loss
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Save best checkpoint
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss


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
            CHECKPOINT_PATH
        )


        print(
            "\nBest model saved:"
        )

        print(
            CHECKPOINT_PATH
        )


# ============================================================
# CONFUSION MATRIX
# ============================================================

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
    val_labels,
    val_predictions
)


print(cm)


# ============================================================
# FINAL SUMMARY
# ============================================================

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
    f"Checkpoint           : "
    f"{CHECKPOINT_PATH}"
)