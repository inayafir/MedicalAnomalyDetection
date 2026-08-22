from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms


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


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 5
IMAGE_SIZE = 224

# Small value for the first pipeline test.
# We will increase this later for actual training.
TEST_BATCH_SIZE = 8

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
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

    def __init__(self, csv_file, transform=None):

        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_id = row["image_id"]
        label = int(row["class_id"])

        image_path = IMAGE_DIR / f"{image_id}.png"

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

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
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ]
)


val_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ]
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("RESNET-50 PIPELINE TEST")
print("=" * 60)

print(f"Device: {DEVICE}")

train_csv = DATA_DIR / "train.csv"
val_csv = DATA_DIR / "val.csv"

print("\nLoading datasets...")

train_dataset = ChestXrayDataset(
    train_csv,
    transform=train_transform
)

val_dataset = ChestXrayDataset(
    val_csv,
    transform=val_transform
)

print(f"Training images   : {len(train_dataset)}")
print(f"Validation images : {len(val_dataset)}")


# ============================================================
# WEIGHTED RANDOM SAMPLER
# ============================================================

print("\nCreating WeightedRandomSampler...")

train_labels = train_dataset.data["class_id"].tolist()

class_counts = (
    train_dataset.data["class_id"]
    .value_counts()
    .sort_index()
)

print("\nClass counts:")

for class_id, count in class_counts.items():

    print(
        f"{class_id} - "
        f"{CLASS_NAMES[class_id]}: "
        f"{count}"
    )


# Inverse-frequency weighting.
#
# Rare class → larger weight
# Common class → smaller weight

class_weights = {
    class_id: 1.0 / count
    for class_id, count in class_counts.items()
}


sample_weights = [
    class_weights[label]
    for label in train_labels
]


sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=TEST_BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# TEST DATA LOADING
# ============================================================

print("\nTesting training DataLoader...")

images, labels = next(iter(train_loader))

print(
    f"Image batch shape : {images.shape}"
)

print(
    f"Label batch shape : {labels.shape}"
)

print(
    f"Labels in batch   : {labels.tolist()}"
)


# ============================================================
# CREATE RESNET-50
# ============================================================

print("\nCreating ResNet-50...")

weights = models.ResNet50_Weights.DEFAULT

model = models.resnet50(
    weights=weights
)


# Replace the original ImageNet classifier.

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)


model = model.to(DEVICE)


# ============================================================
# TEST FORWARD PASS
# ============================================================

print("\nTesting ResNet-50 forward pass...")

images = images.to(DEVICE)

with torch.no_grad():

    outputs = model(images)


print(
    f"Model output shape : {outputs.shape}"
)

print(
    f"Expected shape     : "
    f"({TEST_BATCH_SIZE}, {NUM_CLASSES})"
)


# ============================================================
# TEST LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()

labels = labels.to(DEVICE)

loss = criterion(
    outputs,
    labels
)

print(
    f"\nTest loss          : "
    f"{loss.item():.4f}"
)


# ============================================================
# SUCCESS
# ============================================================

print("\n" + "=" * 60)
print("RESNET-50 PIPELINE TEST PASSED")
print("=" * 60)

print("Dataset loading       : OK")
print("Weighted sampler      : OK")
print("Image transformations : OK")
print("ResNet-50             : OK")
print("5-class output        : OK")
print("Forward pass          : OK")
print("Loss calculation      : OK")

print("\nReady for actual training.")