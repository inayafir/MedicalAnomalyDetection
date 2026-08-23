import os

import json

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms

from PIL import Image

import pandas as pd

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

# Test CSV
TEST_CSV = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "classification",
    "test.csv"
)

# IMPORTANT:
# Training script uses images from the original VinBigData
# training image directory.
IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "vinbigdata",
    "train"
)

# ResNet-50 checkpoint
CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "classification",
    "resnet50_best.pth"
)

# Output directory
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "classification"
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_CLASSES = 5

CLASS_NAMES = [
    "Normal",
    "Cardiomegaly",
    "Pleural effusion",
    "Lung Opacity",
    "Pulmonary fibrosis",
]


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("RESNET-50 TEST EVALUATION")
print("=" * 60)

print(f"Device     : {device}")
print(f"Test CSV   : {TEST_CSV}")
print(f"Image Dir  : {IMAGE_DIR}")
print(f"Checkpoint : {CHECKPOINT}")


# ============================================================
# PATH VALIDATION
# ============================================================

print("\nChecking paths...")

if not os.path.exists(TEST_CSV):
    raise FileNotFoundError(
        f"Test CSV not found:\n{TEST_CSV}"
    )

if not os.path.isdir(IMAGE_DIR):
    raise FileNotFoundError(
        f"Image directory not found:\n{IMAGE_DIR}"
    )

if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(
        f"ResNet-50 checkpoint not found:\n{CHECKPOINT}"
    )

print("All required paths found.")


# ============================================================
# DATASET
# ============================================================

class XRayDataset(Dataset):

    def __init__(self, dataframe, transform=None):

        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_id = str(
            row["image_id"]
        )

        label = int(
            row["class_id"]
        )

        image_path = os.path.join(
            IMAGE_DIR,
            f"{image_id}.png"
        )

        # Give a clear error if a particular image is missing
        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"\nImage not found:\n{image_path}\n"
                f"Image ID: {image_id}\n"
                f"Check that the VinBigData train images "
                f"are present in:\n{IMAGE_DIR}"
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
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_df = pd.read_csv(
    TEST_CSV
)

print(
    f"Test images : {len(test_df)}"
)


# ============================================================
# VALIDATE CSV COLUMNS
# ============================================================

required_columns = [
    "image_id",
    "class_id",
]

missing_columns = [
    column
    for column in required_columns
    if column not in test_df.columns
]

if missing_columns:

    raise ValueError(
        "Required columns missing from test.csv: "
        + ", ".join(missing_columns)
    )


# ============================================================
# TEST CLASS DISTRIBUTION
# ============================================================

print("\nTest class distribution:")

for class_id, class_name in enumerate(
    CLASS_NAMES
):

    count = int(
        (
            test_df["class_id"]
            == class_id
        ).sum()
    )

    print(
        f"{class_id} - "
        f"{class_name}: "
        f"{count}"
    )


# ============================================================
# CHECK TEST IMAGES BEFORE EVALUATION
# ============================================================

print("\nChecking test image files...")

missing_images = []

for image_id in test_df["image_id"]:

    image_path = os.path.join(
        IMAGE_DIR,
        f"{image_id}.png"
    )

    if not os.path.exists(image_path):

        missing_images.append(
            image_id
        )

if missing_images:

    print(
        f"\nERROR: {len(missing_images)} "
        f"test images were not found."
    )

    print("\nFirst missing images:")

    for image_id in missing_images[:10]:

        print(
            f"  {image_id}.png"
        )

    raise FileNotFoundError(
        "\nSome test images are missing. "
        "Evaluation cannot continue."
    )

print(
    f"All {len(test_df)} test images found."
)


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

test_transform = transforms.Compose([

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
])


# ============================================================
# CREATE DATASET
# ============================================================

test_dataset = XRayDataset(
    test_df,
    transform=test_transform
)


# ============================================================
# CREATE DATALOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOAD RESNET-50
# ============================================================

print("\nLoading ResNet-50...")

model = models.resnet50(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("Loading checkpoint...")

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

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

    model.load_state_dict(
        checkpoint
    )


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

model = model.to(device)

model.eval()

print(
    "Model loaded successfully."
)


# ============================================================
# EVALUATION
# ============================================================

print(
    "\nRunning test evaluation..."
)

all_labels = []
all_predictions = []
all_probabilities = []


with torch.no_grad():

    for batch_idx, (
        images,
        labels
    ) in enumerate(test_loader):

        images = images.to(
            device
        )

        outputs = model(
            images
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_labels.extend(
            labels.numpy()
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

        if (
            batch_idx + 1
        ) % 20 == 0:

            print(
                f"  Batch "
                f"{batch_idx + 1}/"
                f"{len(test_loader)}"
            )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

all_labels = np.array(
    all_labels
)

all_predictions = np.array(
    all_predictions
)

all_probabilities = np.array(
    all_probabilities
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)


# ============================================================
# AUC-ROC
# ============================================================

try:

    auc_roc = roc_auc_score(
        all_labels,
        all_probabilities,
        multi_class="ovr",
        average="macro"
    )

except ValueError:

    auc_roc = None


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(
        range(NUM_CLASSES)
    )
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(
        range(NUM_CLASSES)
    ),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "TEST RESULTS"
)

print(
    "=" * 60
)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Macro F1 : {macro_f1:.4f}"
)

if auc_roc is not None:

    print(
        f"AUC-ROC  : {auc_roc:.4f}"
    )

else:

    print(
        "AUC-ROC  : "
        "Could not be calculated"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

print(
    "\nConfusion Matrix:"
)

print(cm)


# ============================================================
# PER-CLASS METRICS
# ============================================================

print(
    "\nPer-class metrics:"
)

for class_name in CLASS_NAMES:

    precision = report[
        class_name
    ]["precision"]

    recall = report[
        class_name
    ]["recall"]

    f1 = report[
        class_name
    ]["f1-score"]

    print(
        f"{class_name}: "
        f"Precision={precision:.4f}, "
        f"Recall={recall:.4f}, "
        f"F1={f1:.4f}"
    )


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SAVE TEST METRICS
# ============================================================

metrics = {

    "accuracy": float(
        accuracy
    ),

    "macro_f1": float(
        macro_f1
    ),

    "auc_roc": (
        float(auc_roc)
        if auc_roc is not None
        else None
    ),

    "test_samples": int(
        len(test_df)
    ),

    "num_classes": NUM_CLASSES,

    "class_names": CLASS_NAMES,

    "confusion_matrix": cm.tolist()
}


metrics_path = os.path.join(
    OUTPUT_DIR,
    "test_metrics.json"
)


with open(
    metrics_path,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_df = pd.DataFrame(
    report
).transpose()


report_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.csv"
)


report_df.to_csv(
    report_path
)


# ============================================================
# SAVE CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.imshow(
    cm
)

plt.title(
    "ResNet-50 Test Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    range(NUM_CLASSES),
    CLASS_NAMES,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(NUM_CLASSES),
    CLASS_NAMES
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "True Class"
)


for i in range(
    NUM_CLASSES
):

    for j in range(
        NUM_CLASSES
    ):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()


cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)


plt.savefig(
    cm_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "TEST EVALUATION COMPLETED"
)

print(
    "=" * 60
)

print(
    f"Metrics JSON       : "
    f"{metrics_path}"
)

print(
    f"Classification CSV : "
    f"{report_path}"
)

print(
    f"Confusion Matrix   : "
    f"{cm_path}"
)

print(
    "=" * 60
)