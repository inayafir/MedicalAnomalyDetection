import os
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from PIL import Image

import pandas as pd
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

TEST_CSV = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "classification",
    "test.csv"
)

IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "vinbigdata",
    "train"
)

CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "classification",
    "resnet50",
    "best_model.pth"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "classification",
    "resnet50",
    "diagnostic"
)

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

print("=" * 60)
print("RESNET-50 DIAGNOSTIC ANALYSIS")
print("=" * 60)

print(f"Device     : {device}")
print(f"Test CSV   : {os.path.abspath(TEST_CSV)}")
print(f"Image Dir  : {os.path.abspath(IMAGE_DIR)}")
print(f"Checkpoint : {os.path.abspath(CHECKPOINT)}")


# ============================================================
# DATASET
# ============================================================

class XRayDataset(Dataset):

    def __init__(self, dataframe, image_dir, transform=None):

        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_id = str(row["image_id"])
        label = int(row["class_id"])

        # VinBigData images are stored as image_id.png
        image_path = os.path.join(
            self.image_dir,
            image_id + ".png"
        )

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_df = pd.read_csv(TEST_CSV)

print(f"Test images : {len(test_df)}")

print("\nTest class distribution:")

for class_id, class_name in enumerate(CLASS_NAMES):

    count = int(
        (test_df["class_id"] == class_id).sum()
    )

    print(
        f"{class_id} - {class_name}: {count}"
    )


# ============================================================
# CHECK IMAGE FILES
# ============================================================

print("\nChecking test image files...")

missing_images = []

for image_id in test_df["image_id"]:

    image_path = os.path.join(
        IMAGE_DIR,
        str(image_id) + ".png"
    )

    if not os.path.exists(image_path):
        missing_images.append(image_path)

if missing_images:

    print(
        f"ERROR: {len(missing_images)} images are missing."
    )

    print("First missing image:")
    print(missing_images[0])

    raise FileNotFoundError(
        "Some test images could not be found."
    )

else:

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
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# DATASET / DATALOADER
# ============================================================

test_dataset = XRayDataset(
    dataframe=test_df,
    image_dir=IMAGE_DIR,
    transform=test_transform
)

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
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )

model = model.to(device)

model.eval()

print("Model loaded successfully.")


# ============================================================
# DIAGNOSTIC INFERENCE
# ============================================================

print("\nRunning diagnostic inference...")

all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():

    for batch_idx, (images, labels) in enumerate(
        test_loader
    ):

        images = images.to(device)

        outputs = model(images)

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
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

        if (batch_idx + 1) % 20 == 0:

            print(
                f"  Batch {batch_idx + 1}/{len(test_loader)}"
            )


# ============================================================
# NUMPY ARRAYS
# ============================================================

all_labels = np.array(all_labels)

all_predictions = np.array(
    all_predictions
)

all_probabilities = np.array(
    all_probabilities
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES))
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

prediction_counts = np.bincount(
    all_predictions,
    minlength=NUM_CLASSES
)

true_counts = np.bincount(
    all_labels,
    minlength=NUM_CLASSES
)


# ============================================================
# PRINT DIAGNOSTIC RESULTS
# ============================================================

print("\n" + "=" * 60)
print("DIAGNOSTIC RESULTS")
print("=" * 60)

print("\nTrue class distribution:")

for i, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_name:20s}: {true_counts[i]}"
    )


print("\nPredicted class distribution:")

for i, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_name:20s}: {prediction_counts[i]}"
    )


print("\nPrediction percentage:")

total_predictions = len(all_predictions)

for i, class_name in enumerate(CLASS_NAMES):

    percentage = (
        prediction_counts[i]
        / total_predictions
        * 100
    )

    print(
        f"{class_name:20s}: "
        f"{percentage:.2f}%"
    )


print("\nConfusion Matrix:")

print(cm)


print("\nPer-class metrics:")

for class_name in CLASS_NAMES:

    precision = report[class_name]["precision"]

    recall = report[class_name]["recall"]

    f1 = report[class_name]["f1-score"]

    print(
        f"{class_name:20s} "
        f"Precision={precision:.4f} "
        f"Recall={recall:.4f} "
        f"F1={f1:.4f}"
    )


# ============================================================
# ANALYZE CONFIDENCE
# ============================================================

max_probabilities = np.max(
    all_probabilities,
    axis=1
)

print("\nConfidence analysis:")

print(
    f"Mean confidence : "
    f"{np.mean(max_probabilities):.4f}"
)

print(
    f"Median confidence : "
    f"{np.median(max_probabilities):.4f}"
)

print(
    f"Minimum confidence : "
    f"{np.min(max_probabilities):.4f}"
)

print(
    f"Maximum confidence : "
    f"{np.max(max_probabilities):.4f}"
)


# ============================================================
# SAVE DIAGNOSTIC RESULTS
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

diagnostic_results = {

    "test_samples": int(len(test_df)),

    "true_class_distribution": {
        CLASS_NAMES[i]: int(true_counts[i])
        for i in range(NUM_CLASSES)
    },

    "predicted_class_distribution": {
        CLASS_NAMES[i]: int(prediction_counts[i])
        for i in range(NUM_CLASSES)
    },

    "confusion_matrix": cm.tolist(),

    "per_class_metrics": {
        class_name: {
            "precision": float(
                report[class_name]["precision"]
            ),
            "recall": float(
                report[class_name]["recall"]
            ),
            "f1": float(
                report[class_name]["f1-score"]
            )
        }

        for class_name in CLASS_NAMES
    },

    "confidence": {

        "mean": float(
            np.mean(max_probabilities)
        ),

        "median": float(
            np.median(max_probabilities)
        ),

        "minimum": float(
            np.min(max_probabilities)
        ),

        "maximum": float(
            np.max(max_probabilities)
        )
    }
}


output_path = os.path.join(
    OUTPUT_DIR,
    "resnet50_diagnostic.json"
)

with open(
    output_path,
    "w"
) as f:

    json.dump(
        diagnostic_results,
        f,
        indent=4
    )


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("DIAGNOSTIC ANALYSIS COMPLETED")
print("=" * 60)

print(
    f"Diagnostic JSON : {os.path.abspath(output_path)}"
)