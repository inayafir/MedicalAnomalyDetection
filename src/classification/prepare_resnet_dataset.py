from pathlib import Path
import shutil
import random

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "detection"
    / "clean_annotations.csv"
)

IMAGE_SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "vinbigdata"
    / "train"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "classification"
    / "resnet50"
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

CLASS_NAMES = [
    "Normal",
    "Cardiomegaly",
    "Pleural effusion",
    "Lung Opacity",
    "Pulmonary fibrosis",
]

CLASS_TO_ID = {
    "Normal": 0,
    "Cardiomegaly": 1,
    "Pleural effusion": 2,
    "Lung Opacity": 3,
    "Pulmonary fibrosis": 4,
}


# ============================================================
# START
# ============================================================

print("=" * 60)
print("RESNET-50 CLASSIFICATION DATASET PREPARATION")
print("=" * 60)

print(f"Input CSV       : {INPUT_CSV}")
print(f"Image directory : {IMAGE_SOURCE_DIR}")
print(f"Output directory: {OUTPUT_DIR}")


# ============================================================
# CHECK INPUTS
# ============================================================

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Annotation CSV not found:\n{INPUT_CSV}"
    )

if not IMAGE_SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Image directory not found:\n{IMAGE_SOURCE_DIR}"
    )


# ============================================================
# LOAD ANNOTATIONS
# ============================================================

print("\nLoading cleaned annotations...")

df = pd.read_csv(INPUT_CSV)

print(f"Annotation rows: {len(df)}")


# ============================================================
# VERIFY CLASSES
# ============================================================

missing_classes = set(CLASS_NAMES) - set(df["project_class"].unique())

if missing_classes:
    raise ValueError(
        f"Missing expected classes: {missing_classes}"
    )

df = df[df["project_class"].isin(CLASS_NAMES)].copy()


# ============================================================
# IMAGE-LEVEL LABELS
# ============================================================
#
# One image can have multiple annotation rows.
# For classification, each image must have ONE label.
#
# If an image contains multiple abnormal classes, choose the
# abnormal class with the highest number of annotations.
#
# Normal images remain Normal.
# ============================================================

print("\nCreating image-level labels...")

image_labels = []

for image_id, group in df.groupby("image_id"):

    classes = group["project_class"].tolist()

    # Normal image
    if all(cls == "Normal" for cls in classes):

        selected_class = "Normal"

    else:

        abnormal_classes = [
            cls for cls in classes
            if cls != "Normal"
        ]

        # Select the most frequently occurring abnormal class.
        selected_class = (
            pd.Series(abnormal_classes)
            .value_counts()
            .index[0]
        )

    image_labels.append(
        {
            "image_id": image_id,
            "project_class": selected_class,
            "class_id": CLASS_TO_ID[selected_class],
        }
    )


image_df = pd.DataFrame(image_labels)


# ============================================================
# REMOVE MISSING IMAGES
# ============================================================

print("\nChecking image files...")

image_df["image_path"] = image_df["image_id"].apply(
    lambda x: IMAGE_SOURCE_DIR / f"{x}.png"
)

exists_mask = image_df["image_path"].apply(Path.exists)

missing_count = int((~exists_mask).sum())

print(f"Images found   : {exists_mask.sum()}")
print(f"Images missing : {missing_count}")

image_df = image_df[exists_mask].copy()


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n")
print("=" * 60)
print("IMAGE-LEVEL CLASS DISTRIBUTION")
print("=" * 60)

distribution = (
    image_df["project_class"]
    .value_counts()
    .reindex(CLASS_NAMES, fill_value=0)
)

for class_name, count in distribution.items():

    print(
        f"{CLASS_TO_ID[class_name]}   "
        f"{class_name:<22} "
        f"{count}"
    )


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print("\nCreating stratified train/validation split...")

random.seed(RANDOM_SEED)

train_parts = []
val_parts = []

for class_name in CLASS_NAMES:

    class_df = image_df[
        image_df["project_class"] == class_name
    ].copy()

    indices = list(class_df.index)

    random.shuffle(indices)

    split_index = int(len(indices) * TRAIN_RATIO)

    train_indices = indices[:split_index]
    val_indices = indices[split_index:]

    train_parts.append(class_df.loc[train_indices])
    val_parts.append(class_df.loc[val_indices])


train_df = pd.concat(train_parts).sample(
    frac=1,
    random_state=RANDOM_SEED
)

val_df = pd.concat(val_parts).sample(
    frac=1,
    random_state=RANDOM_SEED
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

train_dir = OUTPUT_DIR / "train"
val_dir = OUTPUT_DIR / "val"

for class_name in CLASS_NAMES:

    (train_dir / class_name).mkdir(
        parents=True,
        exist_ok=True
    )

    (val_dir / class_name).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# COPY IMAGES
# ============================================================

print("\nCopying training images...")

for _, row in train_df.iterrows():

    source = Path(row["image_path"])

    destination = (
        train_dir
        / row["project_class"]
        / source.name
    )

    shutil.copy2(source, destination)


print("Copying validation images...")

for _, row in val_df.iterrows():

    source = Path(row["image_path"])

    destination = (
        val_dir
        / row["project_class"]
        / source.name
    )

    shutil.copy2(source, destination)


# ============================================================
# SAVE CSV FILES
# ============================================================

train_csv = OUTPUT_DIR / "train.csv"
val_csv = OUTPUT_DIR / "val.csv"
labels_csv = OUTPUT_DIR / "image_labels.csv"

train_df.drop(
    columns=["image_path"]
).to_csv(
    train_csv,
    index=False
)

val_df.drop(
    columns=["image_path"]
).to_csv(
    val_csv,
    index=False
)

image_df.drop(
    columns=["image_path"]
).to_csv(
    labels_csv,
    index=False
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n")
print("=" * 60)
print("RESNET-50 DATASET VALIDATION")
print("=" * 60)

print(f"Total images      : {len(image_df)}")
print(f"Training images   : {len(train_df)}")
print(f"Validation images : {len(val_df)}")

print("\nTraining distribution:")

print(
    train_df["project_class"]
    .value_counts()
    .reindex(CLASS_NAMES, fill_value=0)
    .to_string()
)

print("\nValidation distribution:")

print(
    val_df["project_class"]
    .value_counts()
    .reindex(CLASS_NAMES, fill_value=0)
    .to_string()
)

print("\n")
print("=" * 60)
print("RESNET-50 DATASET PREPARATION COMPLETED")
print("=" * 60)

print(f"Output directory: {OUTPUT_DIR}")
print(f"Train CSV       : {train_csv}")
print(f"Validation CSV  : {val_csv}")
print(f"Labels CSV      : {labels_csv}")

print("\nNext stage:")
print("Create ResNet-50 training pipeline")