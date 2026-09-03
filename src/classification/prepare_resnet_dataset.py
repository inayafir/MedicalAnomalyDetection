import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil


# =========================
# PATHS
# =========================
INPUT_CSV = Path("data/processed/detection/clean_annotations.csv")
IMAGE_SOURCE_DIR = Path("data/raw/vinbigdata/train")
OUTPUT_DIR = Path("data/processed/classification/resnet50")


# =========================
# 15 CLASSES
# =========================
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

CLASS_TO_ID = {
    name: idx for idx, name in enumerate(CLASS_NAMES)
}


# =========================
# LOAD ANNOTATIONS
# =========================
print("Loading annotations...")
df = pd.read_csv(INPUT_CSV)

print(f"Annotation rows: {len(df)}")
print(f"Unique images: {df['image_id'].nunique()}")


# =========================
# CREATE MULTI-LABEL TARGETS
# =========================
print("\nCreating multi-label targets...")

image_labels = []

for image_id, group in df.groupby("image_id"):

    classes = set(group["project_class"])

    # If an image has any abnormality,
    # remove Normal from its labels.
    abnormal_classes = classes - {"Normal"}

    if abnormal_classes:
        classes = abnormal_classes
    else:
        classes = {"Normal"}

    labels = [0] * len(CLASS_NAMES)

    for class_name in classes:
        if class_name in CLASS_TO_ID:
            labels[CLASS_TO_ID[class_name]] = 1

    row = {
        "image_id": image_id,
    }

    for i, class_name in enumerate(CLASS_NAMES):
        row[class_name] = labels[i]

    image_labels.append(row)


labels_df = pd.DataFrame(image_labels)


# =========================
# VERIFY LABELS
# =========================
print(f"Images after grouping: {len(labels_df)}")

print("\nLabel counts:")
for class_name in CLASS_NAMES:
    print(f"{class_name:25s}: {labels_df[class_name].sum()}")


# =========================
# SPLIT DATA
# =========================
train_df, val_df = train_test_split(
    labels_df,
    test_size=0.20,
    random_state=42,
    shuffle=True,
)


print("\nDataset split:")
print(f"Train: {len(train_df)}")
print(f"Val:   {len(val_df)}")


# =========================
# PREPARE OUTPUT DIRECTORIES
# =========================
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

(OUTPUT_DIR / "train").mkdir(parents=True)
(OUTPUT_DIR / "val").mkdir(parents=True)


# =========================
# COPY IMAGES
# =========================
def copy_images(split_df, split_name):

    destination = OUTPUT_DIR / split_name

    copied = 0
    missing = 0

    for _, row in split_df.iterrows():

        image_id = row["image_id"]

        source = IMAGE_SOURCE_DIR / f"{image_id}.png"
        target = destination / f"{image_id}.png"

        if source.exists():
            shutil.copy2(source, target)
            copied += 1
        else:
            missing += 1

    print(
        f"{split_name}: copied={copied}, missing={missing}"
    )


copy_images(train_df, "train")
copy_images(val_df, "val")


# =========================
# SAVE CSV FILES
# =========================
train_df.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False
)

val_df.to_csv(
    OUTPUT_DIR / "val.csv",
    index=False
)

labels_df.to_csv(
    OUTPUT_DIR / "image_labels.csv",
    index=False
)


# =========================
# SAVE CLASS NAMES
# =========================
with open(OUTPUT_DIR / "class_names.txt", "w", encoding="utf-8") as f:
    for i, class_name in enumerate(CLASS_NAMES):
        f.write(f"{i}: {class_name}\n")


print("\nDataset preparation complete!")
print(f"Output: {OUTPUT_DIR}")