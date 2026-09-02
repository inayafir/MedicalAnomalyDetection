
from pathlib import Path
import pandas as pd
import shutil
import random


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_SOURCE_DIR = PROJECT_ROOT / "data" / "raw" / "vinbigdata" / "train"
ANNOTATIONS_FILE = PROJECT_ROOT / "data" / "processed" / "annotations_wbf.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "yolo_dataset"

TRAIN_IMAGE_DIR = OUTPUT_DIR / "images" / "train"
VAL_IMAGE_DIR = OUTPUT_DIR / "images" / "val"

TRAIN_LABEL_DIR = OUTPUT_DIR / "labels" / "train"
VAL_LABEL_DIR = OUTPUT_DIR / "labels" / "val"


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42
VAL_RATIO = 0.20

random.seed(RANDOM_SEED)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for directory in [
    TRAIN_IMAGE_DIR,
    VAL_IMAGE_DIR,
    TRAIN_LABEL_DIR,
    VAL_LABEL_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD WBF ANNOTATIONS
# ============================================================

print("Loading WBF annotations...")

df = pd.read_csv(ANNOTATIONS_FILE)

print(f"Annotation rows: {len(df)}")
print(f"Images with annotations: {df['image_id'].nunique()}")


# ============================================================
# GET ALL TRAIN IMAGES
# ============================================================

image_files = list(IMAGE_SOURCE_DIR.glob("*.png"))

all_image_ids = [image.stem for image in image_files]

print(f"Total source images: {len(all_image_ids)}")


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

random.shuffle(all_image_ids)

split_index = int(len(all_image_ids) * (1 - VAL_RATIO))

train_ids = all_image_ids[:split_index]
val_ids = all_image_ids[split_index:]

train_ids = set(train_ids)
val_ids = set(val_ids)

print(f"Training images: {len(train_ids)}")
print(f"Validation images: {len(val_ids)}")


# ============================================================
# CREATE ANNOTATION LOOKUP
# ============================================================

annotations_by_image = {
    image_id: group
    for image_id, group in df.groupby("image_id")
}


# ============================================================
# FUNCTION TO CREATE YOLO LABEL
# ============================================================

def create_yolo_label(image_id, output_label_file):

    if image_id not in annotations_by_image:

        # Normal image / no abnormality
        output_label_file.touch()

        return 0

    group = annotations_by_image[image_id]

    lines = []

    for _, row in group.iterrows():

        width = row["width"]
        height = row["height"]

        x_min = row["x_min"]
        y_min = row["y_min"]
        x_max = row["x_max"]
        y_max = row["y_max"]

        # Convert bounding box to YOLO format

        x_center = ((x_min + x_max) / 2) / width
        y_center = ((y_min + y_max) / 2) / height

        box_width = (x_max - x_min) / width
        box_height = (y_max - y_min) / height

        class_id = int(row["class_id"])

        lines.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )

    with open(output_label_file, "w") as f:
        f.write("\n".join(lines))

    return len(lines)


# ============================================================
# COPY DATASET
# ============================================================

print("\nPreparing YOLO dataset...")


def process_split(image_ids, image_output_dir, label_output_dir, split_name):

    total_boxes = 0
    empty_labels = 0

    for index, image_id in enumerate(image_ids, start=1):

        source_image = IMAGE_SOURCE_DIR / f"{image_id}.png"

        destination_image = image_output_dir / f"{image_id}.png"
        destination_label = label_output_dir / f"{image_id}.txt"

        # Copy image
        shutil.copy2(source_image, destination_image)

        # Create YOLO label
        boxes = create_yolo_label(
            image_id,
            destination_label
        )

        total_boxes += boxes

        if boxes == 0:
            empty_labels += 1

        if index % 1000 == 0:
            print(
                f"{split_name}: "
                f"{index}/{len(image_ids)} images processed"
            )

    return total_boxes, empty_labels


# ============================================================
# PROCESS TRAINING DATA
# ============================================================

train_boxes, train_empty = process_split(
    train_ids,
    TRAIN_IMAGE_DIR,
    TRAIN_LABEL_DIR,
    "TRAIN"
)


# ============================================================
# PROCESS VALIDATION DATA
# ============================================================

val_boxes, val_empty = process_split(
    val_ids,
    VAL_IMAGE_DIR,
    VAL_LABEL_DIR,
    "VAL"
)


# ============================================================
# CREATE DATASET YAML
# ============================================================

DATASET_YAML = OUTPUT_DIR / "dataset.yaml"

class_names = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
]


yaml_content = f"""path: {OUTPUT_DIR.as_posix()}
train: images/train
val: images/val

nc: {len(class_names)}

names:
"""

for index, name in enumerate(class_names):
    yaml_content += f"  {index}: {name}\n"


with open(DATASET_YAML, "w") as f:
    f.write(yaml_content)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("YOLO DATASET PREPARATION COMPLETED")
print("=" * 60)

print(f"Total images       : {len(all_image_ids)}")
print(f"Training images    : {len(train_ids)}")
print(f"Validation images  : {len(val_ids)}")

print(f"\nTraining boxes     : {train_boxes}")
print(f"Validation boxes   : {val_boxes}")

print(f"\nTraining empty     : {train_empty}")
print(f"Validation empty   : {val_empty}")

print(f"\nDataset YAML       : {DATASET_YAML}")
print(f"Dataset directory  : {OUTPUT_DIR}")

