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
    / "wbf_annotations.csv"
)

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "vinbigdata"
    / "train"
)

YOLO_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "detection"
    / "yolo"
)

TRAIN_IMAGE_DIR = YOLO_DIR / "images" / "train"
VAL_IMAGE_DIR = YOLO_DIR / "images" / "val"

TRAIN_LABEL_DIR = YOLO_DIR / "labels" / "train"
VAL_LABEL_DIR = YOLO_DIR / "labels" / "val"

DATA_YAML = YOLO_DIR / "data.yaml"


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

NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# START
# ============================================================

print("=" * 60)
print("YOLOv8 DATASET PREPARATION")
print("=" * 60)

print(f"Input CSV : {INPUT_CSV}")
print(f"Image Dir : {IMAGE_DIR}")
print(f"YOLO Dir  : {YOLO_DIR}")


# ============================================================
# LOAD WBF ANNOTATIONS
# ============================================================

print("\nLoading WBF annotations...")

df = pd.read_csv(INPUT_CSV)

print(f"Annotation rows : {len(df)}")

print(
    f"Unique images   : "
    f"{df['image_id'].nunique()}"
)


# ============================================================
# VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = [
    "image_id",
    "project_class_id",
    "project_class",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "width",
    "height",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: "
        f"{missing_columns}"
    )

print("Required columns verified.")


# ============================================================
# VERIFY IMAGE FILES
# ============================================================

print("\nChecking image files...")

unique_image_ids = (
    df["image_id"]
    .drop_duplicates()
    .tolist()
)

missing_images = []

for image_id in unique_image_ids:

    image_path = (
        IMAGE_DIR
        / f"{image_id}.png"
    )

    if not image_path.exists():

        missing_images.append(
            image_id
        )


print(
    f"Images checked : "
    f"{len(unique_image_ids)}"
)

print(
    f"Missing images : "
    f"{len(missing_images)}"
)


if missing_images:

    print("\nFirst missing images:")

    for image_id in missing_images[:10]:

        print(
            f"  {image_id}"
        )

    raise FileNotFoundError(
        "Some required images are missing."
    )


# ============================================================
# CREATE YOLO DIRECTORIES
# ============================================================

print("\nCreating YOLO directories...")

for directory in [
    TRAIN_IMAGE_DIR,
    VAL_IMAGE_DIR,
    TRAIN_LABEL_DIR,
    VAL_LABEL_DIR,
]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# IMAGE-LEVEL TRAIN / VALIDATION SPLIT
# ============================================================

print("\nCreating image-level train/validation split...")

random.seed(
    RANDOM_SEED
)

image_ids = list(
    unique_image_ids
)

random.shuffle(
    image_ids
)

split_index = int(
    len(image_ids)
    * TRAIN_RATIO
)

train_image_ids = set(
    image_ids[:split_index]
)

val_image_ids = set(
    image_ids[split_index:]
)


print(
    f"Training images   : "
    f"{len(train_image_ids)}"
)

print(
    f"Validation images : "
    f"{len(val_image_ids)}"
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def convert_to_yolo(
    x_min,
    y_min,
    x_max,
    y_max,
    image_width,
    image_height
):

    # YOLO format:
    #
    # class_id
    # x_center
    # y_center
    # width
    # height
    #
    # All coordinates are normalized
    # between 0 and 1.

    box_width = (
        x_max - x_min
    )

    box_height = (
        y_max - y_min
    )

    x_center = (
        x_min + x_max
    ) / 2.0

    y_center = (
        y_min + y_max
    ) / 2.0

    x_center /= image_width
    y_center /= image_height

    box_width /= image_width
    box_height /= image_height

    return (
        x_center,
        y_center,
        box_width,
        box_height
    )


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(
    image_id,
    image_annotations,
    image_output_dir,
    label_output_dir
):

    image_path = (
        IMAGE_DIR
        / f"{image_id}.png"
    )

    output_image_path = (
        image_output_dir
        / f"{image_id}.png"
    )

    output_label_path = (
        label_output_dir
        / f"{image_id}.txt"
    )


    # --------------------------------------------------------
    # Copy image
    # --------------------------------------------------------

    shutil.copy2(
        image_path,
        output_image_path
    )


    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    first_row = (
        image_annotations.iloc[0]
    )

    image_width = float(
        first_row["width"]
    )

    image_height = float(
        first_row["height"]
    )


    yolo_lines = []


    # --------------------------------------------------------
    # Process annotations
    # --------------------------------------------------------

    for _, row in (
        image_annotations.iterrows()
    ):

        class_id = int(
            row["project_class_id"]
        )


        # ----------------------------------------------------
        # Normal images do not have bounding boxes.
        #
        # YOLO detection labels therefore remain empty.
        # ----------------------------------------------------

        if class_id == 0:

            continue


        # ----------------------------------------------------
        # Skip missing bounding boxes.
        # ----------------------------------------------------

        if pd.isna(
            row["x_min"]
        ) or pd.isna(
            row["y_min"]
        ) or pd.isna(
            row["x_max"]
        ) or pd.isna(
            row["y_max"]
        ):

            continue


        x_min = float(
            row["x_min"]
        )

        y_min = float(
            row["y_min"]
        )

        x_max = float(
            row["x_max"]
        )

        y_max = float(
            row["y_max"]
        )


        # ----------------------------------------------------
        # Validate box
        # ----------------------------------------------------

        if x_max <= x_min:
            continue

        if y_max <= y_min:
            continue


        # ----------------------------------------------------
        # Convert to YOLO format
        # ----------------------------------------------------

        (
            x_center,
            y_center,
            box_width,
            box_height
        ) = convert_to_yolo(
            x_min,
            y_min,
            x_max,
            y_max,
            image_width,
            image_height
        )


        # ----------------------------------------------------
        # Clamp values to valid range
        # ----------------------------------------------------

        x_center = min(
            max(x_center, 0.0),
            1.0
        )

        y_center = min(
            max(y_center, 0.0),
            1.0
        )

        box_width = min(
            max(box_width, 0.0),
            1.0
        )

        box_height = min(
            max(box_height, 0.0),
            1.0
        )


        # ----------------------------------------------------
        # Create YOLO label line
        # ----------------------------------------------------

        yolo_line = (
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )

        yolo_lines.append(
            yolo_line
        )


    # --------------------------------------------------------
    # Write label file
    # --------------------------------------------------------

    with open(
        output_label_path,
        "w"
    ) as file:

        if yolo_lines:

            file.write(
                "\n".join(
                    yolo_lines
                )
            )


# ============================================================
# PROCESS DATASET
# ============================================================

print("\nConverting annotations to YOLO format...")

processed_train = 0
processed_val = 0

train_annotations = df[
    df["image_id"].isin(
        train_image_ids
    )
]

val_annotations = df[
    df["image_id"].isin(
        val_image_ids
    )
]


# ------------------------------------------------------------
# Training images
# ------------------------------------------------------------

print("\nProcessing training images...")

for image_index, (
    image_id,
    group
) in enumerate(
    train_annotations.groupby(
        "image_id",
        sort=False
    ),
    start=1
):

    process_image(
        image_id,
        group,
        TRAIN_IMAGE_DIR,
        TRAIN_LABEL_DIR
    )

    processed_train += 1

    if image_index % 1000 == 0:

        print(
            f"  Training images: "
            f"{image_index}/"
            f"{len(train_image_ids)}"
        )


# ------------------------------------------------------------
# Validation images
# ------------------------------------------------------------

print("\nProcessing validation images...")

for image_index, (
    image_id,
    group
) in enumerate(
    val_annotations.groupby(
        "image_id",
        sort=False
    ),
    start=1
):

    process_image(
        image_id,
        group,
        VAL_IMAGE_DIR,
        VAL_LABEL_DIR
    )

    processed_val += 1

    if image_index % 1000 == 0:

        print(
            f"  Validation images: "
            f"{image_index}/"
            f"{len(val_image_ids)}"
        )


# ============================================================
# WRITE DATA.YAML
# ============================================================

print("\nCreating data.yaml...")

yaml_content = f"""path: {YOLO_DIR.as_posix()}
train: images/train
val: images/val

nc: {NUM_CLASSES}

names:
"""

for class_id, class_name in enumerate(
    CLASS_NAMES
):

    yaml_content += (
        f"  {class_id}: "
        f"{class_name}\n"
    )


with open(
    DATA_YAML,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        yaml_content
    )


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("YOLO DATASET VALIDATION")
print("=" * 60)


train_images = list(
    TRAIN_IMAGE_DIR.glob("*.png")
)

val_images = list(
    VAL_IMAGE_DIR.glob("*.png")
)

train_labels = list(
    TRAIN_LABEL_DIR.glob("*.txt")
)

val_labels = list(
    VAL_LABEL_DIR.glob("*.txt")
)


print(
    f"Train images : "
    f"{len(train_images)}"
)

print(
    f"Train labels : "
    f"{len(train_labels)}"
)

print(
    f"Val images   : "
    f"{len(val_images)}"
)

print(
    f"Val labels   : "
    f"{len(val_labels)}"
)


# ============================================================
# CHECK LABEL VALUES
# ============================================================

invalid_labels = 0

total_detection_boxes = 0

for label_file in (
    train_labels + val_labels
):

    with open(
        label_file,
        "r"
    ) as file:

        lines = [
            line.strip()
            for line in file
            if line.strip()
        ]


    for line in lines:

        parts = line.split()

        if len(parts) != 5:

            invalid_labels += 1
            continue


        class_id = int(
            parts[0]
        )

        values = [
            float(value)
            for value in parts[1:]
        ]


        if not (
            0 <= class_id < NUM_CLASSES
        ):

            invalid_labels += 1
            continue


        if not all(
            0.0 <= value <= 1.0
            for value in values
        ):

            invalid_labels += 1
            continue


        total_detection_boxes += 1


print(
    f"Detection boxes : "
    f"{total_detection_boxes}"
)

print(
    f"Invalid labels   : "
    f"{invalid_labels}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("YOLOv8 DATASET PREPARATION COMPLETED")
print("=" * 60)

print(
    f"Train images : "
    f"{processed_train}"
)

print(
    f"Val images   : "
    f"{processed_val}"
)

print(
    f"Total images : "
    f"{processed_train + processed_val}"
)

print(
    f"Total boxes  : "
    f"{total_detection_boxes}"
)

print(
    f"Invalid labels : "
    f"{invalid_labels}"
)

print(
    f"\ndata.yaml : "
    f"{DATA_YAML}"
)

print("\nClasses:")

for class_id, class_name in enumerate(
    CLASS_NAMES
):

    print(
        f"{class_id} - {class_name}"
    )

print("\nNext stage:")
print("Train YOLOv8 detection model")

print("=" * 60)