
from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = PROJECT_ROOT / "data" / "raw" / "vinbigdata" / "train.csv"
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "vinbigdata" / "train"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "labels"


# ============================================================
# CONFIGURATION
# ============================================================

# We train on 14 abnormal classes.
# Class 14 = "No finding", so it is NOT used as a YOLO class.
VALID_CLASSES = list(range(14))


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading train.csv...")

df = pd.read_csv(CSV_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# REMOVE OLD LABEL FILES
# ============================================================

print("Cleaning old label files...")

for label_file in OUTPUT_DIR.glob("*.txt"):
    label_file.unlink()


# ============================================================
# CREATE LABELS FOR ALL TRAIN IMAGES
# ============================================================

processed_images = 0
images_with_boxes = 0
empty_labels = 0
total_boxes = 0


# Group only abnormal annotations.
abnormal_df = df[df["class_id"].isin(VALID_CLASSES)].copy()

print(f"Rows after removing 'No finding': {len(abnormal_df)}")


# Create a dictionary:
# image_id -> annotations
grouped = {
    image_id: group
    for image_id, group in abnormal_df.groupby("image_id")
}


# Process EVERY image in the train directory
for image_path in IMAGE_DIR.glob("*.png"):

    image_id = image_path.stem

    label_file = OUTPUT_DIR / f"{image_id}.txt"

    yolo_lines = []


    # --------------------------------------------------------
    # IMAGE HAS ABNORMAL ANNOTATIONS
    # --------------------------------------------------------

    if image_id in grouped:

        group = grouped[image_id]

        for _, row in group.iterrows():

            width = row["width"]
            height = row["height"]

            x_min = row["x_min"]
            y_min = row["y_min"]
            x_max = row["x_max"]
            y_max = row["y_max"]


            # Skip invalid bounding boxes
            if (
                pd.isna(x_min)
                or pd.isna(y_min)
                or pd.isna(x_max)
                or pd.isna(y_max)
            ):
                continue


            # ------------------------------------------------
            # CONVERT TO YOLO FORMAT
            # ------------------------------------------------

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2

            box_width = x_max - x_min
            box_height = y_max - y_min


            # Normalize to 0-1
            x_center /= width
            y_center /= height

            box_width /= width
            box_height /= height


            class_id = int(row["class_id"])


            # ------------------------------------------------
            # YOLO LABEL FORMAT
            # ------------------------------------------------

            yolo_lines.append(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}"
            )

            total_boxes += 1


    # --------------------------------------------------------
    # WRITE LABEL FILE
    # --------------------------------------------------------

    with open(label_file, "w") as f:
        f.write("\n".join(yolo_lines))


    processed_images += 1


    if yolo_lines:
        images_with_boxes += 1
    else:
        empty_labels += 1


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("YOLO LABEL CONVERSION COMPLETED")
print("=" * 60)

print(f"Total train images : {processed_images}")
print(f"Images with boxes  : {images_with_boxes}")
print(f"Empty label files  : {empty_labels}")
print(f"Total boxes        : {total_boxes}")
print(f"Output directory   : {OUTPUT_DIR}")


