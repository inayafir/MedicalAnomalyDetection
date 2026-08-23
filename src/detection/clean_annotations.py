from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "vinbigdata"
)

IMAGE_DIR = (
    RAW_DIR
    / "train"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "detection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INPUT_CSV = (
    RAW_DIR
    / "train.csv"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "clean_annotations.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Official VinBigData class names that we are using.

CLASS_MAPPING = {
    "No finding": "Normal",
    "Cardiomegaly": "Cardiomegaly",
    "Pleural effusion": "Pleural effusion",
    "Lung Opacity": "Lung Opacity",
    "Pulmonary fibrosis": "Pulmonary fibrosis",
}


CLASS_IDS = {
    "Normal": 0,
    "Cardiomegaly": 1,
    "Pleural effusion": 2,
    "Lung Opacity": 3,
    "Pulmonary fibrosis": 4,
}


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("VINBIGDATA ANNOTATION CLEANING")
print("=" * 60)

print(
    f"Input CSV  : {INPUT_CSV}"
)

print(
    f"Image Dir  : {IMAGE_DIR}"
)

print(
    f"Output CSV : {OUTPUT_CSV}"
)


# ============================================================
# LOAD ANNOTATIONS
# ============================================================

print(
    "\nLoading annotations..."
)

if not INPUT_CSV.exists():

    raise FileNotFoundError(
        f"Input CSV not found:\n{INPUT_CSV}"
    )


df = pd.read_csv(
    INPUT_CSV
)

print(
    f"Raw annotation rows : {len(df)}"
)


# ============================================================
# VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = [
    "image_id",
    "class_name",
    "class_id",
    "rad_id",
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
        "Missing required columns: "
        + str(missing_columns)
    )


print(
    "Required columns verified."
)


# ============================================================
# SELECT FIVE PROJECT CLASSES
# ============================================================

print(
    "\nSelecting project classes..."
)

df = df[
    df["class_name"].isin(
        CLASS_MAPPING.keys()
    )
].copy()


print(
    f"Rows after class filtering : "
    f"{len(df)}"
)


# ============================================================
# MAP CLASS NAMES
# ============================================================

df["project_class"] = (
    df["class_name"]
    .map(CLASS_MAPPING)
)


df["project_class_id"] = (
    df["project_class"]
    .map(CLASS_IDS)
)


# ============================================================
# CHECK IMAGE FILES
# ============================================================

print(
    "\nChecking PNG image files..."
)


df["image_path"] = (
    df["image_id"]
    .astype(str)
    .apply(
        lambda x:
        str(
            IMAGE_DIR
            / f"{x}.png"
        )
    )
)


df["image_exists"] = (
    df["image_path"]
    .apply(
        lambda x:
        Path(x).exists()
    )
)


missing_images = df[
    ~df["image_exists"]
]


print(
    f"Annotation rows with existing images : "
    f"{df['image_exists'].sum()}"
)

print(
    f"Annotation rows with missing images  : "
    f"{len(missing_images)}"
)


if len(missing_images) > 0:

    print(
        "\nWARNING: Some annotation images "
        "were not found."
    )

    print(
        missing_images[
            "image_id"
        ]
        .drop_duplicates()
        .head(20)
        .to_string(
            index=False
        )
    )


# Keep only annotations whose images exist.

df = df[
    df["image_exists"]
].copy()


# ============================================================
# HANDLE NORMAL CLASS
# ============================================================

# "No finding" does not contain a bounding box.
#
# This is expected.
#
# Normal images will therefore have NaN
# bounding-box coordinates.

normal_mask = (
    df["project_class"]
    == "Normal"
)


abnormal_mask = (
    ~normal_mask
)


print(
    "\nNormal annotations : "
    f"{normal_mask.sum()}"
)

print(
    "Abnormal annotations : "
    f"{abnormal_mask.sum()}"
)


# ============================================================
# VALIDATE ABNORMAL BOUNDING BOXES
# ============================================================

print(
    "\nValidating bounding boxes..."
)


abnormal_df = df[
    abnormal_mask
].copy()


bbox_columns = [
    "x_min",
    "y_min",
    "x_max",
    "y_max",
]


missing_bbox = abnormal_df[
    bbox_columns
].isna().any(
    axis=1
)


invalid_bbox = (
    (abnormal_df["x_min"] < 0)
    | (abnormal_df["y_min"] < 0)
    | (
        abnormal_df["x_max"]
        <= abnormal_df["x_min"]
    )
    | (
        abnormal_df["y_max"]
        <= abnormal_df["y_min"]
    )
    | (
        abnormal_df["x_max"]
        > abnormal_df["width"]
    )
    | (
        abnormal_df["y_max"]
        > abnormal_df["height"]
    )
)


print(
    f"Abnormal rows missing bbox : "
    f"{missing_bbox.sum()}"
)

print(
    f"Abnormal rows with invalid bbox : "
    f"{invalid_bbox.sum()}"
)


# Remove abnormal rows with missing/invalid boxes.

abnormal_df = abnormal_df[
    ~missing_bbox
    & ~invalid_bbox
].copy()


# ============================================================
# NORMAL IMAGE IDS
# ============================================================

normal_df = df[
    normal_mask
].copy()


# A normal image should not have a bounding box.

normal_df[
    bbox_columns
] = None


# ============================================================
# CREATE CLEAN DATAFRAME
# ============================================================

clean_columns = [
    "image_id",
    "project_class",
    "project_class_id",
    "rad_id",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "width",
    "height",
]


normal_clean = normal_df[
    clean_columns
].copy()


abnormal_clean = abnormal_df[
    clean_columns
].copy()


clean_df = pd.concat(
    [
        normal_clean,
        abnormal_clean,
    ],
    ignore_index=True
)


# ============================================================
# REMOVE EXACT DUPLICATES
# ============================================================

before_duplicates = len(
    clean_df
)


clean_df = clean_df.drop_duplicates(
    subset=[
        "image_id",
        "project_class",
        "rad_id",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
    ]
)


removed_duplicates = (
    before_duplicates
    - len(clean_df)
)


print(
    "\nExact duplicate rows removed : "
    f"{removed_duplicates}"
)


# ============================================================
# SORT
# ============================================================

clean_df = clean_df.sort_values(
    by=[
        "image_id",
        "project_class_id",
        "rad_id",
    ]
).reset_index(
    drop=True
)


# ============================================================
# FINAL CLASS DISTRIBUTION
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "FINAL FIVE-CLASS DISTRIBUTION"
)

print(
    "=" * 60
)


distribution = (
    clean_df
    .groupby(
        [
            "project_class_id",
            "project_class",
        ]
    )
    .size()
    .sort_index()
)


print(
    distribution.to_string()
)


# ============================================================
# IMAGE-LEVEL DISTRIBUTION
# ============================================================

print(
    "\nImage-level class distribution:"
)


image_distribution = (
    clean_df[
        [
            "image_id",
            "project_class",
        ]
    ]
    .drop_duplicates()
    .groupby(
        "project_class"
    )
    .size()
)


print(
    image_distribution.to_string()
)


# ============================================================
# SAVE
# ============================================================

clean_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "ANNOTATION CLEANING COMPLETED"
)

print(
    "=" * 60
)

print(
    f"Final annotation rows : "
    f"{len(clean_df)}"
)

print(
    f"Unique images         : "
    f"{clean_df['image_id'].nunique()}"
)

print(
    f"Output file           : "
    f"{OUTPUT_CSV}"
)

print(
    "\nNext stage:"
)

print(
    "Weighted Box Fusion (WBF)"
)

print(
    "=" * 60
)