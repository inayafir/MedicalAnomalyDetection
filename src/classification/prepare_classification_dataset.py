from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "vinbigdata"
    / "train.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "classification"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# Five classes required by our classification pipeline.
CLASS_MAPPING = {
    "Normal": 0,
    "Cardiomegaly": 1,
    "Pleural effusion": 2,
    "Lung Opacity": 3,
    "Pulmonary fibrosis": 4,
}

TARGET_ABNORMAL_CLASSES = [
    "Cardiomegaly",
    "Pleural effusion",
    "Lung Opacity",
    "Pulmonary fibrosis",
]


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading train.csv...")

df = pd.read_csv(CSV_PATH)

print(f"Original annotation rows : {len(df)}")
print(f"Original unique images   : {df['image_id'].nunique()}")


# ============================================================
# STEP 1 — IDENTIFY NORMAL IMAGES
# ============================================================

print("\nIdentifying normal images...")

normal_images = set(
    df.loc[
        df["class_name"] == "No finding",
        "image_id"
    ].unique()
)

print(f"Normal images found : {len(normal_images)}")


# ============================================================
# STEP 2 — FIND TARGET ABNORMALITIES
# ============================================================

print("\nFinding target abnormalities...")

abnormal_df = df[
    df["class_name"].isin(TARGET_ABNORMAL_CLASSES)
].copy()

# For every image, find which of the four target abnormalities
# are present.
image_abnormalities = (
    abnormal_df
    .groupby("image_id")["class_name"]
    .unique()
)


# ============================================================
# STEP 3 — KEEP ONLY UNAMBIGUOUS ABNORMAL IMAGES
# ============================================================

print("\nFiltering abnormal images...")

classification_records = []

ambiguous_count = 0

for image_id, abnormalities in image_abnormalities.items():

    # If an image has more than one target abnormality,
    # we cannot safely give it one single class.
    if len(abnormalities) != 1:
        ambiguous_count += 1
        continue

    class_name = abnormalities[0]

    classification_records.append(
        {
            "image_id": image_id,
            "class_name": class_name,
            "class_id": CLASS_MAPPING[class_name],
        }
    )


print(
    f"Unambiguous abnormal images : "
    f"{len(classification_records)}"
)

print(
    f"Ambiguous abnormal images removed : "
    f"{ambiguous_count}"
)


# ============================================================
# STEP 4 — ADD NORMAL IMAGES
# ============================================================

print("\nAdding normal images...")

for image_id in normal_images:

    classification_records.append(
        {
            "image_id": image_id,
            "class_name": "Normal",
            "class_id": CLASS_MAPPING["Normal"],
        }
    )


# ============================================================
# CREATE CLASSIFICATION DATAFRAME
# ============================================================

classification_df = pd.DataFrame(
    classification_records
)


# ============================================================
# REMOVE DUPLICATE IMAGE IDS
# ============================================================

classification_df = (
    classification_df
    .drop_duplicates(subset=["image_id"])
    .reset_index(drop=True)
)


# ============================================================
# DISPLAY CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION DATASET")
print("=" * 60)

print(
    classification_df[
        ["class_id", "class_name"]
    ]
    .groupby(["class_id", "class_name"])
    .size()
    .sort_index()
)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\nCreating train / validation / test splits...")

# First:
# 80% train
# 20% temporary

train_df, temp_df = train_test_split(
    classification_df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=classification_df["class_id"],
)

# Split temporary 50/50:
# 10% validation
# 10% test

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["class_id"],
)


# ============================================================
# SAVE CSV FILES
# ============================================================

train_path = OUTPUT_DIR / "train.csv"
val_path = OUTPUT_DIR / "val.csv"
test_path = OUTPUT_DIR / "test.csv"

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION DATASET PREPARATION COMPLETED")
print("=" * 60)

print(f"Total images : {len(classification_df)}")
print(f"Training     : {len(train_df)}")
print(f"Validation   : {len(val_df)}")
print(f"Test         : {len(test_df)}")

print("\nTraining distribution:")
print(train_df["class_name"].value_counts().sort_index())

print("\nValidation distribution:")
print(val_df["class_name"].value_counts().sort_index())

print("\nTest distribution:")
print(test_df["class_name"].value_counts().sort_index())

print("\nOutput files:")
print(f"Train : {train_path}")
print(f"Val   : {val_path}")
print(f"Test  : {test_path}")