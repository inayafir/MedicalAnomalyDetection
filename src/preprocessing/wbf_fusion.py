from pathlib import Path

import pandas as pd
from ensemble_boxes import weighted_boxes_fusion


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = PROJECT_ROOT / "data" / "raw" / "vinbigdata" / "train.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_CSV = OUTPUT_DIR / "annotations_wbf.csv"


# ============================================================
# CONFIGURATION
# ============================================================

# VinBigData class 14 = "No finding"
# We only keep the 14 abnormality classes.
VALID_CLASSES = list(range(14))

# IoU threshold used by Weighted Boxes Fusion.
IOU_THR = 0.55

# Minimum confidence required for a fused box.
SKIP_BOX_THR = 0.0


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading train.csv...")

df = pd.read_csv(CSV_PATH)

print(f"Original dataset shape: {df.shape}")


# ============================================================
# REMOVE "NO FINDING"
# ============================================================

df = df[df["class_id"].isin(VALID_CLASSES)].copy()

print(f"Rows after removing 'No finding': {len(df)}")


# ============================================================
# WBF
# ============================================================

fused_annotations = []

processed_images = 0
total_fused_boxes = 0


for image_id, image_group in df.groupby("image_id"):

    processed_images += 1

    # --------------------------------------------------------
    # Process each abnormality class separately
    # --------------------------------------------------------

    for class_id, class_group in image_group.groupby("class_id"):

        boxes = []
        scores = []
        labels = []

        # ----------------------------------------------------
        # Each annotation row represents one radiologist box
        # ----------------------------------------------------

        for _, row in class_group.iterrows():

            width = float(row["width"])
            height = float(row["height"])

            x_min = float(row["x_min"])
            y_min = float(row["y_min"])
            x_max = float(row["x_max"])
            y_max = float(row["y_max"])

            # Skip invalid annotations
            if width <= 0 or height <= 0:
                continue

            if x_max <= x_min or y_max <= y_min:
                continue

            # ------------------------------------------------
            # Normalize coordinates to 0-1
            # ------------------------------------------------

            box = [
                max(0.0, min(1.0, x_min / width)),
                max(0.0, min(1.0, y_min / height)),
                max(0.0, min(1.0, x_max / width)),
                max(0.0, min(1.0, y_max / height)),
            ]

            boxes.append(box)

            # ------------------------------------------------
            # VinBigData does not provide confidence scores.
            #
            # Therefore every radiologist annotation gets
            # equal weight.
            # ------------------------------------------------

            scores.append(1.0)
            labels.append(int(class_id))

        if not boxes:
            continue

        # ----------------------------------------------------
        # WBF expects a list of models/sources.
        #
        # We treat each radiologist as a source.
        # ----------------------------------------------------

        radiologist_boxes = []
        radiologist_scores = []
        radiologist_labels = []

        for rad_id, rad_group in class_group.groupby("rad_id"):

            rad_boxes = []
            rad_scores = []
            rad_labels = []

            for _, row in rad_group.iterrows():

                width = float(row["width"])
                height = float(row["height"])

                x_min = float(row["x_min"])
                y_min = float(row["y_min"])
                x_max = float(row["x_max"])
                y_max = float(row["y_max"])

                if width <= 0 or height <= 0:
                    continue

                if x_max <= x_min or y_max <= y_min:
                    continue

                box = [
                    max(0.0, min(1.0, x_min / width)),
                    max(0.0, min(1.0, y_min / height)),
                    max(0.0, min(1.0, x_max / width)),
                    max(0.0, min(1.0, y_max / height)),
                ]

                rad_boxes.append(box)
                rad_scores.append(1.0)
                rad_labels.append(int(class_id))

            if rad_boxes:
                radiologist_boxes.append(rad_boxes)
                radiologist_scores.append(rad_scores)
                radiologist_labels.append(rad_labels)

        if not radiologist_boxes:
            continue

        # ----------------------------------------------------
        # WEIGHTED BOXES FUSION
        # ----------------------------------------------------

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            radiologist_boxes,
            radiologist_scores,
            radiologist_labels,
            weights=None,
            iou_thr=IOU_THR,
            skip_box_thr=SKIP_BOX_THR,
        )

        # ----------------------------------------------------
        # Convert fused normalized coordinates back to pixels
        # ----------------------------------------------------

        width = float(class_group.iloc[0]["width"])
        height = float(class_group.iloc[0]["height"])

        for box, score, label in zip(
            fused_boxes,
            fused_scores,
            fused_labels
        ):

            x_min = box[0] * width
            y_min = box[1] * height
            x_max = box[2] * width
            y_max = box[3] * height

            fused_annotations.append(
                {
                    "image_id": image_id,
                    "class_id": int(label),
                    "class_name": class_group.iloc[0]["class_name"],
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "width": width,
                    "height": height,
                    "wbf_score": float(score),
                }
            )

            total_fused_boxes += 1


# ============================================================
# SAVE RESULT
# ============================================================

fused_df = pd.DataFrame(fused_annotations)

fused_df.to_csv(OUTPUT_CSV, index=False)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("WBF FUSION COMPLETED")
print("=" * 60)

print(f"Images processed     : {processed_images}")
print(f"Original boxes       : {len(df)}")
print(f"Fused boxes          : {total_fused_boxes}")
print(f"Output file          : {OUTPUT_CSV}")

print("\nFused boxes by class:")

print(
    fused_df
    .groupby(["class_id", "class_name"])
    .size()
    .sort_index()
)