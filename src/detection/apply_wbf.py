from pathlib import Path

import pandas as pd
from ensemble_boxes import weighted_boxes_fusion


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

OUTPUT_CSV = (
    OUTPUT_DIR
    / "wbf_annotations.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

IOU_THRESHOLD = 0.5

SKIP_BOX_THRESHOLD = 0.0

# These are the five project classes.
CLASS_NAMES = [
    "Normal",
    "Cardiomegaly",
    "Pleural effusion",
    "Lung Opacity",
    "Pulmonary fibrosis",
]


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("WEIGHTED BOX FUSION")
print("=" * 60)

print(
    f"Input CSV : {INPUT_CSV}"
)

print(
    f"Output CSV: {OUTPUT_CSV}"
)

print(
    f"IoU threshold: {IOU_THRESHOLD}"
)


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT_CSV.exists():

    raise FileNotFoundError(
        f"Input annotation file not found:\n"
        f"{INPUT_CSV}"
    )


print(
    "\nLoading clean annotations..."
)

df = pd.read_csv(
    INPUT_CSV
)

print(
    f"Input rows : {len(df)}"
)

print(
    f"Unique images : "
    f"{df['image_id'].nunique()}"
)


# ============================================================
# VERIFY COLUMNS
# ============================================================

required_columns = [
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


# ============================================================
# SEPARATE NORMAL AND ABNORMAL
# ============================================================

normal_df = df[
    df["project_class"]
    == "Normal"
].copy()


abnormal_df = df[
    df["project_class"]
    != "Normal"
].copy()


print(
    "\nNormal rows : "
    f"{len(normal_df)}"
)

print(
    "Abnormal rows : "
    f"{len(abnormal_df)}"
)


# ============================================================
# NORMAL IMAGES
# ============================================================

# Normal images have no bounding boxes.
#
# They are retained as image-level annotations.
#
# WBF is only applied to abnormal bounding boxes.

normal_output = (
    normal_df[
        [
            "image_id",
            "project_class",
            "project_class_id",
            "width",
            "height",
        ]
    ]
    .drop_duplicates()
    .copy()
)

normal_output[
    "x_min"
] = None

normal_output[
    "y_min"
] = None

normal_output[
    "x_max"
] = None

normal_output[
    "y_max"
] = None

normal_output[
    "num_boxes_fused"
] = 0


# ============================================================
# WBF FUNCTION
# ============================================================

def fuse_group(
    group
):

    image_id = group[
        "image_id"
    ].iloc[0]

    class_name = group[
        "project_class"
    ].iloc[0]

    class_id = int(
        group[
            "project_class_id"
        ].iloc[0]
    )

    width = float(
        group[
            "width"
        ].iloc[0]
    )

    height = float(
        group[
            "height"
        ].iloc[0]
    )


    boxes = []

    scores = []

    labels = []


    # --------------------------------------------------------
    # Convert pixel coordinates to normalized coordinates.
    # WBF expects values between 0 and 1.
    # --------------------------------------------------------

    for _, row in group.iterrows():

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


        normalized_box = [

            x_min / width,

            y_min / height,

            x_max / width,

            y_max / height,

        ]


        boxes.append(
            normalized_box
        )

        # All radiologist annotations are treated
        # equally because they are ground-truth
        # annotations rather than model predictions.

        scores.append(1.0)

        labels.append(
            class_id
        )


    # --------------------------------------------------------
    # Apply WBF
    # --------------------------------------------------------

    fused_boxes, fused_scores, fused_labels = (
        weighted_boxes_fusion(

            [boxes],

            [scores],

            [labels],

            weights=[1.0],

            iou_thr=IOU_THRESHOLD,

            skip_box_thr=SKIP_BOX_THRESHOLD,

        )
    )


    results = []


    for box, score, label in zip(
        fused_boxes,
        fused_scores,
        fused_labels,
    ):

        # Convert normalized coordinates
        # back to pixel coordinates.

        fused_x_min = (
            float(box[0])
            * width
        )

        fused_y_min = (
            float(box[1])
            * height
        )

        fused_x_max = (
            float(box[2])
            * width
        )

        fused_y_max = (
            float(box[3])
            * height
        )


        results.append(
            {
                "image_id": image_id,

                "project_class":
                    class_name,

                "project_class_id":
                    int(label),

                "x_min":
                    fused_x_min,

                "y_min":
                    fused_y_min,

                "x_max":
                    fused_x_max,

                "y_max":
                    fused_y_max,

                "width":
                    width,

                "height":
                    height,

                "num_boxes_fused":
                    len(group),

                "wbf_score":
                    float(score),

            }
        )


    return results


# ============================================================
# APPLY WBF
# ============================================================

print(
    "\nApplying WBF..."
)

fused_results = []

grouped = abnormal_df.groupby(
    [
        "image_id",
        "project_class",
    ],
    sort=False,
)


total_groups = len(
    grouped
)

print(
    f"Image/class groups : "
    f"{total_groups}"
)


for group_index, (
    group_key,
    group,
) in enumerate(
    grouped,
    start=1
):

    results = fuse_group(
        group
    )

    fused_results.extend(
        results
    )


    if (
        group_index % 1000
        == 0
    ):

        print(
            f"  Processed "
            f"{group_index}/"
            f"{total_groups}"
        )


# ============================================================
# CREATE ABNORMAL OUTPUT
# ============================================================

abnormal_output = pd.DataFrame(
    fused_results
)


# ============================================================
# COMBINE NORMAL + ABNORMAL
# ============================================================

normal_output[
    "wbf_score"
] = None


final_columns = [
    "image_id",
    "project_class",
    "project_class_id",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "width",
    "height",
    "num_boxes_fused",
    "wbf_score",
]


normal_output = normal_output[
    final_columns
]


abnormal_output = abnormal_output[
    final_columns
]


final_df = pd.concat(
    [
        normal_output,
        abnormal_output,
    ],
    ignore_index=True
)


# ============================================================
# SORT
# ============================================================

final_df = final_df.sort_values(
    by=[
        "image_id",
        "project_class_id",
    ]
).reset_index(
    drop=True
)


# ============================================================
# VALIDATION
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "WBF VALIDATION"
)

print(
    "=" * 60
)


print(
    f"Input annotation rows  : "
    f"{len(df)}"
)

print(
    f"Output annotation rows : "
    f"{len(final_df)}"
)

print(
    f"Input unique images    : "
    f"{df['image_id'].nunique()}"
)

print(
    f"Output unique images   : "
    f"{final_df['image_id'].nunique()}"
)


# Check that fused boxes are valid.

bbox_df = final_df[
    final_df["project_class"]
    != "Normal"
].copy()


invalid_fused_boxes = (
    (bbox_df["x_min"] < 0)
    | (bbox_df["y_min"] < 0)
    | (
        bbox_df["x_max"]
        <= bbox_df["x_min"]
    )
    | (
        bbox_df["y_max"]
        <= bbox_df["y_min"]
    )
    | (
        bbox_df["x_max"]
        > bbox_df["width"]
    )
    | (
        bbox_df["y_max"]
        > bbox_df["height"]
    )
)


print(
    f"Invalid fused boxes : "
    f"{invalid_fused_boxes.sum()}"
)


if invalid_fused_boxes.sum() > 0:

    raise ValueError(
        "Invalid WBF boxes detected."
    )


# ============================================================
# FUSION STATISTICS
# ============================================================

print(
    "\nFusion statistics:"
)


if len(abnormal_output) > 0:

    print(
        f"Fused abnormal boxes : "
        f"{len(abnormal_output)}"
    )

    print(
        f"Average source boxes per "
        f"fused box : "
        f"{abnormal_output['num_boxes_fused'].mean():.2f}"
    )

    print(
        f"Maximum source boxes fused : "
        f"{abnormal_output['num_boxes_fused'].max()}"
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print(
    "\nFinal WBF class distribution:"
)


distribution = (
    final_df
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
# SAVE
# ============================================================

final_df.to_csv(
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
    "WEIGHTED BOX FUSION COMPLETED"
)

print(
    "=" * 60
)

print(
    f"Output rows : "
    f"{len(final_df)}"
)

print(
    f"Output images : "
    f"{final_df['image_id'].nunique()}"
)

print(
    f"Output file : "
    f"{OUTPUT_CSV}"
)

print(
    "\nNext stage:"
)

print(
    "Create YOLO dataset + train/validation split"
)

print(
    "=" * 60
)