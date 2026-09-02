
import albumentations as A


# ============================================================
# TRAINING AUGMENTATION
# ============================================================

def get_train_augmentation():

    """
    Returns augmentation pipeline for chest X-ray training.

    The same geometric transformation is applied to:
        1. The X-ray image
        2. Its bounding boxes

    This is important for object detection.
    """

    return A.Compose(
        [
            # Small rotation to handle slight patient positioning differences
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.02, 0.02),
                rotate=(-5, 5),
                shear=(-2, 2),
                p=0.5,
            ),

            # X-rays can have different exposure/contrast levels
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.4,
            ),

            # Small amount of image noise
            A.GaussNoise(
                std_range=(0.01, 0.03),
                p=0.2,
            ),

            # Horizontal flip
            A.HorizontalFlip(
                p=0.5,
            ),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


# ============================================================
# VALIDATION AUGMENTATION
# ============================================================

def get_val_augmentation():

    """
    Validation data should NOT receive random augmentation.

    Only deterministic preprocessing should be applied.
    """

    return A.Compose([])


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    train_transform = get_train_augmentation()
    val_transform = get_val_augmentation()

    print("Training augmentation pipeline:")
    print(train_transform)

    print("\nValidation augmentation pipeline:")
    print(val_transform)

    print("\nAugmentation module loaded successfully.")
