from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "DATASET"
FOLDS_DIR = PROJECT_ROOT / "FOLDS"
CHECKPOINTS_DIR = PROJECT_ROOT / "CHECKPOINTS"

WEEKS = ["WEEK_6", "WEEK_12"]
TREATMENTS = ["Control", "Gel", "GelLaser", "Laser"]
CATEGORIES = {1: "normal_bone", 2: "fracture_gap", 3: "callus"}
NUM_CLASSES = len(CATEGORIES) + 1

RANDOM_STATE = 42
N_FOLDS = 5
SCORE_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
IOU_MATCH_THRESHOLD = 0.5

TRAIN = dict(
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005,
    grad_clip=1.0,
    epochs=100,
    early_stop_patience=10,
    batch_size=2,
    num_workers=0,
    warmup_epochs=3,
    warmup_start_factor=0.001,
    scheduler="cosine",
    class_weights=[1.025, 1.025, 1.0, 1.577],
    augmentation_count=None,
)

MODEL = dict(
    num_classes=NUM_CLASSES,
    pretrained=True,
    nms_thresh=0.5,
)

ANALYTICS = dict(
    mAP_iou_start=0.5,
    mAP_iou_end=0.95,
    mAP_iou_steps=10,
)

CONF_MAT_IOU_THRESHOLD = 0.5

IOU_THRESHOLDS = [round(x, 2) for x in np.linspace(ANALYTICS["mAP_iou_start"], ANALYTICS["mAP_iou_end"], ANALYTICS["mAP_iou_steps"])]

VIS = dict(
    save_overlays=True,
    n_failure_cases=9,
    f1_threshold_step=0.05,
)

AUGMENTATION = dict(
    resize_short=800,
    resize_max=1333,

    geometric_prob=0.8,
    flip_h_prob=0.5,
    flip_v_prob=0.3,
    intensity_prob=0.7,
    noise_prob=0.4,
    rotation_limits=(-45, 45),
    shift_limit=0.05,
    scale_limit=0.1,
    shear_limit=10,
    brightness_limit=0.2,
    contrast_limit=0.1,
    tone_curve_low=0.8,
    clahe_clip=2.0,
    clahe_tile=(8, 8),
    sharpen_alpha=(0.2, 0.5),
    blur_limit=(3, 7),
    perspective_scale=(0.03, 0.07),
)
