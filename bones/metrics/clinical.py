from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from bones.config import MASK_THRESHOLD


def fracture_gap_width(mask: np.ndarray) -> dict[str, float]:
    if mask.sum() == 0:
        return {"min_width_px": 0.0, "max_width_px": 0.0, "mean_width_px": 0.0}

    binary = mask.astype(bool)
    skeleton = skeletonize(binary)
    if skeleton.sum() == 0:
        return {"min_width_px": 0.0, "max_width_px": 0.0, "mean_width_px": 0.0}

    dist = distance_transform_edt(binary)
    half_widths = dist[skeleton]
    if len(half_widths) == 0:
        return {"min_width_px": 0.0, "max_width_px": 0.0, "mean_width_px": 0.0}

    full_widths = half_widths * 2.0
    return {
        "min_width_px": round(float(full_widths.min()), 2),
        "max_width_px": round(float(full_widths.max()), 2),
        "mean_width_px": round(float(full_widths.mean()), 2),
    }


def callus_ratio(callus_mask: np.ndarray, bone_mask: np.ndarray) -> float:
    bone_area = float(bone_mask.sum())
    if bone_area > 0:
        return round(float(callus_mask.sum()) / bone_area, 4)
    return 0.0


def compute_measurements(predictions: dict[str, Any], categories: dict[int, str]) -> dict[str, Any]:
    if "masks" not in predictions or "labels" not in predictions or "scores" not in predictions:
        return {}

    masks: np.ndarray = predictions["masks"]
    labels: np.ndarray = predictions["labels"]

    result: dict[str, Any] = {}
    fracture_mask: np.ndarray | None = None
    callus_mask_arr: np.ndarray | None = None
    bone_mask_arr: np.ndarray | None = None

    for i in range(len(labels)):
        cat_name = categories.get(int(labels[i]), "")
        mask = (masks[i, 0] > MASK_THRESHOLD).astype(np.uint8)

        if cat_name == "fracture_gap":
            fracture_mask = mask
        elif cat_name == "callus":
            callus_mask_arr = mask
        elif cat_name == "normal_bone":
            bone_mask_arr = mask

    if fracture_mask is not None:
        result["fracture_gap"] = fracture_gap_width(fracture_mask)

    if callus_mask_arr is not None and bone_mask_arr is not None:
        result["callus_ratio"] = callus_ratio(callus_mask_arr, bone_mask_arr)
    elif callus_mask_arr is not None:
        result["callus_ratio"] = 0.0

    return result
