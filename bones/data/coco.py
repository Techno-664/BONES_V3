from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_coco(path: Path | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def polygon_to_mask(
    polygon: list[float], height: int, width: int
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(polygon) < 6:
        return mask
    pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask
