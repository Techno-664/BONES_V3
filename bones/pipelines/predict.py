from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

from bones.cli import resolve_device
from bones.config import AUGMENTATION, CATEGORIES, MASK_THRESHOLD, MODEL, SCORE_THRESHOLD
from bones.logging import setup_logger
from bones.models.mask_rcnn import load_checkpoint

log = setup_logger("predict")

COLOR_MAP = {1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0)}


def _draw_predictions(
    display: np.ndarray,
    masks: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    score_threshold: float,
) -> list[int]:
    img_h, img_w = display.shape[:2]
    img_area = img_h * img_w
    max_area_fraction = 0.9

    valid: list[int] = []
    for i in range(len(scores)):
        if scores[i] < score_threshold:
            continue

        x1, y1, x2, y2 = map(int, boxes[i])
        box_area = (x2 - x1) * (y2 - y1)
        if box_area > max_area_fraction * img_area:
            continue

        valid.append(i)
        cat_id = int(labels[i])
        color = COLOR_MAP.get(cat_id, (255, 255, 255))
        cat_name = CATEGORIES.get(cat_id, "unknown")

        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        label = f"{cat_name}: {scores[i]:.2f}"
        cv2.putText(display, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        mask = masks[i, 0] > MASK_THRESHOLD
        overlay = np.zeros_like(display)
        overlay[mask] = color
        display = cv2.addWeighted(display, 1.0, overlay, 0.3, 0)
    return valid


def _predict_single(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    score_threshold: float = SCORE_THRESHOLD,
    extract_measurements: bool = False,
) -> dict:
    image = Image.open(image_path).convert("RGB")
    image = F.resize(image, (AUGMENTATION["resize_height"], AUGMENTATION["resize_width"]))
    image_tensor = F.to_tensor(image).to(device)

    with torch.no_grad():
        pred = model([image_tensor])[0]

    masks = pred["masks"].cpu().numpy()
    boxes = pred["boxes"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()

    display = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    valid_idx = _draw_predictions(display, masks, boxes, labels, scores, score_threshold)

    result = {
        "masks": masks, "boxes": boxes, "labels": labels,
        "scores": scores, "display": display, "valid_idx": valid_idx,
    }

    if extract_measurements:
        from bones.metrics.clinical import compute_measurements
        result["measurements"] = compute_measurements(
            {"masks": masks, "labels": labels, "scores": scores},
            CATEGORIES,
        )

    return result


def _build_output(
    result: dict,
    image_path: Path,
    checkpoint_path: str,
    score_threshold: float,
    nms_threshold: float | None,
) -> dict:
    display_h, display_w = result["display"].shape[:2]

    detections = []
    for i in result["valid_idx"]:
        cat_id = int(result["labels"][i])
        x1, y1, x2, y2 = map(int, result["boxes"][i])
        mask = result["masks"][i, 0] > MASK_THRESHOLD
        detections.append({
            "category_id": cat_id,
            "category_name": CATEGORIES.get(cat_id, "unknown"),
            "score": round(float(result["scores"][i]), 4),
            "bbox": [x1, y1, x2, y2],
            "bbox_area_px": (x2 - x1) * (y2 - y1),
            "mask_area_px": int(mask.sum()),
        })

    structure = {
        "file": image_path.name,
        "checkpoint": checkpoint_path,
        "input_size": {"width": display_w, "height": display_h},
        "score_threshold": score_threshold,
        "nms_threshold": nms_threshold if nms_threshold is not None else MODEL["nms_threshold"],
        "n_detections": len(detections),
        "detections": detections,
    }
    if "measurements" in result:
        structure["measurements"] = result["measurements"]
    return structure


def predict(
    image_path: str,
    checkpoint_path: str,
    score_threshold: float = SCORE_THRESHOLD,
    output_path: str | None = None,
    device_choice: str = "auto",
    nms_threshold: float | None = None,
    extract_measurements: bool = False,
) -> dict:
    device = resolve_device(device_choice)
    log.info("Using device: %s", device)

    model = load_checkpoint(checkpoint_path, device, nms_threshold)
    path = Path(image_path)
    result = _predict_single(model, path, device, score_threshold, extract_measurements)

    log.info("Detected %d objects (threshold=%s)", len(result["valid_idx"]), score_threshold)

    if output_path is None:
        output_path = str(path.parent / f"pred_{path.name}")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out), result["display"])
    log.info("Overlay saved to: %s", out)

    structure = _build_output(result, path, checkpoint_path, score_threshold, nms_threshold)

    json_path = out.parent / f"pred_{path.stem}_predictions.json"
    with open(json_path, "w") as f:
        json.dump(structure, f, indent=2)
    log.info("Predictions saved to: %s", json_path)

    if extract_measurements and "measurements" in result:
        meas_path = out.parent / f"pred_{path.stem}_measurements.json"
        with open(meas_path, "w") as f:
            json.dump(result["measurements"], f, indent=2)
        log.info("Measurements saved to: %s", meas_path)

    return structure


def main() -> int:
    from bones.cli import prompt_bool, prompt_choice, prompt_float, prompt_path

    ckpt = prompt_path("Checkpoint path", must_exist=True)

    threshold = prompt_float("Score threshold", default=0.5, min_val=0.0, max_val=1.0)

    nms = prompt_float("NMS IoU threshold", default=MODEL["nms_threshold"], min_val=0.0, max_val=1.0)

    device = prompt_choice(
        "Select device:",
        {"auto": "Auto-detect", "cuda": "CUDA", "cpu": "CPU"},
        default="auto",
    )

    extract_measurements = prompt_bool("Extract clinical measurements?", default=False)

    image = prompt_path("Image path", must_exist=True)
    output = prompt_path("Output path (leave blank for auto-named)")
    predict(
        str(image), str(ckpt), threshold, str(output) if output else None,
        device, nms, extract_measurements,
    )

    return 0
