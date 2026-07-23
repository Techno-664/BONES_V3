from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F
from tqdm import tqdm

from bones.config import CATEGORIES, MASK_THRESHOLD, MODEL, SCORE_THRESHOLD
from bones.logging import setup_logger
from bones.models.mask_rcnn import load_checkpoint
from bones.cli import resolve_device

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


def predict(
    image_path: str,
    checkpoint_path: str,
    score_threshold: float = SCORE_THRESHOLD,
    output_path: str | None = None,
    device_choice: str = "auto",
    nms_threshold: float | None = None,
    extract_measurements: bool = False,
) -> dict | None:
    device = resolve_device(device_choice)
    log.info("Using device: %s", device)

    model = load_checkpoint(checkpoint_path, device, nms_threshold)
    path = Path(image_path)
    result = _predict_single(model, path, device, score_threshold, extract_measurements)

    log.info("Detected %d objects (threshold=%s)", len(result["valid_idx"]), score_threshold)

    if output_path is None:
        output_path = str(path.parent / f"pred_{path.name}")

    cv2.imwrite(output_path, result["display"])
    log.info("Result saved to: %s", output_path)

    if extract_measurements and "measurements" in result:
        meas = result["measurements"]
        meas_path = str(Path(output_path).parent / f"pred_{path.stem}_measurements.json")
        with open(meas_path, "w") as f:
            json.dump(meas, f, indent=2)
        log.info("Measurements saved to %s", meas_path)
        return meas
    return None


def batch_predict(
    input_dir: str,
    output_dir: str,
    checkpoint_path: str,
    score_threshold: float = SCORE_THRESHOLD,
    device_choice: str = "auto",
    nms_threshold: float | None = None,
    save_json: bool = False,
    extract_measurements: bool = False,
) -> dict | None:
    src = Path(input_dir)
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted(p for p in src.iterdir() if p.suffix.lower() in exts)
    if not paths:
        log.warning("No images found in %s", input_dir)
        return None

    device = resolve_device(device_choice)
    model = load_checkpoint(checkpoint_path, device, nms_threshold)

    all_preds = []
    all_meas = []

    for path in tqdm(paths, desc="Predicting"):
        result = _predict_single(model, path, device, score_threshold, extract_measurements)
        masks, boxes, labels, scores, valid_idx = (
            result["masks"], result["boxes"], result["labels"], result["scores"], result["valid_idx"]
        )

        sample_preds = [
            {
                "category_id": int(labels[i]),
                "category_name": CATEGORIES.get(int(labels[i]), "unknown"),
                "score": float(scores[i]),
                "bbox": [int(v) for v in boxes[i]],
            }
            for i in valid_idx
        ]

        out_path = dst / f"pred_{path.name}"
        cv2.imwrite(str(out_path), result["display"])

        all_preds.append({"file": path.name, "predictions": sample_preds})

        if extract_measurements and "measurements" in result:
            meas = result["measurements"]
            if meas:
                meas["file"] = path.name
                all_meas.append(meas)

    log.info("Saved %d predictions to %s", len(paths), output_dir)
    if save_json:
        json_path = dst / "predictions.json"
        with open(json_path, "w") as f:
            json.dump(all_preds, f, indent=2)
        log.info("Prediction JSON saved to %s", json_path)

    if extract_measurements and all_meas:
        meas_path = dst / "measurements.json"
        with open(meas_path, "w") as f:
            json.dump(all_meas, f, indent=2)
        log.info("Measurements saved to %s", meas_path)
        return {"measurements": all_meas}
    return None


def main() -> int:
    from bones.cli import prompt_choice, prompt_path, prompt_float, prompt_bool

    mode = prompt_choice(
        "Select prediction mode:",
        {"single": "Single image", "batch": "Batch directory"},
        default="single",
    )

    ckpt = prompt_path("Checkpoint path", must_exist=True)

    threshold = prompt_float("Score threshold", default=0.5, min_val=0.0, max_val=1.0)

    nms = prompt_float("NMS IoU threshold", default=MODEL["nms_thresh"], min_val=0.0, max_val=1.0)

    device = prompt_choice(
        "Select device:",
        {"auto": "Auto-detect", "cuda": "CUDA", "cpu": "CPU"},
        default="auto",
    )

    extract_measurements = prompt_bool("Extract clinical measurements?", default=False)

    if mode == "single":
        image = prompt_path("Image path", must_exist=True)
        output = prompt_path("Output path (leave blank for auto-named)")
        predict(
            str(image), str(ckpt), threshold, str(output) if output else None,
            device, nms, extract_measurements,
        )
    else:
        input_dir = prompt_path("Input directory", must_exist=True)
        output_dir = prompt_path("Output directory")
        if output_dir is None:
            output_dir = Path(str(input_dir) + "_predictions")
        save_json = prompt_bool("Save predictions.json?", default=False)
        batch_predict(
            str(input_dir), str(output_dir), str(ckpt),
            threshold, device, nms, save_json, extract_measurements,
        )

    return 0
