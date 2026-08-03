from __future__ import annotations

import os

import torch
from torchvision.models.detection import (
    faster_rcnn,
    mask_rcnn,
    maskrcnn_resnet50_fpn,  # noqa: SC100
)

from bones.config import MODEL
from bones.logging import setup_logger

log = setup_logger("mask_rcnn")


def compile_enabled() -> tuple[bool, str]:
    env = os.environ.get("BONES_COMPILE")
    if env is not None:
        value = env.strip().lower() not in ("0", "false", "no", "off")
        return value, f"BONES_COMPILE={env}"
    value = bool(MODEL.get("compile", True))
    return value, f"config MODEL['compile']={value}"


def maybe_compile(model: torch.nn.Module) -> torch.nn.Module:
    enabled, source = compile_enabled()
    if not enabled:
        log.info("torch.compile skipped (%s = False)", source)
        return model
    if not torch.cuda.is_available():
        log.info("torch.compile skipped: CUDA not available")
        return model
    try:
        compiled = torch.compile(model)
        log.info("Model compiled with torch.compile")
        return compiled
    except Exception as e:  # noqa: BLE001
        log.warning("torch.compile failed (%s); using uncompiled model", e)
        return model


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    nms_threshold: float | None = None,
) -> torch.nn.Module:
    model = build_mask_rcnn().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if nms_threshold is not None:
        model.roi_heads.nms_thresh = nms_threshold
    model.eval()
    return maybe_compile(model)


def build_mask_rcnn(num_classes: int | None = None, class_weights: list[float] | None = None) -> torch.nn.Module:
    cfg = MODEL
    if num_classes is None:
        num_classes = cfg["num_classes"]

    weights = "DEFAULT" if cfg["pretrained"] else None
    model = maskrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels  # type: ignore[union-attr]
    hidden_layer = 256
    model.roi_heads.mask_predictor = mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    if class_weights is not None:
        model.class_weights = torch.tensor(class_weights, dtype=torch.float32)
    else:
        model.class_weights = None

    return model
