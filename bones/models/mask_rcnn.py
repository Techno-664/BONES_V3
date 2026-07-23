from __future__ import annotations

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from bones.config import MODEL


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
    return model


def build_mask_rcnn(num_classes: int | None = None, class_weights: list[float] | None = None) -> torch.nn.Module:
    cfg = MODEL
    if num_classes is None:
        num_classes = cfg["num_classes"]

    weights = "DEFAULT" if cfg["pretrained"] else None
    model = maskrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    if class_weights is not None:
        model.class_weights = torch.tensor(class_weights, dtype=torch.float32)
    else:
        model.class_weights = None

    return model
