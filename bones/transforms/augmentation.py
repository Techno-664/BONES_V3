from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F

from bones.config import AUGMENTATION


class AlbumentationsAdapter:
    def __init__(self, pipeline: A.Compose):
        self.pipeline = pipeline
        bbox_proc = pipeline.processors.get("bboxes")
        if bbox_proc is not None:
            p = bbox_proc.params
            p.check_each_transform = False
            p.min_area = 0.0
            p.min_visibility = 0.0
            p.min_width = 0.0
            p.min_height = 0.0

    def __call__(self, image, target):
        image_np = np.array(image)

        if target["masks"].shape[0] == 0:
            transformed = self.pipeline(image=image_np)
            return F.to_tensor(transformed["image"]), target

        boxes = target["boxes"].numpy()
        bboxes_coco = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]

        labels = target["labels"].tolist()

        masks_np = target["masks"].numpy()
        mask_stack = np.transpose(masks_np, (1, 2, 0))

        transformed = self.pipeline(
            image=image_np,
            mask=mask_stack,
            bboxes=bboxes_coco,
            category_ids=labels,
        )

        aug_image = transformed["image"]
        aug_bboxes = transformed["bboxes"]
        aug_labels = transformed["category_ids"]
        aug_masks = transformed.get("mask")

        if aug_masks is None or aug_masks.ndim < 3:
            h, w = aug_image.shape[:2]
            return self._empty_target(aug_image, target, h, w)

        new_boxes = []
        new_labels = []
        new_masks = []
        new_area = []

        for i in range(aug_masks.shape[-1]):
            mask_ch = aug_masks[:, :, i]
            if i >= len(aug_bboxes):
                break
            x, y, w, h = aug_bboxes[i]
            area = w * h
            if mask_ch.sum() <= 0 or area <= 0:
                continue
            new_boxes.append([x, y, x + w, y + h])
            new_labels.append(aug_labels[i])
            new_masks.append(mask_ch)
            new_area.append(area)

        if not new_boxes:
            h, w = aug_image.shape[:2]
            return self._empty_target(aug_image, target, h, w)

        target["boxes"] = torch.as_tensor(new_boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(new_labels, dtype=torch.int64)
        target["masks"] = torch.as_tensor(
            np.stack(new_masks, axis=0), dtype=torch.uint8
        )
        target["area"] = torch.as_tensor(new_area, dtype=torch.float32)
        target["iscrowd"] = torch.zeros(len(new_boxes), dtype=torch.int64)

        return F.to_tensor(aug_image), target

    @staticmethod
    def _empty_target(aug_image, target, h, w):
        target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        target["labels"] = torch.zeros((0,), dtype=torch.int64)
        target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)
        target["area"] = torch.zeros((0,), dtype=torch.float32)
        target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)
        return F.to_tensor(aug_image), target


def build_augmentation_pipeline() -> A.Compose:
    cfg = AUGMENTATION

    return A.Compose(
        [
            A.SmallestMaxSize(max_size=cfg["resize_short"], interpolation=cv2.INTER_NEAREST),
            A.LongestMaxSize(max_size=cfg["resize_max"], interpolation=cv2.INTER_NEAREST),
            A.OneOf(
                [
                    A.Affine(
                        translate_percent=(-cfg["shift_limit"], cfg["shift_limit"]),
                        scale=(1 - cfg["scale_limit"], 1 + cfg["scale_limit"]),
                        rotate=cfg["rotation_limits"],
                        border_mode=cv2.BORDER_CONSTANT,
                        p=0.5,
                    ),
                    A.Affine(
                        shear=(-cfg["shear_limit"], cfg["shear_limit"]),
                        border_mode=cv2.BORDER_CONSTANT,
                        p=0.3,
                    ),
                    A.Perspective(
                        scale=cfg["perspective_scale"],
                        border_mode=cv2.BORDER_CONSTANT,
                        p=0.2,
                    ),
                ],
                p=cfg["geometric_prob"],
            ),
            A.HorizontalFlip(p=cfg["flip_h_prob"]),
            A.VerticalFlip(p=cfg["flip_v_prob"]),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=cfg["brightness_limit"],
                        contrast_limit=cfg["contrast_limit"],
                        p=0.5,
                    ),
                    A.RandomGamma(p=0.3),
                    A.RandomToneCurve(scale=cfg["tone_curve_low"], p=0.2),
                ],
                p=cfg["intensity_prob"],
            ),
            A.CLAHE(
                clip_limit=cfg["clahe_clip"],
                tile_grid_size=cfg["clahe_tile"],
                p=0.3,
            ),
            A.Sharpen(alpha=cfg["sharpen_alpha"], p=0.3),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.05, 0.2), p=0.5),
                    A.GaussianBlur(blur_limit=cfg["blur_limit"], p=0.5),
                ],
                p=cfg["noise_prob"],
            ),
        ],
        bbox_params=A.BboxParams(
            format="coco", label_fields=["category_ids"]
        ),
    )


def build_val_pipeline() -> A.Compose:
    cfg = AUGMENTATION
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=cfg["resize_short"], interpolation=cv2.INTER_NEAREST),
            A.LongestMaxSize(max_size=cfg["resize_max"], interpolation=cv2.INTER_NEAREST),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"]),
    )
