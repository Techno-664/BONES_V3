from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from bones.data.coco import load_coco, polygon_to_mask

IS_CROWD = "iscrowd"


class FilteredBonesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        coco_json: Path | str,
        image_root: Path | str,
        allowed_stems: set[str],
        transforms: Callable | None = None,
    ):
        self._ds = BonesDataset(coco_json, image_root, transforms)
        self._indices = [
            i for i in range(len(self._ds))
            if self._ds.image_stem(i) in allowed_stems
        ]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> Any:
        return self._ds[self._indices[index]]


class BonesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        coco_json: Path | str,
        image_root: Path | str,
        transforms: Callable | None = None,
    ):
        data: dict[str, Any] = load_coco(coco_json)
        self.image_root = Path(image_root)
        self.transforms = transforms

        coco_images: list[dict[str, Any]] = data.get("images", [])
        coco_categories: list[dict[str, Any]] = data.get("categories", [])
        coco_annotations: list[dict[str, Any]] = data.get("annotations", [])

        self._images: dict[int, dict[str, Any]] = {}
        for img in coco_images:
            self._images[img["id"]] = img

        self._categories = {cat["id"]: cat["name"] for cat in coco_categories}
        self._anns_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in coco_annotations:
            self._anns_by_image.setdefault(ann["image_id"], []).append(ann)

        self.image_ids = sorted(self._images)

    def image_stem(self, index: int) -> str:
        image_id = self.image_ids[index]
        return self._images[image_id]["file_name"].rsplit(".", 1)[0]

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Any:
        image_id = self.image_ids[index]
        img_info = self._images[image_id]
        image_path = self.image_root / img_info["file_name"]

        image = Image.open(image_path).convert("RGB")

        anns: list[dict[str, Any]] = self._anns_by_image.get(image_id, [])

        boxes: list[list[float]] = []
        labels: list[int] = []
        masks: list[np.ndarray] = []
        area: list[float] = []
        is_crowd: list[int] = []

        for ann in anns:
            iscrowd_val: int = ann.get(IS_CROWD, 0)
            if iscrowd_val:
                continue

            cat_id: int = ann.get("category_id", 0)
            cat_name: str | None = self._categories.get(cat_id)
            if cat_name is None:
                continue

            bbox: list[float] | None = ann.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue
            x, y, w, h = bbox
            boxes.append([x, y, x + w, y + h])

            labels.append(cat_id)

            seg: list[list[float]] = ann.get("segmentation", [])
            mask = None
            for polygon in seg:
                m = polygon_to_mask(polygon, img_info["height"], img_info["width"])
                if mask is None:
                    mask = m
                else:
                    mask = mask | m
            if mask is not None:
                masks.append(mask)

            area.append(ann.get("area", 0))
            is_crowd.append(ann.get(IS_CROWD, 0))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "masks": (
                torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)
                if masks
                else torch.zeros((0, img_info["height"], img_info["width"]), dtype=torch.uint8)
            ),
            "area": torch.as_tensor(area, dtype=torch.float32) if area else torch.zeros((0,), dtype=torch.float32),
            IS_CROWD: torch.as_tensor(is_crowd, dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            image = to_tensor(image)

        return image, target
