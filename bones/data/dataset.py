from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

from bones.data.coco import load_coco, polygon_to_mask


class FilteredBonesDataset(torch.utils.data.Dataset):
    def __init__(self, coco_json, image_root, allowed_stems, transforms=None):
        self._ds = BonesDataset(coco_json, image_root, transforms)
        self._indices = [
            i for i in range(len(self._ds))
            if self._ds.image_stem(i) in allowed_stems
        ]

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        return self._ds[self._indices[idx]]


class BonesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        coco_json: Path | str,
        image_root: Path | str,
        transforms: Callable | None = None,
    ):
        data = load_coco(coco_json)
        self.image_root = Path(image_root)
        self.transforms = transforms

        self._images = {img["id"]: img for img in data.get("images", [])}
        self._categories = {
            cat["id"]: cat["name"] for cat in data.get("categories", [])
        }
        self._anns_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in data.get("annotations", []):
            self._anns_by_image.setdefault(ann["image_id"], []).append(ann)

        self.image_ids = sorted(self._images.keys())

    def image_stem(self, idx: int) -> str:
        image_id = self.image_ids[idx]
        return self._images[image_id]["file_name"].rsplit(".", 1)[0]

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[Any, dict[str, Any]]:
        image_id = self.image_ids[idx]
        img_info = self._images[image_id]
        image_path = self.image_root / img_info["file_name"]

        image = Image.open(image_path).convert("RGB")

        annotations = self._anns_by_image.get(image_id, [])

        boxes = []
        labels = []
        masks = []
        area = []
        iscrowd = []

        for ann in annotations:
            if ann.get("iscrowd", 0):
                continue

            cat_name = self._categories.get(ann.get("category_id", 0))
            if cat_name is None:
                continue

            bbox = ann.get("bbox")
            if bbox is not None and len(bbox) == 4:
                x, y, w, h = bbox
                boxes.append([x, y, x + w, y + h])

            labels.append(ann.get("category_id", 0))

            seg = ann.get("segmentation", [])
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
            iscrowd.append(ann.get("iscrowd", 0))

        target: dict[str, Any] = {}

        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([image_id], dtype=torch.int64)
        target["masks"] = (
            torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)
            if masks
            else torch.zeros((0, img_info["height"], img_info["width"]), dtype=torch.uint8)
        )
        target["area"] = torch.as_tensor(area, dtype=torch.float32) if area else torch.zeros((0,), dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            image = F.to_tensor(image)

        return image, target
