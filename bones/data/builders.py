from __future__ import annotations

from typing import Callable

import torch
from torch.utils.data import ConcatDataset


class RepeatDataset(torch.utils.data.Dataset):
    def __init__(self, base, n):
        self.base = base
        self.n = n

    def __len__(self):
        return len(self.base) * self.n

    def __getitem__(self, idx):
        return self.base[idx % len(self.base)]

from bones.config import DATASET_DIR, WEEKS, TREATMENTS
from bones.data.dataset import FilteredBonesDataset


def stem_prefix(week: str, treatment: str) -> str:
    return f"W{week.split('_')[1].zfill(2)}_{treatment}"


def collate_fn(batch):
    return tuple(zip(*batch))


def build_group_datasets(stems: set[str], transforms: Callable | None = None) -> list[FilteredBonesDataset]:
    ds_list = []
    for week in WEEKS:
        for treatment in TREATMENTS:
            sp = stem_prefix(week, treatment)
            coco_path = DATASET_DIR / week / treatment / f"{sp}.json"
            if not coco_path.exists():
                continue
            img_root = DATASET_DIR / week / treatment
            group_stems = {s for s in stems if s.startswith(sp)}
            if group_stems:
                ds_list.append(FilteredBonesDataset(coco_path, img_root, group_stems, transforms))
    return ds_list


def build_concat_dataset(stems: set[str], transforms: Callable | None = None) -> ConcatDataset | None:
    ds_list = build_group_datasets(stems, transforms)
    return ConcatDataset(ds_list) if ds_list else None
