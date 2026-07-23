from __future__ import annotations

import json
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

from bones.config import (
    FOLDS_DIR,
    RANDOM_STATE,
    N_FOLDS,
    DATASET_DIR,
    WEEKS,
    TREATMENTS,
)
from bones.logging import setup_logger
from bones.data.builders import stem_prefix

log = setup_logger("generate_splits")


def main() -> int:
    stems: list[str] = []
    groups: list[str] = []

    for week in WEEKS:
        for treatment in TREATMENTS:
            group_dir = DATASET_DIR / week / treatment
            json_path = group_dir / f"{stem_prefix(week, treatment)}.json"
            if not json_path.exists():
                continue

            with open(json_path, "r") as f:
                data = json.load(f)

            label = f"{week}_{treatment}"
            for img in data.get("images", []):
                stem = Path(img["file_name"]).stem
                stems.append(stem)
                groups.append(label)

    log.info("Total images: %d", len(stems))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(stems, groups))

    FOLDS_DIR.mkdir(parents=True, exist_ok=True)

    for k, (train_idx, val_idx) in enumerate(splits):
        train_ids = [stems[i] for i in train_idx]
        val_ids = [stems[i] for i in val_idx]

        train_path = FOLDS_DIR / f"fold_{k}_train.json"
        with open(train_path, "w") as f:
            json.dump({"split": f"fold_{k}_train", "image_ids": sorted(train_ids)}, f, indent=2)
        log.info("  %s (%d images)", train_path, len(train_ids))

        val_path = FOLDS_DIR / f"fold_{k}_val.json"
        with open(val_path, "w") as f:
            json.dump({"split": f"fold_{k}_val", "image_ids": sorted(val_ids)}, f, indent=2)
        log.info("  %s (%d images)", val_path, len(val_ids))

    return 0
