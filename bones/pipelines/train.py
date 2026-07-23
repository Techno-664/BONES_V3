from __future__ import annotations

import json
from contextlib import nullcontext

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    StepLR,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from bones.transforms.augmentation import (
    AlbumentationsAdapter,
    build_augmentation_pipeline,
    build_val_pipeline,
)
from bones.config import (
    CATEGORIES,
    CHECKPOINTS_DIR,
    N_FOLDS,
    SCORE_THRESHOLD,
    MASK_THRESHOLD,
    IOU_MATCH_THRESHOLD,
    FOLDS_DIR,
    TRAIN,
)
from bones.logging import setup_logger
from bones.models.mask_rcnn import build_mask_rcnn
from bones.cli import resolve_device
from bones.data.builders import RepeatDataset, build_concat_dataset, collate_fn
from bones.metrics.matching import compute_class_metrics, derive_class_metrics

log = setup_logger("train")


def load_datasets(fold: int | None = None, train_transforms=None, val_transforms=None):
    if fold is not None:
        train_name = f"fold_{fold}_train"
        val_name = f"fold_{fold}_val"
    else:
        train_name = "train"
        val_name = "val"
    with open(FOLDS_DIR / f"{train_name}.json") as f:
        train_data = json.load(f)
    with open(FOLDS_DIR / f"{val_name}.json") as f:
        val_data = json.load(f)

    train_stems = set(train_data["image_ids"])
    val_stems = set(val_data["image_ids"])

    train_ds = build_concat_dataset(train_stems, train_transforms)
    val_ds = build_concat_dataset(val_stems, val_transforms)

    return train_ds, val_ds


def train_one_epoch(model, loader, optimizer, device, grad_clip=None, scaler=None, class_weights=None):
    model.train()
    total_loss = 0.0
    amp_ctx = autocast() if device.type == "cuda" else nullcontext()

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with amp_ctx:
            loss_dict = model(images, targets)

            if class_weights is not None:
                all_labels = torch.cat([t["labels"] for t in targets])
                fg_mask = all_labels > 0
                if fg_mask.any():
                    fg_weights = class_weights[all_labels[fg_mask]]
                    avg_weight = fg_weights.mean()
                    loss_dict["loss_classifier"] *= avg_weight
                    if "loss_mask" in loss_dict:
                        loss_dict["loss_mask"] *= avg_weight

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += losses.item()
        pbar.set_postfix(Loss=f"{losses.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    total_loss = 0.0
    n = 0
    cat_id_to_name = CATEGORIES
    class_ids = sorted(CATEGORIES.keys())
    results = {cid: {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0, "iou_count": 0, "dice_list": []} for cid in class_ids}

    model.eval()
    for images, targets in tqdm(loader, desc="  Val", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        model.train()
        with torch.no_grad():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        model.eval()

        total_loss += losses.item()
        n += 1

        output = model(images)[0]
        target = targets[0]

        for cat_id in class_ids:
            gt_idx = (target["labels"] == cat_id).nonzero(as_tuple=True)[0]
            gt_masks = target["masks"][gt_idx]

            score_ok = output["scores"] > 0.0
            pred_idx = ((output["labels"] == cat_id) & score_ok).nonzero(as_tuple=True)[0]
            pred_masks = (output["masks"][pred_idx, 0] > MASK_THRESHOLD).to(torch.uint8)

            r = compute_class_metrics(gt_masks, pred_masks, pred_masks, iou_threshold=IOU_MATCH_THRESHOLD)
            for key in ("tp", "fp", "fn", "iou_sum", "iou_count"):
                results[cat_id][key] += r[key]
            results[cat_id]["dice_list"].extend(r["dice_list"])

    avg_loss = total_loss / n if n > 0 else 0.0
    metrics_str = f"Val Loss: {avg_loss:.4f}"
    metrics_dict = {"val_loss": avg_loss}

    for cat_id in class_ids:
        r = derive_class_metrics(results[cat_id])
        name = cat_id_to_name[cat_id]
        metrics_str += f" | {name}: P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} IoU={r['mean_iou']:.3f} Dice={r['mean_dice']:.3f}"
        metrics_dict[name] = r

    log.info("  %s", metrics_str)
    return metrics_dict


def train(
    num_epochs: int | None = None,
    device_choice: str = "auto",
    fold: int | None = None,
    lr: float | None = None,
    augmentation_count: int | None = None,
) -> torch.nn.Module:
    device = resolve_device(device_choice)
    log.info("Using device: %s", device)

    cfg = TRAIN
    if num_epochs is None:
        num_epochs = cfg["epochs"]
    if lr is not None:
        cfg = {**cfg, "lr": lr}
    if augmentation_count is None:
        augmentation_count = cfg.get("augmentation_count")

    pipeline = build_augmentation_pipeline()
    train_adapter = AlbumentationsAdapter(pipeline)
    val_pipeline = build_val_pipeline()
    val_adapter = AlbumentationsAdapter(val_pipeline)
    train_ds, val_ds = load_datasets(
        fold=fold, train_transforms=train_adapter, val_transforms=val_adapter
    )

    if train_ds is None or len(train_ds) == 0:
        raise ValueError("No training samples found")
    if val_ds is None or len(val_ds) == 0:
        raise ValueError("No validation samples found")

    orig_len = len(train_ds)
    if augmentation_count is not None and augmentation_count > 1:
        train_ds = RepeatDataset(train_ds, augmentation_count)
        log.info("Augmentation multiplier: %d → %d samples/epoch", orig_len, len(train_ds))

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=cfg["num_workers"]
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        collate_fn=collate_fn, num_workers=cfg["num_workers"]
    )

    log.info("%d train / %d val", len(train_ds), len(val_ds))

    model = build_mask_rcnn(class_weights=cfg.get("class_weights")).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=cfg["lr"], momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"]
    )

    warmup = cfg["warmup_epochs"]
    if warmup > 0:
        warmup_sched = LinearLR(optimizer, start_factor=cfg["warmup_start_factor"], total_iters=warmup)
    if cfg.get("scheduler", "cosine") == "step":
        main_sched = StepLR(optimizer, step_size=5, gamma=0.5)
    else:
        t_max = max(1, num_epochs - warmup)
        main_sched = CosineAnnealingLR(optimizer, T_max=t_max)

    if warmup > 0:
        scheduler = SequentialLR(optimizer, [warmup_sched, main_sched], milestones=[warmup])
    else:
        scheduler = main_sched

    scaler = GradScaler() if device.type == "cuda" else None

    if fold is not None:
        ckpt_dir = CHECKPOINTS_DIR / f"fold_{fold}"
    else:
        ckpt_dir = CHECKPOINTS_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(start_epoch, num_epochs):
        log.info("Epoch %d/%d [%d train / %d val]", epoch + 1, num_epochs, len(train_ds), len(val_ds))

        class_wt = model.class_weights.to(device) if model.class_weights is not None else None
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, cfg["grad_clip"], scaler, class_wt
        )
        val_metrics = validate(model, val_loader, device)
        val_loss = val_metrics["val_loss"]
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint, ckpt_dir / "best.pth")
            patience_counter = 0
            log.info("  New best val loss: %.4f", val_loss)
        else:
            patience_counter += 1
            if patience_counter >= cfg["early_stop_patience"]:
                log.info("  Early stopping at epoch %d", epoch + 1)
                break

        log.info("")

    log.info("Training complete. Best val loss: %.4f", best_val_loss)

    best_path = ckpt_dir / "best.pth"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def main() -> int:
    from bones.cli import prompt_int, prompt_path, prompt_choice, prompt_float

    epochs = prompt_int("Number of epochs", default=TRAIN["epochs"], min_val=1)

    fold = prompt_int("Fold (0-4, or leave blank for all 5 folds)", default=None, min_val=0, max_val=N_FOLDS - 1)

    lr = prompt_float("Learning rate", default=TRAIN["lr"], min_val=0.00001, max_val=1.0)

    aug_count = prompt_int("Augmentation multiplier (blank = on-the-fly default)", default=None, min_val=1)

    device = prompt_choice(
        "Select device:",
        {"auto": "Auto-detect", "cuda": "CUDA", "cpu": "CPU"},
        default="auto",
    )

    if fold is not None:
        train(epochs, device_choice=device, fold=fold, lr=lr, augmentation_count=aug_count)
        log.info("Fold %d complete. Best checkpoint: %s", fold, CHECKPOINTS_DIR / f"fold_{fold}" / "best.pth")
    else:
        for k in range(N_FOLDS):
            log.info("=== Fold %d / %d ===", k + 1, N_FOLDS)
            train(epochs, device_choice=device, fold=k, lr=lr, augmentation_count=aug_count)
            log.info("  Fold %d checkpoint: %s", k, CHECKPOINTS_DIR / f"fold_{k}" / "best.pth")
        log.info("All %d folds complete.", N_FOLDS)
    return 0
