from __future__ import annotations

import json
import os
from contextlib import nullcontext

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    StepLR,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from bones.cli import resolve_device
from bones.config import (
    CATEGORIES,
    CHECKPOINTS_DIR,
    FOLDS_DIR,
    N_FOLDS,
    SCORE_THRESHOLD,
    TRAIN,
)
from bones.data.builders import RepeatDataset, build_concat_dataset, collate_fn
from bones.logging import setup_logger
from bones.models.mask_rcnn import build_mask_rcnn, maybe_compile
from bones.transforms.augmentation import (
    AlbumentationsAdapter,
    build_augmentation_pipeline,
    build_val_pipeline,
)

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


def _batch_accuracy(model, images, targets) -> float:
    class_ids = sorted(CATEGORIES.keys())
    correct = 0
    total = 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            preds = model(images)
    finally:
        if was_training:
            model.train()
    for pred, target in zip(preds, targets):
        for cid in class_ids:
            gt_present = bool((target["labels"] == cid).any())
            cid_scores = pred["scores"][pred["labels"] == cid]
            pred_present = bool(len(cid_scores) > 0 and cid_scores.max().item() > SCORE_THRESHOLD)
            correct += int(gt_present == pred_present)
            total += 1
    return correct / total if total > 0 else 0.0


def train_one_epoch(model, loader, optimizer, device, grad_clip=None, scaler=None, class_weights=None):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
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

        losses: torch.Tensor = sum(loss for loss in loss_dict.values())

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

        acc = _batch_accuracy(model, images, targets)
        total_loss += losses.item()
        total_acc += acc
        pbar.set_postfix(acc=f"{acc:.4f}", loss=f"{losses.item():.4f}")

    return total_loss / len(loader), total_acc / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    total_loss = 0.0
    n = 0
    class_ids = sorted(CATEGORIES.keys())
    img_counts = {cid: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for cid in class_ids}

    model.eval()
    for images, targets in tqdm(loader, desc="  Val", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        model.train()
        with torch.no_grad():
            loss_dict = model(images, targets)
        losses: torch.Tensor = sum(loss for loss in loss_dict.values())
        model.eval()

        total_loss += losses.item()
        n += 1

        output = model(images)[0]
        target = targets[0]

        for cat_id in class_ids:
            gt_present = bool((target["labels"] == cat_id).any())
            cid_scores = output["scores"][output["labels"] == cat_id]
            pred_present = bool(len(cid_scores) > 0 and cid_scores.max().item() > SCORE_THRESHOLD)
            c = img_counts[cat_id]
            if gt_present and pred_present:
                c["tp"] += 1
            elif gt_present:
                c["fn"] += 1
            elif pred_present:
                c["fp"] += 1
            else:
                c["tn"] += 1

    avg_loss = total_loss / n if n > 0 else 0.0

    accs = []
    for cat_id in class_ids:
        c = img_counts[cat_id]
        total = c["tp"] + c["tn"] + c["fp"] + c["fn"]
        accs.append((c["tp"] + c["tn"]) / total if total > 0 else 0.0)
    val_acc = float(np.mean(accs)) if accs else 0.0

    return {"val_loss": avg_loss, "val_accuracy": val_acc}


def train(
    num_epochs: int | None = None,
    device_choice: str = "auto",
    fold: int | None = None,
    lr: float | None = None,
    augmented_copies_per_image: int = 1,
) -> torch.nn.Module:
    device = resolve_device(device_choice)
    log.info("Using device: %s", device)

    cfg = TRAIN
    if num_epochs is None:
        num_epochs = cfg["epochs"]
    if lr is not None:
        cfg = {**cfg, "lr": lr}

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
    if augmented_copies_per_image > 1:
        train_ds = RepeatDataset(train_ds, augmented_copies_per_image)
        log.info("Augmented copies per image: %d → %d samples/epoch", orig_len, len(train_ds))

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
    model = maybe_compile(model)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=cfg["lr"], momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"]
    )

    warmup = cfg["warmup_epochs"]
    warmup_sched = None
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

    writer = SummaryWriter(log_dir=str(ckpt_dir / "tensorboard"))

    start_epoch = 0
    best_val_accuracy = 0.0
    patience_counter = 0

    assert num_epochs is not None
    for epoch in range(start_epoch, num_epochs):
        log.info("Epoch %d/%d", epoch + 1, num_epochs)

        class_weights = model.class_weights.to(device) if model.class_weights is not None else None
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, cfg["grad_clip"], scaler, class_weights
        )
        val_metrics = validate(model, val_loader, device)
        val_loss = val_metrics["val_loss"]
        val_acc = val_metrics["val_accuracy"]
        scheduler.step()

        log.info("  accuracy: %.4f - loss: %.4f - val_accuracy: %.4f - val_loss: %.4f",
                 train_acc, train_loss, val_acc, val_loss)

        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)

        if val_acc > best_val_accuracy:
            log.info("  val_accuracy improved from %.4f to %.4f, saving model to %s",
                     best_val_accuracy, val_acc, ckpt_dir / "best.pth")
            best_val_accuracy = val_acc
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "best_val_accuracy": best_val_accuracy,
            }
            torch.save(checkpoint, ckpt_dir / "best.pth")
            patience_counter = 0
        else:
            log.info("  val_accuracy did not improve from %.4f", best_val_accuracy)
            patience_counter += 1
            if patience_counter >= cfg["early_stop_patience"]:
                log.info("  Early stopping at epoch %d", epoch + 1)
                break

        log.info("")

    log.info("Training complete. Best val accuracy: %.4f", best_val_accuracy)

    writer.add_hparams(
        {
            "lr": cfg["lr"],
            "epochs": num_epochs,
            "batch_size": cfg["batch_size"],
            "augmented_copies": cfg["augmented_copies_per_image"],
            "fold": fold if fold is not None else -1,
        },
        {"best_val_accuracy": best_val_accuracy},
    )
    writer.close()

    best_path = ckpt_dir / "best.pth"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def main() -> int:
    from bones.cli import prompt_bool, prompt_choice, prompt_float, prompt_int

    if not prompt_bool(
        "Enable torch.compile? (no on Colab: avoids recompilation overhead)",
        default=True,
    ):
        os.environ["BONES_COMPILE"] = "0"

    epochs = prompt_int("Number of epochs", default=TRAIN["epochs"], min_val=1)

    fold = prompt_int("Fold (0-4, or leave blank for all 5 folds)", default=None, min_val=0, max_val=N_FOLDS - 1)

    lr = prompt_float("Learning rate", default=TRAIN["lr"], min_val=0.00001, max_val=1.0)

    augmented_copies_raw = prompt_int(
        "Augmented copies per image (3 = each image appears 3x per epoch"
        " with different augmentations)",
        default=TRAIN["augmented_copies_per_image"], min_val=1,
    )
    augmented_copies: int = augmented_copies_raw if augmented_copies_raw is not None else 1

    device = prompt_choice(
        "Select device:",
        {"auto": "Auto-detect", "cuda": "CUDA", "cpu": "CPU"},
        default="auto",
    )

    if fold is not None:
        train(epochs, device_choice=device, fold=fold, lr=lr, augmented_copies_per_image=augmented_copies)
        log.info("Fold %d complete. Best checkpoint: %s", fold, CHECKPOINTS_DIR / f"fold_{fold}" / "best.pth")
    else:
        for k in range(N_FOLDS):
            log.info("=== Fold %d / %d ===", k + 1, N_FOLDS)
            train(epochs, device_choice=device, fold=k, lr=lr, augmented_copies_per_image=augmented_copies)
            log.info("  Fold %d checkpoint: %s", k, CHECKPOINTS_DIR / f"fold_{k}" / "best.pth")
        log.info("All %d folds complete.", N_FOLDS)
    return 0
