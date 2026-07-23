from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bones.config import (
    CATEGORIES,
    CHECKPOINTS_DIR,
    CONF_MAT_IOU_THRESHOLD,
    IOU_THRESHOLDS,
    MODEL,
    N_FOLDS,
    SCORE_THRESHOLD,
    MASK_THRESHOLD,
    IOU_MATCH_THRESHOLD,
    FOLDS_DIR,
    VIS,
)
from bones.metrics.analytics import (
    compute_mAP,
    compute_f1_vs_threshold,
    compute_tide_errors,
    confusion_matrix,
    sensitivity_specificity,
    multiclass_auc_roc,
)
from bones.viz.plots import save_all_figures, plot_cross_fold_metrics
from bones.models.mask_rcnn import load_checkpoint
from bones.logging import setup_logger
from bones.cli import resolve_device
from bones.data.builders import build_concat_dataset, collate_fn
from bones.metrics.matching import compute_class_metrics, derive_class_metrics

log = setup_logger("evaluate")


@torch.no_grad()
def _run_eval(
    model: torch.nn.Module,
    loader: DataLoader,
    desc: str,
    device: torch.device,
    save_dir: str | None = None,
) -> dict:
    cat_id_to_name = CATEGORIES
    class_ids = sorted(CATEGORIES.keys())

    results = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "total_gt": 0, "total_pred": 0, "iou_sum": 0.0, "iou_count": 0, "dice_list": []})
    all_gt_masks: list[np.ndarray] = []
    all_gt_labels: list[int] = []
    all_pred_masks: list[np.ndarray] = []
    all_pred_labels: list[int] = []
    all_pred_scores: list[float] = []
    all_image_scores: list[dict[int, float]] = []
    all_image_labels: list[dict[int, int]] = []

    matched_ious_per_class: dict[int, list[float]] = {cid: [] for cid in class_ids}
    all_images: list = []
    all_targets: list[dict] = []
    all_outputs: list[dict] = []
    per_image_errors: list[dict] = []
    viz_on = save_dir is not None
    if viz_on:
        n_total = len(loader) * loader.batch_size if loader.batch_size else len(loader.dataset)
        log.info("Storing %d image tensors in memory for overlay generation (~%d × H × W × 3)", n_total, n_total)

    if loader.batch_size > 1:
        log.warning("_run_eval assumes batch_size=1; only first image per batch used")

    for images, targets in tqdm(loader, desc=desc):
        images = [img.to(device) for img in images]
        outputs = model(images)

        target = targets[0]
        output = outputs[0]

        if viz_on:
            all_images.append(images[0].cpu())
            all_targets.append(target)
            all_outputs.append(output)

        image_labels: dict[int, int] = {}
        image_scores: dict[int, float] = {}
        for cid in class_ids:
            image_labels[cid] = 1 if (target["labels"] == cid).any() else 0
            cid_scores = output["scores"][output["labels"] == cid]
            image_scores[cid] = float(cid_scores.max().item()) if len(cid_scores) > 0 else 0.0
        all_image_scores.append(image_scores)
        all_image_labels.append(image_labels)

        img_fp = 0
        img_fn = 0

        for cat_id in class_ids:
            cat_name = cat_id_to_name[cat_id]

            gt_idx = (target["labels"] == cat_id).nonzero(as_tuple=True)[0]
            gt_masks = target["masks"][gt_idx]
            n_gt = len(gt_idx)

            score_ok = output["scores"] > SCORE_THRESHOLD
            pred_idx = ((output["labels"] == cat_id) & score_ok).nonzero(as_tuple=True)[0]
            pred_masks_tensor = (output["masks"][pred_idx, 0] > MASK_THRESHOLD).to(torch.uint8)
            n_pred = len(pred_idx)

            results[cat_name]["total_gt"] += n_gt
            results[cat_name]["total_pred"] += n_pred

            r = compute_class_metrics(gt_masks, pred_masks_tensor, pred_masks_tensor, iou_threshold=IOU_MATCH_THRESHOLD)
            for key in ("tp", "fp", "fn", "iou_sum", "iou_count"):
                results[cat_name][key] += r[key]
            results[cat_name]["dice_list"].extend(r["dice_list"])
            matched_ious_per_class[cat_id].extend(r.get("matched_ious", []))

            img_fp += r["fp"]
            img_fn += r["fn"]

            for j in range(n_gt):
                all_gt_masks.append(gt_masks[j].cpu().numpy())
                all_gt_labels.append(cat_id)
            for j in range(n_pred):
                all_pred_masks.append(pred_masks_tensor[j].cpu().numpy())
                all_pred_labels.append(cat_id)
                all_pred_scores.append(float(output["scores"][pred_idx[j]].item()))

        if viz_on:
            per_image_errors.append({"fp": img_fp, "fn": img_fn})

    metrics = {}
    for cat_id in class_ids:
        cat_name = cat_id_to_name[cat_id]
        counts = results[cat_name]
        r = derive_class_metrics(counts)
        f1_val = r.pop("f1")
        metrics[cat_name] = {
            **r,
            "f1_score": f1_val,
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
        }
        log.info("  %s: P=%.4f R=%.4f F1=%.4f mIoU=%.4f Dice=%.4f",
                 cat_name, r["precision"], r["recall"], f1_val, r["mean_iou"], r["mean_dice"])

    if all_gt_masks:
        map_results = compute_mAP(all_pred_masks, all_gt_masks, all_pred_scores, all_pred_labels, all_gt_labels, class_ids, IOU_THRESHOLDS, return_details=viz_on)
        metrics.update(map_results)
        log.info("  mAP@0.5: %.4f  mAP@0.5:0.95: %.4f", map_results['mAP_50'], map_results['mAP_50_95'])
    else:
        metrics.update({"mAP_50": 0.0, "mAP_50_95": 0.0})

    cm = confusion_matrix(all_pred_masks, all_gt_masks, all_pred_labels, all_gt_labels, class_ids, CONF_MAT_IOU_THRESHOLD)
    metrics["confusion_matrix"] = cm
    ss = sensitivity_specificity(cm)
    metrics["sensitivity_specificity"] = ss

    for cid in class_ids:
        key = int(cid)
        if key in ss:
            log.info("  %s: Sens=%.4f Spec=%.4f", cat_id_to_name[key], ss[key]['sensitivity'], ss[key]['specificity'])
    log.info("  Macro: Sens=%.4f Spec=%.4f", ss['macro_avg']['sensitivity'], ss['macro_avg']['specificity'])
    log.info("  Micro: Sens=%.4f Spec=%.4f", ss['micro_avg']['sensitivity'], ss['micro_avg']['specificity'])

    if all_image_scores:
        auc_roc = multiclass_auc_roc(all_image_scores, all_image_labels, class_ids)
        metrics["auc_roc"] = {str(k): v["auc"] for k, v in auc_roc.items()}
        for cid in class_ids:
            auc = auc_roc[cid]["auc"]
            log.info("  %s: AUC-ROC=%.4f", cat_id_to_name[cid], auc)

    if viz_on:
        eval_data: dict = {}
        if all_gt_masks:
            eval_data["pr_curves"] = map_results.get("pr_curves", {})
            eval_data["per_class_ap"] = map_results.get("per_class_ap", {})
        eval_data["f1_data"] = compute_f1_vs_threshold(all_pred_masks, all_gt_masks, all_pred_scores, all_pred_labels, all_gt_labels, class_ids, IOU_MATCH_THRESHOLD, VIS.get("f1_threshold_step", 0.05))
        eval_data["tide_data"] = compute_tide_errors(all_pred_masks, all_gt_masks, all_pred_scores, all_pred_labels, all_gt_labels, class_ids, CONF_MAT_IOU_THRESHOLD)
        eval_data["matched_ious_per_class"] = {int(k): v for k, v in matched_ious_per_class.items()}
        eval_data["images"] = all_images
        eval_data["targets"] = all_targets
        eval_data["outputs"] = all_outputs
        eval_data["per_image_errors"] = per_image_errors
        log.info("Generating figures in %s ...", save_dir)
        save_all_figures(metrics, eval_data, save_dir)

    return metrics


def _load_split_stems(split_name: str) -> set[str]:
    path = FOLDS_DIR / f"{split_name}.json"
    if not path.exists():
        log.warning("  Split file not found: %s", path)
        return set()
    with open(path) as f:
        data = json.load(f)
    return set(data["image_ids"])


@torch.no_grad()
def evaluate_split(fold: int, checkpoint_path: str, device_choice: str = "auto",
                   nms_threshold: float | None = None, save_dir: str | None = None) -> dict:
    device = resolve_device(device_choice)
    split_name = f"fold_{fold}_val"
    log.info("Evaluating %s on %s...", split_name, device)

    stems = _load_split_stems(split_name)
    if not stems:
        log.warning("  No images found for %s", split_name)
        return {}

    ds = build_concat_dataset(stems)
    loader = DataLoader(
        ds, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    model = load_checkpoint(checkpoint_path, device, nms_threshold)
    return _run_eval(model, loader, f"{split_name} eval", device, save_dir)


def _aggregate_metrics(all_metrics: list[dict]) -> dict:
    keys = ["precision", "recall", "f1_score", "mean_iou", "mean_dice"]
    aggregated = {"n_folds": len(all_metrics)}
    for cat_name in [v for k, v in CATEGORIES.items()]:
        vals = {k: [] for k in keys}
        for m in all_metrics:
            if cat_name in m:
                for k in keys:
                    vals[k].append(m[cat_name].get(k, 0.0))
        aggregated[cat_name] = {}
        for k in keys:
            arr = np.array(vals[k])
            aggregated[cat_name][f"{k}_mean"] = round(float(arr.mean()), 4)
            if len(arr) > 1:
                aggregated[cat_name][f"{k}_std"] = round(float(arr.std(ddof=1)), 4)
    return aggregated


def main() -> int:
    from bones.cli import prompt_int, prompt_path, prompt_choice, prompt_float, prompt_bool

    all_folds = prompt_bool("Evaluate all 5 folds?", default=False)

    nms = prompt_float("NMS IoU threshold", default=MODEL["nms_thresh"], min_val=0.0, max_val=1.0)

    device = prompt_choice(
        "Select device:",
        {"auto": "Auto-detect", "cuda": "CUDA", "cpu": "CPU"},
        default="auto",
    )

    gen_figures = prompt_bool("Generate evaluation figures?", default=True)

    if all_folds:
        all_metrics = []
        for k in range(N_FOLDS):
            ckpt = str(CHECKPOINTS_DIR / f"fold_{k}" / "best.pth")
            log.info("=== Fold %d ===", k)
            if not Path(ckpt).exists():
                log.warning("  Checkpoint not found: %s", ckpt)
                continue
            save_dir = str(CHECKPOINTS_DIR / f"fold_{k}" / "figures") if gen_figures else None
            m = evaluate_split(k, ckpt, device, nms, save_dir=save_dir)
            all_metrics.append(m)
        if all_metrics:
            agg = _aggregate_metrics(all_metrics)
            log.info("=== Aggregated across %d folds ===", len(all_metrics))
            log.info(json.dumps(agg, indent=2))
            if gen_figures:
                fig_dir = str(CHECKPOINTS_DIR / "figures")
                Path(fig_dir).mkdir(parents=True, exist_ok=True)
                plot_cross_fold_metrics(all_metrics, str(Path(fig_dir) / "cross_fold_metrics.png"),
                                        [CATEGORIES[cid] for cid in sorted(CATEGORIES.keys())])
    else:
        fold = prompt_int("Fold index (0-4)", default=0, min_val=0, max_val=N_FOLDS - 1)
        default_ckpt = str(CHECKPOINTS_DIR / f"fold_{fold}" / "best.pth")
        ckpt = prompt_path("Checkpoint path (leave blank for default)", default=default_ckpt, must_exist=True)
        if not Path(str(ckpt)).exists():
            log.error("Checkpoint not found: %s", ckpt)
            return 1
        save_dir = str(CHECKPOINTS_DIR / f"fold_{fold}" / "figures") if gen_figures else None
        metrics = evaluate_split(fold, str(ckpt), device, nms, save_dir=save_dir)
        log.info(json.dumps(metrics, indent=2))
    return 0
