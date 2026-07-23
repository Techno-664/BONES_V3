from __future__ import annotations

from typing import Any

import numpy as np
import torch


def compute_iou_matrix(
    gt_masks: torch.Tensor,
    pred_masks: torch.Tensor,
) -> np.ndarray:
    n_gt = gt_masks.size(0)
    n_pred = pred_masks.size(0)
    gt_flat = gt_masks.float().view(n_gt, -1)
    pred_flat = pred_masks.float().view(n_pred, -1)
    inter = gt_flat @ pred_flat.T
    union = gt_flat.sum(dim=1, keepdim=True) + pred_flat.sum(dim=1, keepdim=True).T - inter
    iou = inter / (union + 1e-8)
    return iou.cpu().numpy()


def greedy_iou_match(
    iou_np: np.ndarray,
    threshold: float = 0.5,
) -> tuple[set[int], set[int], list[tuple[int, int]], int, float]:
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matched_pairs: list[tuple[int, int]] = []
    tp = 0
    iou_sum = 0.0
    iou_work = iou_np.copy()

    while True:
        max_iou = float(iou_work.max())
        if max_iou < threshold:
            break
        flat_idx = iou_work.argmax()
        gi, pi = divmod(flat_idx, iou_work.shape[1])
        if gi in matched_gt or pi in matched_pred:
            iou_work[gi, pi] = 0.0
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matched_pairs.append((gi, pi))
        tp += 1
        iou_sum += max_iou
        iou_work[gi, :] = 0.0
        iou_work[:, pi] = 0.0

    return matched_gt, matched_pred, matched_pairs, tp, iou_sum


def compute_class_metrics(
    gt_masks: torch.Tensor,
    pred_masks: torch.Tensor,
    pred_masks_binary: torch.Tensor | None = None,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    n_gt = gt_masks.size(0)
    n_pred = pred_masks.size(0)

    result: dict[str, Any] = {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0, "iou_count": 0, "dice_list": []}

    if n_gt == 0 and n_pred == 0:
        return result
    if n_gt == 0:
        result["fp"] = n_pred
        return result
    if n_pred == 0:
        result["fn"] = n_gt
        return result

    iou_np = compute_iou_matrix(gt_masks, pred_masks)
    matched_gt, matched_pred, matched_pairs, tp, iou_sum = greedy_iou_match(iou_np, iou_threshold)

    if pred_masks_binary is None:
        pred_masks_binary = pred_masks

    for gi, pi in matched_pairs:
        pred_np = pred_masks_binary[pi].cpu().numpy()
        gt_np = gt_masks[gi].cpu().numpy()
        inter = float((pred_np & gt_np).sum())
        total = float(pred_np.sum()) + float(gt_np.sum())
        dice = 2.0 * inter / total if total > 0 else 0.0
        result["dice_list"].append(dice)

    result["tp"] = tp
    result["fp"] = n_pred - len(matched_pred)
    result["fn"] = n_gt - len(matched_gt)
    result["iou_sum"] = iou_sum
    result["iou_count"] = tp
    result["matched_ious"] = [iou_np[gi, pi] for gi, pi in matched_pairs]

    return result


def derive_class_metrics(counts: dict) -> dict[str, float]:
    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if (counts["tp"] + counts["fp"]) > 0 else 0.0
    recall = counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = counts["iou_sum"] / counts["iou_count"] if counts["iou_count"] > 0 else 0.0
    dice_vals = counts.get("dice_list", [])
    mean_dice = float(np.mean(dice_vals)) if dice_vals else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_iou": round(mean_iou, 4),
        "mean_dice": round(mean_dice, 4),
    }
