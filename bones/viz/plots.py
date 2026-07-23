from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bones.config import CATEGORIES, VIS


def _class_names() -> list[str]:
    return [CATEGORIES[cid] for cid in sorted(CATEGORIES.keys())]


def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_pr_curves(
    pr_curves: dict[int, dict],
    save_path: str,
    class_names: list[str] | None = None,
):
    if class_names is None:
        class_names = _class_names()
    fig, ax = plt.subplots(figsize=(8, 6))
    for cid, data in pr_curves.items():
        prec = data.get("precision", [])
        rec = data.get("recall", [])
        if not prec or not rec:
            continue
        name = class_names[cid - 1] if cid - 1 < len(class_names) else str(cid)
        ap = data.get("ap_50", None)
        label = f"{name} (AP={ap:.3f})" if ap else name
        ax.plot(rec, prec, label=label, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_ap_barchart(
    per_class_ap: dict[int, dict],
    save_path: str,
    class_names: list[str] | None = None,
):
    if class_names is None:
        class_names = _class_names()
    cids = sorted(per_class_ap.keys())
    names = [class_names[cid - 1] if cid - 1 < len(class_names) else str(cid) for cid in cids]
    ap50 = [per_class_ap[cid].get("AP_50", 0.0) for cid in cids]
    ap5095 = [per_class_ap[cid].get("AP_50_95", 0.0) for cid in cids]

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, ap50, w, label="AP@0.5")
    ax.bar(x + w / 2, ap5095, w, label="AP@0.5:0.95")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Average Precision")
    ax.set_title("Per-Class Average Precision")
    ax.legend()
    ax.set_ylim(0, 1)
    for i in range(len(cids)):
        ax.text(i - w / 2, ap50[i] + 0.02, f"{ap50[i]:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, ap5095[i] + 0.02, f"{ap5095[i]:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_f1_vs_threshold(
    f1_data: dict[int, dict],
    save_path: str,
    class_names: list[str] | None = None,
):
    if class_names is None:
        class_names = _class_names()
    fig, ax = plt.subplots(figsize=(8, 6))
    for cid, data in f1_data.items():
        thresh = data.get("thresholds", [])
        f1 = data.get("f1_scores", [])
        if not thresh or not f1:
            continue
        name = class_names[cid - 1] if cid - 1 < len(class_names) else str(cid)
        ax.plot(thresh, f1, label=name, lw=2)
        best_idx = int(np.argmax(f1))
        ax.plot(thresh[best_idx], f1[best_idx], "o", markersize=8)
        ax.annotate(f"{f1[best_idx]:.3f} @ {thresh[best_idx]:.2f}",
                     (thresh[best_idx], f1[best_idx]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("F1-Score")
    ax.set_title("F1-Score vs. Confidence Threshold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    cm_data: dict,
    save_path: str,
    class_names: list[str] | None = None,
):
    if class_names is None:
        class_names = _class_names()
    matrix = np.array(cm_data["matrix"])
    labels = class_names + ["background"]
    n = len(labels)

    row_sums = matrix.sum(axis=1, keepdims=True)
    norm = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(max(6, n * 1.5), max(5, n * 1.2)))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Normalized Confusion Matrix")

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            pct = norm[i, j]
            color = "white" if pct > 0.5 else "black"
            ax.text(j, i, f"{val}\n({pct:.0%})", ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_tide_errors(
    tide_data: dict[int, dict],
    save_path: str,
    class_names: list[str] | None = None,
):
    if class_names is None:
        class_names = _class_names()
    cids = sorted(tide_data.keys())
    names = [class_names[cid - 1] if cid - 1 < len(class_names) else str(cid) for cid in cids]
    categories = ["class_error", "loc_error", "background_fp", "missed_gt"]
    cat_labels = ["Class Error", "Loc Error", "Bg FP", "Missed GT"]
    colors = ["#e74c3c", "#f39c12", "#3498db", "#95a5a6"]

    values = np.zeros((len(cids), len(categories)))
    for i, cid in enumerate(cids):
        for j, cat in enumerate(categories):
            values[i, j] = tide_data[cid].get(cat, 0)

    fig, ax = plt.subplots(figsize=(max(8, len(cids) * 2), 5))
    bottom = np.zeros(len(cids))
    for j in range(len(categories)):
        ax.bar(names, values[:, j], bottom=bottom, label=cat_labels[j], color=colors[j])
        bottom += values[:, j]
    ax.set_ylabel("Error Count")
    ax.set_title("Error Breakdown (TIDE-style)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_iou_histogram(
    matched_ious_per_class: dict[int, list[float]],
    save_path: str,
    class_names: list[str] | None = None,
):
    if class_names is None:
        class_names = _class_names()
    fig, ax = plt.subplots(figsize=(8, 5))
    cids = sorted(matched_ious_per_class.keys())
    for cid in cids:
        ious = matched_ious_per_class[cid]
        if not ious:
            continue
        name = class_names[cid - 1] if cid - 1 < len(class_names) else str(cid)
        ax.hist(ious, bins=20, alpha=0.5, label=name, range=(0, 1))
    ax.set_xlabel("IoU")
    ax.set_ylabel("Frequency")
    ax.set_title("IoU Distribution of Matched Detections")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _draw_overlay(
    image_np: np.ndarray,
    masks: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray | None,
    class_names: list[str],
    color: tuple[int, int, int],
    score_threshold: float = 0.0,
) -> np.ndarray:
    disp = image_np.copy()
    for i in range(len(labels)):
        if scores is not None and scores[i] < score_threshold:
            continue
        if labels[i] == 0:
            continue
        cid = int(labels[i])
        name = class_names[cid - 1] if cid - 1 < len(class_names) else str(cid)
        x1, y1, x2, y2 = map(int, boxes[i])
        cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
        label = name
        if scores is not None:
            label += f" {scores[i]:.2f}"
        cv2.putText(disp, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if masks is not None and i < len(masks):
            m = masks[i]
            if m.ndim == 3 and m.shape[0] == 1:
                m = m[0]
            mask = m > 0.5 if m.dtype.kind == "f" else m > 0
            overlay = np.zeros_like(disp, dtype=np.uint8)
            overlay[mask] = color
            disp = cv2.addWeighted(disp, 1.0, overlay, 0.3, 0)
    return disp


def _tensor_to_np(image_tensor) -> np.ndarray:
    arr = image_tensor.cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.max() <= 1.0:
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def plot_overlay_grid(
    images: list,
    targets: list[dict],
    outputs: list[dict],
    save_path: str,
    class_names: list[str] | None = None,
    n: int = 9,
    score_threshold: float = 0.5,
):
    if class_names is None:
        class_names = _class_names()
    n = min(n, len(images))
    if n == 0:
        return
    indices = np.linspace(0, len(images) - 1, n, dtype=int)
    rows = int(np.ceil(n / 3))
    fig, axes = plt.subplots(rows, min(n, 3), figsize=(min(n, 3) * 6, rows * 4))
    axes = np.atleast_1d(axes).flatten()

    for idx, ax_idx in enumerate(indices):
        img_np = _tensor_to_np(images[ax_idx])
        target = targets[ax_idx]
        output = outputs[ax_idx]

        gt_boxes = target["boxes"].cpu().numpy() if hasattr(target["boxes"], "cpu") else np.array(target["boxes"])
        gt_labels = target["labels"].cpu().numpy() if hasattr(target["labels"], "cpu") else np.array(target["labels"])
        gt_masks = target["masks"].cpu().numpy() if hasattr(target["masks"], "cpu") else np.array(target["masks"])

        pred_boxes = output["boxes"].cpu().numpy() if hasattr(output["boxes"], "cpu") else np.array(output["boxes"])
        pred_labels = output["labels"].cpu().numpy() if hasattr(output["labels"], "cpu") else np.array(output["labels"])
        pred_masks = output["masks"].cpu().numpy() if hasattr(output["masks"], "cpu") else np.array(output["masks"])
        pred_scores = output["scores"].cpu().numpy() if hasattr(output["scores"], "cpu") else np.array(output["scores"])

        gt_disp = _draw_overlay(img_np, gt_masks, gt_boxes, gt_labels, None, class_names, (0, 180, 0))
        pred_disp = _draw_overlay(img_np, pred_masks, pred_boxes, pred_labels, pred_scores, class_names, (0, 0, 220), score_threshold)
        combined = np.concatenate([gt_disp, pred_disp], axis=1)
        axes[idx].imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        axes[idx].axis("off")
        axes[idx].set_title(f"Image {ax_idx}")

    for idx in range(len(indices), len(axes)):
        axes[idx].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_failure_cases(
    per_image_errors: list[dict],
    images: list,
    targets: list[dict],
    outputs: list[dict],
    save_path: str,
    class_names: list[str] | None = None,
    n: int = 9,
    score_threshold: float = 0.5,
):
    if class_names is None:
        class_names = _class_names()
    if not per_image_errors or len(images) == 0:
        return
    total_errors = [e.get("fp", 0) + e.get("fn", 0) for e in per_image_errors]
    worst_indices = np.argsort(total_errors)[::-1][: min(n, len(images))]
    worst_errors = [total_errors[i] for i in worst_indices]

    rows = int(np.ceil(len(worst_indices) / 3))
    fig, axes = plt.subplots(rows, min(len(worst_indices), 3), figsize=(min(len(worst_indices), 3) * 6, rows * 4))
    axes = np.atleast_1d(axes).flatten()

    for idx, img_idx in enumerate(worst_indices):
        img_np = _tensor_to_np(images[img_idx])
        target = targets[img_idx]
        output = outputs[img_idx]

        gt_boxes = target["boxes"].cpu().numpy() if hasattr(target["boxes"], "cpu") else np.array(target["boxes"])
        gt_labels = target["labels"].cpu().numpy() if hasattr(target["labels"], "cpu") else np.array(target["labels"])
        gt_masks = target["masks"].cpu().numpy() if hasattr(target["masks"], "cpu") else np.array(target["masks"])

        pred_boxes = output["boxes"].cpu().numpy() if hasattr(output["boxes"], "cpu") else np.array(output["boxes"])
        pred_labels = output["labels"].cpu().numpy() if hasattr(output["labels"], "cpu") else np.array(output["labels"])
        pred_masks = output["masks"].cpu().numpy() if hasattr(output["masks"], "cpu") else np.array(output["masks"])
        pred_scores = output["scores"].cpu().numpy() if hasattr(output["scores"], "cpu") else np.array(output["scores"])

        gt_disp = _draw_overlay(img_np, gt_masks, gt_boxes, gt_labels, None, class_names, (0, 180, 0))
        pred_disp = _draw_overlay(img_np, pred_masks, pred_boxes, pred_labels, pred_scores, class_names, (0, 0, 220), score_threshold)
        combined = np.concatenate([gt_disp, pred_disp], axis=1)
        axes[idx].imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        axes[idx].axis("off")
        axes[idx].set_title(f"Image {img_idx} | Errors: {worst_errors[idx]}")

    for idx in range(len(worst_indices), len(axes)):
        axes[idx].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_cross_fold_metrics(
    all_fold_metrics: list[dict],
    save_path: str,
    class_names: list[str] | None = None,
):
    if class_names is None:
        class_names = _class_names()
    keys = ["precision", "recall", "f1_score", "mean_iou", "mean_dice"]
    key_labels = ["Precision", "Recall", "F1", "mIoU", "mDice"]

    n_metrics = len(keys)
    x = np.arange(n_metrics)
    w = 0.8 / len(class_names)

    fig, ax = plt.subplots(figsize=(max(8, n_metrics * 2), 5))
    for ci, cat_name in enumerate(class_names):
        means = []
        stds = []
        for key in keys:
            vals = [m.get(cat_name, {}).get(key, 0.0) for m in all_fold_metrics]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        offset = (ci - len(class_names) / 2 + 0.5) * w
        ax.bar(x + offset, means, w, yerr=stds, label=cat_name, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(key_labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Cross-Fold Metrics (mean ± std)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_all_figures(
    metrics: dict,
    eval_data: dict,
    save_dir: str,
):
    save_path = _ensure_dir(save_dir)
    class_names = _class_names()

    pr_curves = eval_data.get("pr_curves", {})
    if pr_curves:
        plot_pr_curves(pr_curves, str(save_path / "pr_curves.png"), class_names)

    per_class_ap = eval_data.get("per_class_ap", {})
    if per_class_ap:
        plot_ap_barchart(per_class_ap, str(save_path / "ap_barchart.png"), class_names)

    f1_data = eval_data.get("f1_data", {})
    if f1_data:
        plot_f1_vs_threshold(f1_data, str(save_path / "f1_vs_threshold.png"), class_names)

    cm_data = metrics.get("confusion_matrix")
    if cm_data:
        plot_confusion_matrix(cm_data, str(save_path / "confusion_matrix.png"), class_names)

    tide_data = eval_data.get("tide_data", {})
    if tide_data:
        plot_tide_errors(tide_data, str(save_path / "tide_errors.png"), class_names)

    ious = eval_data.get("matched_ious_per_class", {})
    if ious:
        plot_iou_histogram(ious, str(save_path / "iou_histogram.png"), class_names)

    if VIS.get("save_overlays", True):
        images = eval_data.get("images", [])
        targets = eval_data.get("targets", [])
        outputs = eval_data.get("outputs", [])
        if images:
            plot_overlay_grid(images, targets, outputs, str(save_path / "overlay_samples.png"), class_names)

        errors = eval_data.get("per_image_errors", [])
        if errors and images:
            plot_failure_cases(errors, images, targets, outputs, str(save_path / "failure_cases.png"), class_names,
                               n=VIS.get("n_failure_cases", 9))
