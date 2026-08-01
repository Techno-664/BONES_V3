from __future__ import annotations

from typing import Any

import numpy as np
import torch

from bones.metrics.matching import compute_iou_matrix


def _iou_matrix_from_lists(
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
) -> np.ndarray:
    all_masks = gt_masks + pred_masks
    target_h = max(m.shape[0] for m in all_masks)
    target_w = max(m.shape[1] for m in all_masks)

    def _resize(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        if mask.shape == (out_h, out_w):
            return mask
        t = torch.from_numpy(mask.astype(float)).unsqueeze(0).unsqueeze(0)
        interp = torch.nn.functional.interpolate(t, size=(out_h, out_w), mode="nearest")
        return interp.squeeze().numpy().astype(np.uint8)

    gt_t = torch.from_numpy(np.stack([_resize(m, target_h, target_w) for m in gt_masks], axis=0))
    pred_t = torch.from_numpy(np.stack([_resize(m, target_h, target_w) for m in pred_masks], axis=0))
    return compute_iou_matrix(gt_t, pred_t)


def _greedy_match(
    iou_matrix: np.ndarray,
    gt_matched: np.ndarray,
    pred_matched: np.ndarray,
    iou_threshold: float,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    while True:
        available = iou_matrix[~gt_matched][:, ~pred_matched]
        if not available.size:
            break
        max_iou = float(available.max())
        if max_iou < iou_threshold:
            break
        flat_max = int(available.argmax())
        n_col = available.shape[1]
        gi_rel = flat_max // n_col
        pi_rel = flat_max % n_col
        gi = int(np.where(~gt_matched)[0][gi_rel])
        pi = int(np.where(~pred_matched)[0][pi_rel])
        gt_matched[gi] = True
        pred_matched[pi] = True
        pairs.append((gi, pi))
    return pairs


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    idx = np.where(recall[1:] != recall[:-1])[0]
    if len(idx) == 0:
        return 0.0
    return float(((recall[idx + 1] - recall[idx]) * precision[idx + 1]).sum())


def compute_map(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
    pred_scores: list[float],
    pred_labels: list[int],
    gt_labels: list[int],
    class_ids: list[int],
    iou_thresholds: list[float] | None = None,
    return_details: bool = False,
) -> dict[str, Any]:
    if iou_thresholds is None:
        iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]

    n_gt = len(gt_masks)
    n_pred = len(pred_masks)
    result: dict[str, Any]
    if n_gt == 0 or n_pred == 0:
        result = {"mAP_50": 0.0, "mAP_50_95": 0.0}
        if return_details:
            result["per_class_ap"] = {}
            result["pr_curves"] = {}
        return result

    iou_matrix = _iou_matrix_from_lists(gt_masks, pred_masks)

    aps: list[float] = []
    per_class_ap: dict[int, dict[str, float]] = {cid: {"AP_50": 0.0, "AP_50_95": 0.0} for cid in class_ids}
    pr_curves: dict[int, dict[str, Any]] = {}

    for iou_threshold in iou_thresholds:
        class_aps = []
        for cid in class_ids:
            gt_idx = [i for i, l in enumerate(gt_labels) if l == cid]
            pred_idx = [j for j, l in enumerate(pred_labels) if l == cid]
            if not gt_idx:
                continue
            if not pred_idx:
                class_aps.append(0.0)
                if iou_threshold == iou_thresholds[0]:
                    per_class_ap[cid]["AP_50"] = 0.0
                continue

            matches = iou_matrix[np.ix_(gt_idx, pred_idx)] >= iou_threshold
            gt_matched = np.zeros(len(gt_idx), dtype=bool)
            pred_matched = np.zeros(len(pred_idx), dtype=bool)

            scores = np.array([pred_scores[j] for j in pred_idx])
            sort_order = np.argsort(-scores)

            tp = np.zeros(len(pred_idx), dtype=bool)
            fp = np.zeros(len(pred_idx), dtype=bool)

            for rank, pi in enumerate(sort_order):
                candidates = np.where(matches[:, pi] & ~gt_matched)[0]
                if len(candidates) > 0:
                    gt_matched[candidates[0]] = True
                    pred_matched[pi] = True
                    tp[rank] = True
                else:
                    fp[rank] = True

            cum_tp = np.cumsum(tp).astype(float)
            cum_fp = np.cumsum(fp).astype(float)
            precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)
            recall = cum_tp / len(gt_idx)
            ap = compute_ap(recall, precision)
            class_aps.append(ap)

            if return_details and iou_threshold == iou_thresholds[0]:
                per_class_ap[cid]["AP_50"] = round(ap, 4)
                pr_curves[cid] = {
                    "precision": [round(float(p), 4) for p in precision],
                    "recall": [round(float(r), 4) for r in recall],
                    "scores": [round(float(s), 4) for s in scores[sort_order]],
                }

        if class_aps:
            aps.append(float(np.mean(class_aps)))

    map_50 = aps[0] if len(aps) > 0 else 0.0
    map_50_95 = float(np.mean(aps)) if aps else 0.0
    result = {"mAP_50": round(map_50, 4), "mAP_50_95": round(map_50_95, 4)}
    if return_details:
        result["per_class_ap"] = per_class_ap
        result["pr_curves"] = pr_curves
    return result


def compute_f1_vs_threshold(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
    pred_scores: list[float],
    pred_labels: list[int],
    gt_labels: list[int],
    class_ids: list[int],
    iou_threshold: float = 0.5,
    step: float = 0.05,
) -> dict[int, Any]:
    n_gt = len(gt_masks)
    n_pred = len(pred_masks)
    thresholds = np.arange(step, 1.0, step)
    result: dict[int, Any] = {}

    if n_gt == 0 or n_pred == 0:
        for cid in class_ids:
            result[cid] = {"thresholds": thresholds.tolist(), "f1_scores": [0.0] * len(thresholds)}
        return result

    iou_matrix = _iou_matrix_from_lists(gt_masks, pred_masks)

    for cid in class_ids:
        gt_idx = [i for i, l in enumerate(gt_labels) if l == cid]
        pred_idx_all = [j for j, l in enumerate(pred_labels) if l == cid]
        if not gt_idx:
            result[cid] = {"thresholds": thresholds.tolist(), "f1_scores": [0.0] * len(thresholds)}
            continue

        f1_vals = []
        precision_vals = []
        recall_vals = []

        for threshold in thresholds:
            pred_idx = [j for j in pred_idx_all if pred_scores[j] >= threshold]
            if not pred_idx:
                f1_vals.append(0.0)
                precision_vals.append(0.0)
                recall_vals.append(0.0)
                continue

            sub = iou_matrix[np.ix_(gt_idx, pred_idx)]
            gt_m = np.zeros(len(gt_idx), dtype=bool)
            pred_m = np.zeros(len(pred_idx), dtype=bool)

            pairs = _greedy_match(sub, gt_m, pred_m, iou_threshold)
            tp = len(pairs)

            fp = len(pred_idx) - tp
            fn = len(gt_idx) - tp
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            f1_vals.append(round(f1, 4))
            precision_vals.append(round(p, 4))
            recall_vals.append(round(r, 4))

        result[cid] = {
            "thresholds": [round(float(t), 2) for t in thresholds],
            "f1_scores": f1_vals,
            "precisions": precision_vals,
            "recalls": recall_vals,
        }

    return result


def confusion_matrix(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
    pred_labels: list[int],
    gt_labels: list[int],
    class_ids: list[int],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    n_gt = len(gt_masks)
    n_pred = len(pred_masks)

    cmap = {cid: i for i, cid in enumerate(class_ids)}
    n = len(class_ids)
    cm = np.zeros((n + 1, n + 1), dtype=int)

    if n_gt == 0 or n_pred == 0:
        return {"matrix": cm.tolist(), "class_ids": class_ids}

    iou_matrix = _iou_matrix_from_lists(gt_masks, pred_masks)

    gt_matched = np.zeros(n_gt, dtype=bool)
    pred_matched = np.zeros(n_pred, dtype=bool)

    pairs = _greedy_match(iou_matrix, gt_matched, pred_matched, iou_threshold)
    for gi, pi in pairs:
        cm[cmap[gt_labels[gi]], cmap[pred_labels[pi]]] += 1

    for gi, matched in enumerate(gt_matched):
        if not matched:
            cm[cmap[gt_labels[gi]], -1] += 1

    for pi, matched in enumerate(pred_matched):
        if not matched:
            cm[-1, cmap[pred_labels[pi]]] += 1

    return {"matrix": cm.tolist(), "class_ids": class_ids}


def compute_tide_errors(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
    pred_labels: list[int],
    gt_labels: list[int],
    class_ids: list[int],
    iou_threshold: float = 0.5,
) -> dict[int, dict[str, int]]:
    n_gt = len(gt_masks)
    n_pred = len(pred_masks)
    error_counts: dict[int, dict[str, int]] = {
        cid: {"class_error": 0, "loc_error": 0, "background_fp": 0, "missed_gt": 0}
        for cid in class_ids
    }

    if n_gt == 0:
        for cid in class_ids:
            error_counts[cid]["background_fp"] = sum(1 for l in pred_labels if l == cid)
        return error_counts
    if n_pred == 0:
        for cid in class_ids:
            error_counts[cid]["missed_gt"] = sum(1 for l in gt_labels if l == cid)
        return error_counts

    iou_matrix = _iou_matrix_from_lists(gt_masks, pred_masks)

    gt_matched = np.zeros(n_gt, dtype=bool)
    pred_matched = np.zeros(n_pred, dtype=bool)
    pred_to_gt: dict[int, int] = {}

    pairs = _greedy_match(iou_matrix, gt_matched, pred_matched, iou_threshold)
    for gi, pi in pairs:
        pred_to_gt[pi] = gi

    for pi in range(n_pred):
        p_label = pred_labels[pi]
        if pi in pred_to_gt:
            gi = pred_to_gt[pi]
            gt_label = gt_labels[gi]
            if p_label == gt_label:
                continue
            error_counts[p_label]["class_error"] += 1
        else:
            error_counts[p_label]["background_fp"] += 1

    for gi in range(n_gt):
        if not gt_matched[gi]:
            error_counts[gt_labels[gi]]["missed_gt"] += 1

    return error_counts


def sensitivity_specificity(cm: dict[str, Any]) -> dict[str, Any]:
    matrix = np.array(cm["matrix"], dtype=int)
    class_ids = cm["class_ids"]
    n = len(class_ids)

    results: dict[str, Any] = {}
    for i, cid in enumerate(class_ids):
        row = matrix[i, :n].sum()
        col = matrix[:n, i].sum()
        total = matrix[:n, :n].sum()

        tp = int(matrix[i, i])
        fn = int(row) + int(matrix[i, n]) - tp
        fp = int(col) + int(matrix[n, i]) - tp
        tn = int(total) - tp - fn - fp

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        results[str(cid)] = {
            "sensitivity": round(float(tpr), 4),
            "specificity": round(float(tnr), 4),
            "accuracy": round(float(accuracy), 4),
            "far": round(float(far), 4),
            "frr": round(float(frr), 4),
            "ppv": round(float(ppv), 4),
            "npv": round(float(npv), 4),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        }

    tp_sum = sum(r["tp"] for r in results.values())
    fn_sum = sum(r["fn"] for r in results.values())
    fp_sum = sum(r["fp"] for r in results.values())
    tn_sum = sum(r["tn"] for r in results.values())
    total_sum = tp_sum + tn_sum + fp_sum + fn_sum

    macro_tpr = float(np.mean([r["sensitivity"] for r in results.values()]))
    macro_tnr = float(np.mean([r["specificity"] for r in results.values()]))
    macro_acc = float(np.mean([r["accuracy"] for r in results.values()]))
    macro_far = float(np.mean([r["far"] for r in results.values()]))
    macro_frr = float(np.mean([r["frr"] for r in results.values()]))
    micro_tpr = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    micro_tnr = tn_sum / (tn_sum + fp_sum) if (tn_sum + fp_sum) > 0 else 0.0
    micro_acc = (tp_sum + tn_sum) / total_sum if total_sum > 0 else 0.0
    micro_far = fp_sum / (fp_sum + tn_sum) if (fp_sum + tn_sum) > 0 else 0.0
    micro_frr = fn_sum / (fn_sum + tp_sum) if (fn_sum + tp_sum) > 0 else 0.0

    results["macro_avg"] = {
        "sensitivity": round(macro_tpr, 4),
        "specificity": round(macro_tnr, 4),
        "accuracy": round(macro_acc, 4),
        "far": round(macro_far, 4),
        "frr": round(macro_frr, 4),
    }
    results["micro_avg"] = {
        "sensitivity": round(micro_tpr, 4),
        "specificity": round(micro_tnr, 4),
        "accuracy": round(micro_acc, 4),
        "far": round(micro_far, 4),
        "frr": round(micro_frr, 4),
    }

    return results


def multiclass_auc_roc(
    all_scores: list[dict[int, float]],
    all_label_dicts: list[dict[int, int]],
    class_ids: list[int],
    return_curve: bool = False,
) -> dict[int, Any]:
    results: dict[int, Any] = {}
    for cid in class_ids:
        y_true = [ld[cid] for ld in all_label_dicts]
        y_score = [s.get(cid, 0.0) for s in all_scores]

        n_pos = sum(y_true)
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            entry: dict[str, Any] = {"auc": 0.0}
            if return_curve:
                entry["tpr"] = []
                entry["fpr"] = []
            results[int(cid)] = entry
            continue

        pairs = sorted(zip(y_score, y_true), key=lambda x: -x[0])
        tpr_list = [0.0]
        fpr_list = [0.0]
        tp = 0
        fp = 0
        for score, true in pairs:
            if true == 1:
                tp += 1
            else:
                fp += 1
            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)

        auc = float(np.trapezoid(tpr_list, fpr_list))
        entry = {"auc": round(auc, 4)}
        if return_curve:
            entry["tpr"] = [round(float(x), 4) for x in tpr_list]
            entry["fpr"] = [round(float(x), 4) for x in fpr_list]
        results[int(cid)] = entry

    return results
