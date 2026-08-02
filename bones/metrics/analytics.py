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


def _rle_encode(mask: np.ndarray) -> dict:
    from pycocotools import mask as mask_utils
    return mask_utils.encode(np.asfortranarray(np.ascontiguousarray(mask, dtype=np.uint8)))


def compute_coco_map(
    per_image_gt: list[list[tuple[np.ndarray, int]]],
    per_image_dt: list[list[tuple[np.ndarray, int, float]]],
    class_ids: list[int],
    return_details: bool = False,
) -> dict[str, Any]:
    """COCO-style mAP via pycocotools COCOeval (iouType='segm').

    per_image_gt: per image, list of (binary mask, category_id)
    per_image_dt: per image, list of (binary mask, category_id, score)
    """
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    n_gt = sum(len(g) for g in per_image_gt)
    n_dt = sum(len(d) for d in per_image_dt)

    result: dict[str, Any] = {"mAP_50": 0.0, "mAP_50_95": 0.0}
    if return_details:
        result["per_class_ap"] = {}
        result["pr_curves"] = {}
    if n_gt == 0 or n_dt == 0:
        return result

    images = []
    gt_anns = []
    dt_anns = []
    ann_id = 1

    for i, (gt_list, dt_list) in enumerate(zip(per_image_gt, per_image_dt)):
        if not gt_list and not dt_list:
            continue
        h, w = gt_list[0][0].shape if gt_list else dt_list[0][0].shape
        img_id = i + 1
        images.append({"id": img_id, "width": int(w), "height": int(h)})

        for mask, cat in gt_list:
            rle = _rle_encode(mask)
            gt_anns.append({
                "id": ann_id, "image_id": img_id, "category_id": int(cat),
                "segmentation": rle, "area": float(mask_utils.area(rle)),
                "bbox": [float(v) for v in mask_utils.toBbox(rle)], "iscrowd": 0,
            })
            ann_id += 1

        for mask, cat, score in dt_list:
            rle = _rle_encode(mask)
            dt_anns.append({
                "id": ann_id, "image_id": img_id, "category_id": int(cat),
                "segmentation": rle, "area": float(mask_utils.area(rle)),
                "bbox": [float(v) for v in mask_utils.toBbox(rle)], "score": float(score),
            })
            ann_id += 1

    categories = [{"id": int(c), "name": str(int(c))} for c in class_ids]

    coco_gt = COCO()
    coco_gt.dataset = {"images": images, "annotations": gt_anns, "categories": categories}
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(dt_anns)

    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    precision_arr = coco_eval.eval["precision"]  # (T, R, K, A, M)

    def _ap(iou_thr: float | None, class_idx: int | None = None) -> float:
        s = precision_arr
        if iou_thr is not None:
            t = np.where(np.isclose(coco_eval.params.iouThrs, iou_thr))[0]
            s = s[t]
        if class_idx is not None:
            s = s[:, :, [class_idx]]
        s = s[:, :, :, 0, -1]  # area 'all', maxDets last
        vals = s[s > -1]
        return float(np.mean(vals)) if len(vals) > 0 else 0.0

    result = {
        "mAP_50": round(_ap(0.5), 4),
        "mAP_50_95": round(_ap(None), 4),
    }

    if return_details:
        rec_thrs = coco_eval.params.recThrs
        area_idx = 0
        max_dets_idx = len(coco_eval.params.maxDets) - 1
        per_class_ap: dict[int, dict[str, float]] = {}
        pr_curves: dict[int, dict[str, Any]] = {}

        for k, cid in enumerate(class_ids):
            ap_50 = _ap(0.5, k)
            ap_50_95 = _ap(None, k)
            per_class_ap[cid] = {
                "AP_50": round(ap_50, 4),
                "AP_50_95": round(ap_50_95, 4),
            }

            p_curve = precision_arr[0, :, k, area_idx, max_dets_idx]
            if np.all(p_curve < 0):
                continue
            pr_curves[cid] = {
                "precision": [round(float(v), 4) for v in p_curve],
                "recall": [round(float(r), 4) for r in rec_thrs],
                "AP_50": round(ap_50, 4),
            }

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
