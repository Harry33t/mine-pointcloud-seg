"""Segmentation metrics: confusion matrix, per-class IoU, mIoU, OA.

Shared by the label-efficiency (B1) and LOCO (B2) building blocks. Label -1 = ignore.
"""
from __future__ import annotations

import numpy as np


def confusion_matrix(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> np.ndarray:
    mask = gt >= 0
    pred, gt = pred[mask], gt[mask]
    idx = gt * num_classes + pred
    return np.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def iou_from_confusion(cm: np.ndarray) -> np.ndarray:
    """Per-class IoU = TP / (TP + FP + FN). NaN for classes absent from gt+pred."""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(denom > 0, tp / denom, np.nan)
    return iou


def scores(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> dict:
    cm = confusion_matrix(pred, gt, num_classes)
    iou = iou_from_confusion(cm)
    tp = np.diag(cm).sum()
    total = cm.sum()
    return {
        "iou_per_class": iou,
        "miou": float(np.nanmean(iou)),
        "oa": float(tp / total) if total > 0 else float("nan"),
        "confusion": cm,
    }
