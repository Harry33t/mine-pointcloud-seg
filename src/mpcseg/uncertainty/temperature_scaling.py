"""Per-point uncertainty + post-hoc calibration for semantic segmentation.

Approach (cheapest defensible option for a single PTv3 model):
  * per-point predictive uncertainty = softmax entropy of the logits;
  * calibration = temperature scaling — fit one scalar T on a validation split by
    minimising NLL; argmax (hence mIoU) is unchanged, only confidence is rescaled.

Report expected calibration error (ECE) before/after + a reliability diagram.

Note: a single softmax model gives *total/aleatoric* uncertainty only. Do not call
this "epistemic" — that needs an ensemble (a few seeds) or MC-dropout, which is not
available on a pretrained PTv3 (its dropout rates are 0). See docs/pipeline.md.

Inputs are numpy logits of shape (N, C) and int labels (N,) with -1 = ignore.
"""
from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-point predictive entropy (natural log). Shape (N,)."""
    return -(probs * np.log(probs + eps)).sum(axis=1)


def fit_temperature(
    logits: np.ndarray, labels: np.ndarray, lr: float = 0.01, iters: int = 200
) -> float:
    """Fit temperature T>0 minimising NLL on (logits, labels) via torch LBFGS.

    Falls back to a numpy grid search if torch is unavailable.
    """
    mask = labels >= 0
    logits, labels = logits[mask], labels[mask]
    try:
        import torch

        z = torch.from_numpy(logits.astype(np.float32))
        y = torch.from_numpy(labels.astype(np.int64))
        log_T = torch.zeros(1, requires_grad=True)  # optimise log T to keep T>0
        opt = torch.optim.LBFGS([log_T], lr=lr, max_iter=iters)
        nll = torch.nn.CrossEntropyLoss()

        def closure():
            opt.zero_grad()
            loss = nll(z / log_T.exp(), y)
            loss.backward()
            return loss

        opt.step(closure)
        return float(log_T.exp().item())
    except ImportError:
        best_T, best_nll = 1.0, np.inf
        for T in np.linspace(0.5, 5.0, 91):
            p = softmax(logits / T)
            nll = -np.log(p[np.arange(len(labels)), labels] + 1e-12).mean()
            if nll < best_nll:
                best_T, best_nll = float(T), nll
        return best_T


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """ECE over confidence bins. Ignores points with label < 0."""
    mask = labels >= 0
    probs, labels = probs[mask], labels[mask]
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.sum() == 0:
            continue
        ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def reliability_curve(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin_confidence, bin_accuracy) for a reliability diagram."""
    mask = labels >= 0
    probs, labels = probs[mask], labels[mask]
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        xs.append(conf[m].mean())
        ys.append(correct[m].mean())
    return np.array(xs), np.array(ys)


def calibrate(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    test_logits: np.ndarray,
    test_labels: np.ndarray,
) -> dict:
    """Fit T on val, report ECE before/after on test, and return calibrated outputs."""
    T = fit_temperature(val_logits, val_labels)
    p_before = softmax(test_logits)
    p_after = softmax(test_logits / T)
    return {
        "temperature": T,
        "ece_before": expected_calibration_error(p_before, test_labels),
        "ece_after": expected_calibration_error(p_after, test_labels),
        "probs": p_after,
        "entropy": entropy(p_after),
    }
