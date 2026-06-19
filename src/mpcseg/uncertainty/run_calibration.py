"""[B3] Post-hoc calibration of a trained PTv3 model via temperature scaling.

Splits the val set in half: fit a single temperature T on the first half (calibration),
report Expected Calibration Error (ECE) before/after on the second half (test), and
save a reliability diagram. Argmax (hence mIoU) is unchanged by temperature scaling —
only the confidence is rescaled. See mpcseg.uncertainty.temperature_scaling.

Runs on the GPU box. Example:
    python -m mpcseg.uncertainty.run_calibration \
        --config /root/Pointcept/configs/fractal/semseg-pt-v3m1-1-flash.py \
        --checkpoint /root/Pointcept/exp/fractal/flash30/model/model_best.pth \
        --out-dir /root/autodl-tmp/fractal/calib --max-points 20000
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from mpcseg.infer.predict import _load_model
from mpcseg.uncertainty.temperature_scaling import (
    expected_calibration_error,
    fit_temperature,
    reliability_curve,
    softmax,
)


def _collect(model, dataset, indices, max_points: int, seed: int = 0):
    """Run the model over scenes, return stacked (logits, gt) at gridded resolution."""
    import torch

    from pointcept.datasets import point_collate_fn

    rng = np.random.default_rng(seed)
    logits_all, gt_all = [], []
    for idx in indices:
        batch = point_collate_fn([dataset[idx]])
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda(non_blocking=True)
        with torch.no_grad():
            logits = model(batch)["seg_logits"].float().cpu().numpy()
        gt = batch["segment"].cpu().numpy().reshape(-1)
        if max_points and len(gt) > max_points:
            sel = rng.choice(len(gt), size=max_points, replace=False)
            logits, gt = logits[sel], gt[sel]
        logits_all.append(logits)
        gt_all.append(gt)
    return np.concatenate(logits_all), np.concatenate(gt_all)


def _plot(out_png, xb, yb, xa, ya, ece_b, ece_a, T):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(xb, yb, "o-", color="#e53935", label=f"before (ECE {ece_b:.3f})")
    ax.plot(xa, ya, "s-", color="#1e88e5", label=f"after T={T:.2f} (ECE {ece_a:.3f})")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("Reliability diagram — temperature scaling")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-points", type=int, default=20000, help="per-scene point cap")
    args = ap.parse_args()

    from pointcept.datasets import build_dataset

    cfg, model = _load_model(args.config, args.checkpoint)
    val = build_dataset(cfg.data.val)
    n = len(val)
    half = n // 2
    calib_idx = list(range(half))
    test_idx = list(range(half, n))
    print(f"calibration scenes: {len(calib_idx)} | test scenes: {len(test_idx)}")

    Lc, Gc = _collect(model, val, calib_idx, args.max_points)
    Lt, Gt = _collect(model, val, test_idx, args.max_points, seed=1)

    T = fit_temperature(Lc, Gc)
    pb = softmax(Lt)
    pa = softmax(Lt / T)
    ece_b = expected_calibration_error(pb, Gt)
    ece_a = expected_calibration_error(pa, Gt)
    xb, yb = reliability_curve(pb, Gt)
    xa, ya = reliability_curve(pa, Gt)

    print(f"Temperature T = {T:.4f}")
    print(f"ECE before = {ece_b:.4f}")
    print(f"ECE after  = {ece_a:.4f}  ({100*(ece_b-ece_a)/max(ece_b,1e-9):.1f}% lower)")
    _plot(os.path.join(args.out_dir, "reliability.png"), xb, yb, xa, ya, ece_b, ece_a, T)
    np.save(os.path.join(args.out_dir, "temperature.npy"), np.array([T]))


if __name__ == "__main__":
    main()
