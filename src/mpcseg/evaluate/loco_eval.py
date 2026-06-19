"""[B2] Cross-evaluate per-site models to build the LOCO transfer matrix.

For every (train_site i, test_site j) pair, load model_i and compute mIoU on site_j's
val scenes (gridded resolution). Rows = train site, cols = test site; the diagonal is
in-domain, off-diagonal is transfer. Saves a heatmap.

    python -m mpcseg.evaluate.loco_eval --config /root/Pointcept/configs/fractal/semseg-spunet-loco.py \
        --exp-root /root/Pointcept/exp/fractal --loco-root /root/autodl-tmp/fractal/loco \
        --sites 4 --out /root/autodl-tmp/fractal/loco/loco_matrix.png
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from mpcseg.evaluate.metrics import confusion_matrix, iou_from_confusion
from mpcseg.evaluate.loco_matrix import plot_heatmap, summarise
from mpcseg.infer.predict import _load_model


def eval_model_on_site(model, dataset, num_classes: int) -> float:
    import torch
    from pointcept.datasets import point_collate_fn

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for idx in range(len(dataset)):
        batch = point_collate_fn([dataset[idx]])
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda(non_blocking=True)
        with torch.no_grad():
            logits = model(batch)["seg_logits"].float()
        pred = logits.argmax(1).cpu().numpy()
        gt = batch["segment"].cpu().numpy().reshape(-1)
        cm += confusion_matrix(pred, gt, num_classes)
    return float(np.nanmean(iou_from_confusion(cm)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp-root", required=True)
    ap.add_argument("--loco-root", required=True)
    ap.add_argument("--sites", type=int, default=4)
    ap.add_argument("--num-classes", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from pointcept.datasets import build_dataset

    sites = list(range(args.sites))
    val_sets = {}  # site_j -> dataset, built lazily once cfg is available

    M = np.full((args.sites, args.sites), np.nan)
    for i in sites:
        ckpt = os.path.join(args.exp_root, f"loco_site_{i}", "model", "model_best.pth")
        if not os.path.exists(ckpt):
            ckpt = os.path.join(args.exp_root, f"loco_site_{i}", "model", "model_last.pth")
        if not os.path.exists(ckpt):
            print(f"[skip] no checkpoint for site {i}")
            continue
        cfg, model = _load_model(args.config, ckpt)
        for j in sites:
            if j not in val_sets:
                cfg.data.val["data_root"] = os.path.join(args.loco_root, f"site_{j}")
                cfg.data.val["split"] = "val"
                val_sets[j] = build_dataset(cfg.data.val)
            M[i, j] = eval_model_on_site(model, val_sets[j], args.num_classes)
            print(f"train site_{i} -> test site_{j}: mIoU {M[i, j]:.4f}")

    names = [f"site_{k}" for k in sites]
    s = summarise(M)
    print(f"mean diag (in-domain) {s['mean_diag']:.4f} | mean off-diag (transfer) "
          f"{s['mean_offdiag']:.4f} | generalisation gap {s['gap']:.4f}")
    plot_heatmap(M, names, args.out)
    np.save(os.path.join(os.path.dirname(args.out), "loco_matrix.npy"), M)


if __name__ == "__main__":
    main()
