"""[B2] Build geographic "sites" from FRACTAL tiles for leave-one-site-out transfer.

FRACTAL tile names encode a Lambert-93 km block: TEST-<X>_<Y>-<id>. We k-means the
(X, Y) block centres into N sites (distinct regions of France -> real cross-site
domain shift), then split each site into train/val with symlinked assets.

    python -m mpcseg.evaluate.loco_setup --src /root/autodl-tmp/fractal/pretrain/train \
        --out /root/autodl-tmp/fractal/loco --sites 4 --val-cap 80
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np

_NAME = re.compile(r"TEST-(\d+)_(\d+)-")


def parse_xy(name: str):
    m = _NAME.match(name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def setup(src: str, out_root: str, n_sites: int = 4, val_cap: int = 80,
          val_frac: float = 0.2, seed: int = 0) -> None:
    from sklearn.cluster import KMeans

    scenes = sorted(glob.glob(os.path.join(src, "*")))
    names = [os.path.basename(s) for s in scenes]
    coords = np.array([parse_xy(n) for n in names], dtype=float)

    km = KMeans(n_clusters=n_sites, random_state=seed, n_init=10).fit(coords)
    labels = km.labels_
    rng = np.random.default_rng(seed)

    for k in range(n_sites):
        idx = [i for i in range(len(scenes)) if labels[i] == k]
        rng.shuffle(idx)
        n_val = min(val_cap, max(1, int(len(idx) * val_frac)))
        val_idx = set(idx[:n_val])
        for split in ("train", "val"):
            os.makedirs(os.path.join(out_root, f"site_{k}", split), exist_ok=True)
        for i in idx:
            split = "val" if i in val_idx else "train"
            d = os.path.join(out_root, f"site_{k}", split, names[i])
            os.makedirs(d, exist_ok=True)
            for asset in ("coord", "color", "strength", "segment"):
                srcf = os.path.join(scenes[i], f"{asset}.npy")
                dstf = os.path.join(d, f"{asset}.npy")
                if os.path.exists(srcf) and not os.path.exists(dstf):
                    os.symlink(os.path.abspath(srcf), dstf)
        cx, cy = km.cluster_centers_[k].round(0)
        print(f"site_{k}: {len(idx)} scenes ({len(idx) - n_val} train / {n_val} val) "
              f"centroid X={cx:.0f} Y={cy:.0f} km")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="/root/autodl-tmp/fractal/pretrain/train")
    ap.add_argument("--out", default="/root/autodl-tmp/fractal/loco")
    ap.add_argument("--sites", type=int, default=4)
    ap.add_argument("--val-cap", type=int, default=80)
    args = ap.parse_args()
    setup(args.src, args.out, args.sites, args.val_cap)


if __name__ == "__main__":
    main()
