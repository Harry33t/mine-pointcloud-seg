"""[B1] Label-efficiency curve: mIoU vs label fraction (1 / 5 / 10 / 100%).

Compares self-supervised-init (Sonata) vs from-scratch at each fraction. This module
owns (a) deterministic label-fraction subsampling and (b) plotting; the actual
training runs are driven by Pointcept configs (see configs/pointcept/, docs/setup_gpu.md).

Status: subsampling implemented; plot implemented; the training-driver glue (launch
Pointcept per fraction, collect val mIoU) is a stub.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

FRACTIONS = (0.01, 0.05, 0.10, 1.00)


def sample_label_mask(num_points: int, fraction: float, seed: int = 0) -> np.ndarray:
    """Boolean mask selecting ``fraction`` of points to keep labelled (rest -> ignore).

    Point-level random sampling. For 'weak label' realism you may instead sample a
    few points per scene/class — swap this for a stratified sampler when needed.
    """
    rng = np.random.default_rng(seed)
    n_keep = max(1, int(round(num_points * fraction)))
    mask = np.zeros(num_points, dtype=bool)
    mask[rng.choice(num_points, size=n_keep, replace=False)] = True
    return mask


def apply_fraction_to_scene(scene_dir: str, fraction: float, seed: int = 0,
                            ignore_index: int = -1) -> Path:
    """Write a segment_<frac>.npy with only ``fraction`` of labels kept."""
    scene = Path(scene_dir)
    seg = np.load(scene / "segment.npy")
    mask = sample_label_mask(len(seg), fraction, seed)
    weak = np.where(mask, seg, ignore_index)
    out = scene / f"segment_{int(fraction * 100):03d}.npy"
    np.save(out, weak)
    return out


def build_fraction_dataset(src_train: str, out_train: str, fraction: float,
                           seed: int = 0, ignore_index: int = -1) -> int:
    """Create a label-fraction copy of a Pointcept train dir.

    coord/color/strength are symlinked (shared); segment.npy is rewritten keeping only
    ``fraction`` of the point labels (the rest set to ignore_index). Returns scene count.
    """
    import glob

    scenes = sorted(glob.glob(os.path.join(src_train, "*")))
    Path(out_train).mkdir(parents=True, exist_ok=True)
    for s in scenes:
        name = os.path.basename(s)
        d = Path(out_train) / name
        d.mkdir(exist_ok=True)
        for asset in ("coord", "color", "strength", "normal"):
            src = os.path.join(s, f"{asset}.npy")
            dst = d / f"{asset}.npy"
            if os.path.exists(src) and not dst.exists():
                os.symlink(os.path.abspath(src), dst)
        seg = np.load(os.path.join(s, "segment.npy"))
        if fraction >= 1.0:
            weak = seg
        else:
            mask = sample_label_mask(len(seg), fraction, seed)
            weak = np.where(mask, seg, ignore_index).astype(seg.dtype)
        np.save(d / "segment.npy", weak)
    return len(scenes)


def plot_curve(results: dict[str, dict[float, float]], out_png: str) -> None:
    """results: {'sonata': {0.01: miou, ...}, 'scratch': {...}}  ->  PNG curve."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, fr2miou in results.items():
        xs = sorted(fr2miou)
        ax.plot([x * 100 for x in xs], [fr2miou[x] for x in xs], marker="o", label=name)
    ax.set_xscale("log")
    ax.set_xlabel("labelled fraction (%)")
    ax.set_ylabel("mIoU")
    ax.set_title("Label-efficiency curve")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")


# --- TODO ---------------------------------------------------------------------
def run_curve(config, scenes, out_dir):  # noqa: D401 - stub
    """Launch Pointcept per (init, fraction), collect val mIoU, plot. TODO."""
    raise NotImplementedError("wire to Pointcept train.sh / val parsing")
