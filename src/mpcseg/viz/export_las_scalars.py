"""Write model outputs as LAS 1.4 extra dimensions for the Potree viewer.

PotreeConverter surfaces LAS extra dimensions as switchable attributes in the
browser, giving the "RGB <-> prediction <-> ground-truth <-> uncertainty" toggle
with zero front-end code.

Given a Pointcept scene (coord/color) plus prediction/gt/entropy arrays, this
produces a single LAS 1.4 file carrying:
    pred_class  (float32)   predicted semantic class
    gt_class    (float32)   ground-truth class (-1 where unknown)
    entropy     (float32)   per-point predictive entropy
plus RGB if available.

Example (from Python):
    export_scalars("data/processed/dales/scene", "outputs/scene_viz.las",
                   pred=pred, gt=gt, entropy=ent)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mpcseg.common.io import PointCloud, write_las


def export_scalars(
    scene_dir: str,
    out_las: str,
    pred: np.ndarray,
    gt: np.ndarray | None = None,
    entropy: np.ndarray | None = None,
) -> Path:
    scene = Path(scene_dir)
    coord = np.load(scene / "coord.npy").astype(np.float64)
    color = None
    if (scene / "color.npy").exists():
        color = np.load(scene / "color.npy").astype(np.float32)

    extra = {"pred_class": pred.astype(np.float32)}
    if gt is not None:
        extra["gt_class"] = gt.astype(np.float32)
    if entropy is not None:
        extra["entropy"] = entropy.astype(np.float32)

    pc = PointCloud(coord=coord, color=color, extra=extra)
    write_las(out_las, pc)
    print(f"Wrote {len(pc):,} pts with fields {list(extra)} -> {out_las}")
    return Path(out_las)
