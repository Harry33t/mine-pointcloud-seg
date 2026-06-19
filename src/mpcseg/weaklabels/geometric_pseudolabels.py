"""Geometric weak/pseudo labels from per-point features (no manual annotation).

Recipe (cheapest defensible path for outdoor/mine ALS):
  1. ground / non-ground split  -> CSF cloth simulation
  2. height-above-ground (HAG)  -> z minus local ground height
  3. eigen-features (planarity, linearity, sphericity, verticality) -> jakteristics
  4. threshold features into coarse pseudo-classes
  5. (optional) propagate a few manual seed points to homogeneous clusters
  6. self-training: train on pseudo-labels, keep high-confidence preds, iterate

These pseudo-labels are reliable for COARSE classes (ground / vegetation / planar
structures / linear features); they are weak on fine semantics — budget human seeds
for hard classes. Feature definitions follow Hackel et al. 2016.

Status: scaffold — steps 1-4 implemented against jakteristics + CSF; step 5/6 are
stubs to wire once the supervised loop exists.
"""
from __future__ import annotations

import numpy as np

from mpcseg.common.io import PointCloud

# coarse pseudo-class ids (keep aligned with the training label map)
GROUND, VEGETATION, PLANAR_STRUCT, LINEAR, UNKNOWN = 0, 1, 2, 3, -1


def ground_mask_csf(coord: np.ndarray, cloth_resolution: float = 1.0) -> np.ndarray:
    """Cloth-simulation ground filter. Returns boolean mask (True = ground).

    Requires ``cloth-simulation-filter`` (import name ``CSF``).
    """
    import CSF

    csf = CSF.CSF()
    csf.params.cloth_resolution = cloth_resolution
    csf.params.bSloopSmooth = True  # better on steep mine highwalls/benches
    csf.setPointCloud(coord.astype(np.float64))
    ground_idx, _ = CSF.VecInt(), CSF.VecInt()
    csf.do_filtering(ground_idx, _)
    mask = np.zeros(len(coord), dtype=bool)
    mask[np.asarray(ground_idx, dtype=int)] = True
    return mask


def height_above_ground(coord: np.ndarray, ground_mask: np.ndarray,
                        cell: float = 2.0) -> np.ndarray:
    """Per-point height above a rasterised ground surface (simple min-z grid)."""
    gx = np.floor(coord[:, 0] / cell).astype(np.int64)
    gy = np.floor(coord[:, 1] / cell).astype(np.int64)
    keys = gx * 1_000_003 + gy
    ground_z: dict[int, float] = {}
    for k, z in zip(keys[ground_mask], coord[ground_mask, 2]):
        if k not in ground_z or z < ground_z[k]:
            ground_z[k] = z
    base = np.array([ground_z.get(k, np.nan) for k in keys])
    # fill cells with no ground point using the global ground median
    base[np.isnan(base)] = np.nanmedian(coord[ground_mask, 2]) if ground_mask.any() else 0.0
    return coord[:, 2] - base


def eigen_features(coord: np.ndarray, search_radius: float = 3.0) -> dict[str, np.ndarray]:
    """Per-point eigen-features via jakteristics (radius neighbourhood search).

    search_radius is in coordinate units (metres); pick a few × the point spacing.
    """
    import jakteristics as jak

    names = ["planarity", "linearity", "sphericity", "verticality"]
    feats = jak.compute_features(coord.astype(np.float64), search_radius=search_radius,
                                 feature_names=names)
    return {n: feats[:, i] for i, n in enumerate(names)}


def assign_pseudo_labels(pc: PointCloud, hag_veg: float = 0.5) -> np.ndarray:
    """Threshold geometric features into coarse pseudo-classes. Returns (N,) int."""
    gmask = ground_mask_csf(pc.coord)
    hag = height_above_ground(pc.coord, gmask)
    feats = eigen_features(pc.coord)

    labels = np.full(len(pc), UNKNOWN, dtype=np.int64)
    labels[gmask] = GROUND
    above = ~gmask
    labels[above & (feats["sphericity"] > 0.25) & (hag > hag_veg)] = VEGETATION
    labels[above & (feats["planarity"] > 0.6)] = PLANAR_STRUCT
    labels[above & (feats["linearity"] > 0.6)] = LINEAR
    return labels


# --- TODO: step 5/6 -------------------------------------------------------------
def propagate_seeds(labels, seeds, coord, k=10):  # noqa: D401 - stub
    """Propagate a few manual seed labels to homogeneous kNN clusters. TODO."""
    raise NotImplementedError("wire after the supervised loop exists")


def self_train_round(model, scenes, conf_threshold=0.9):  # noqa: D401 - stub
    """One self-training round: predict, keep high-confidence, add to train set. TODO."""
    raise NotImplementedError("wire after the supervised loop exists")
