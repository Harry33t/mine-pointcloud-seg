"""Convert LAS/LAZ/PLY point clouds to the Pointcept on-disk format.

Pointcept expects one directory per scene, each holding separate ``.npy`` arrays:

    <out_root>/<scene>/
        coord.npy     (N, 3) float32   XYZ            (required)
        strength.npy  (N, 1) float32   intensity      (outdoor/ALS)
        color.npy     (N, 3) float32   RGB in [0,1]   (optional)
        normal.npy    (N, 3) float32                  (optional)
        segment.npy   (N,)   int64     semantic label (required for training)

For raw mine data with no semantic labels, ``--no-label`` writes an all-(-1)
``segment.npy`` (ignore index) so the scene can still be used for inference /
self-supervised pretraining.

Example:
    python -m mpcseg.data.to_pointcept --in data/raw/dales/5080_54435.ply \
        --out-root data/processed/dales --scene 5080_54435
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mpcseg.common.io import PointCloud, estimate_normals, read_las

IGNORE_INDEX = -1

# FRACTAL ships raw IGN Lidar HD ASPRS-style class codes; remap to the published
# 7-class benchmark: 0 other | 1 ground | 2 vegetation | 3 building | 4 water |
# 5 bridge | 6 permanent_structure. (3/4/5 low/med/high veg -> single vegetation.)
FRACTAL_REMAP = {1: 0, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 9: 4, 17: 5, 64: 6}
PRESETS = {"fractal": FRACTAL_REMAP}


def _read_ply(path: str) -> PointCloud:
    """Read a PLY (DALES distributes .ply with a per-point label property)."""
    from plyfile import PlyData  # part of the Pointcept/plyfile stack

    ply = PlyData.read(path)
    v = ply["vertex"].data
    coord = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    names = v.dtype.names
    label = None
    for key in ("sem_class", "class", "label", "scalar_Label"):
        if key in names:
            label = np.asarray(v[key], dtype=np.int32)
            break
    intensity = np.asarray(v["intensity"], dtype=np.float32) if "intensity" in names else None
    return PointCloud(coord=coord, intensity=intensity, label=label)


def convert(
    in_path: str,
    out_root: str,
    scene: str,
    with_normals: bool = False,
    normal_radius: float = 1.0,
    no_label: bool = False,
    label_remap: dict[int, int] | None = None,
) -> Path:
    in_path = str(in_path)
    pc = _read_ply(in_path) if in_path.lower().endswith(".ply") else read_las(in_path)

    out_dir = Path(out_root) / scene
    out_dir.mkdir(parents=True, exist_ok=True)

    # center XY to keep float32 precision; keep Z absolute (height matters)
    coord = pc.coord.copy()
    coord[:, :2] -= coord[:, :2].mean(axis=0)
    np.save(out_dir / "coord.npy", coord.astype(np.float32))

    if pc.intensity is not None:
        inten = pc.intensity.astype(np.float32)
        # normalise intensity to ~[0,1]; helps cross-site transfer
        if inten.max() > 1.0:
            inten = inten / max(inten.max(), 1.0)
        np.save(out_dir / "strength.npy", inten.reshape(-1, 1))

    if pc.color is not None:
        np.save(out_dir / "color.npy", pc.color.astype(np.float32))

    if with_normals:
        np.save(out_dir / "normal.npy", estimate_normals(pc, radius=normal_radius))

    if no_label or pc.label is None:
        segment = np.full(len(pc), IGNORE_INDEX, dtype=np.int64)
    else:
        segment = pc.label.astype(np.int64)
        if label_remap:
            remapped = np.full_like(segment, IGNORE_INDEX)
            for src, dst in label_remap.items():
                remapped[segment == src] = dst
            segment = remapped
    np.save(out_dir / "segment.npy", segment)

    print(f"{scene}: {len(pc):,} pts -> {out_dir}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--with-normals", action="store_true")
    ap.add_argument("--normal-radius", type=float, default=1.0)
    ap.add_argument("--no-label", action="store_true",
                    help="write ignore-index labels (raw/unlabelled data)")
    ap.add_argument("--preset", choices=sorted(PRESETS),
                    help="apply a known class remap (e.g. fractal)")
    args = ap.parse_args()
    convert(
        args.in_path, args.out_root, args.scene,
        with_normals=args.with_normals, normal_radius=args.normal_radius,
        no_label=args.no_label, label_remap=PRESETS.get(args.preset),
    )


if __name__ == "__main__":
    main()
