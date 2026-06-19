"""Process a raw open-pit mine LiDAR tile into a web-viewable, geometrically
weak-labelled point cloud.

Raw mine LiDAR has no semantic labels (and often no RGB), so we segment it with
geometry alone (the project's weak-label pillar, applied to a real mine):
  * CSF cloth filter            -> floor / bench (ground)
  * height-above-ground (HAG)   -> relief field (drives a dramatic depth colour ramp)
  * eigen verticality/planarity -> highwall (vertical faces) vs slope/spoil vs rough

Output is the web viewer's binary format (see web/src/data/loadPointCloud.ts) with mine
class names and a "height above ground" scalar. Points are voxel-downsampled so a
multi-million-point mine tile stays interactive in the browser.

    python -m mpcseg.data.process_mine --in mine.laz --out web/public/data/mine \
        --voxel 1.0 --max-points 400000
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

# mine pseudo-classes (index = id)
MINE_NAMES = ["floor/bench", "highwall", "slope/spoil", "rough/veg", "water"]
MINE_COLORS = ["#c9b27a", "#e53935", "#fb8c00", "#43a047", "#1e88e5"]
FLOOR, HIGHWALL, SLOPE, ROUGH, WATER = 0, 1, 2, 3, 4


def _voxel_downsample(coord, rgb, voxel):
    import open3d as o3d

    o = o3d.geometry.PointCloud()
    o.points = o3d.utility.Vector3dVector(coord)
    if rgb is not None:
        o.colors = o3d.utility.Vector3dVector(rgb)
    o = o.voxel_down_sample(voxel)
    c = np.asarray(o.points)
    col = np.asarray(o.colors) if rgb is not None and len(o.colors) else None
    return c, col


def assign_mine_classes(coord, search_radius=3.0, hag_veg=2.0):
    from mpcseg.weaklabels.geometric_pseudolabels import (
        eigen_features, ground_mask_csf, height_above_ground,
    )

    gmask = ground_mask_csf(coord, cloth_resolution=1.0)
    hag = height_above_ground(coord, gmask, cell=2.0)
    feats = eigen_features(coord, search_radius=search_radius)
    vert, plan, sph = feats["verticality"], feats["planarity"], feats["sphericity"]

    labels = np.full(len(coord), SLOPE, dtype=np.int64)
    labels[gmask] = FLOOR
    above = ~gmask
    # highwall: steep vertical, planar faces standing above the floor
    labels[above & (vert > 0.6) & (plan > 0.4)] = HIGHWALL
    # rough / vegetation: scattered points well above ground
    labels[above & (sph > 0.25) & (hag > hag_veg)] = ROUGH
    return labels, hag, gmask


def write_web(out_base, pos, scalar, rgb, cls, names, colors, scalar_label):
    n = len(pos)
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    with open(out_base + ".bin", "wb") as f:
        f.write(pos.astype(np.float32).tobytes())           # Float32[N*3]
        f.write(scalar.astype(np.float32).tobytes())        # Float32[N]
        f.write(rgb.astype(np.uint8).tobytes())             # Uint8[N*3]
        f.write(np.clip(cls, 0, 254).astype(np.uint8).tobytes())  # Uint8[N]
        f.write(np.full(n, 255, np.uint8).tobytes())        # Uint8[N] gt = unknown
    meta = {
        "numPoints": int(n),
        "classNames": names,
        "classColors": colors,
        "hasColor": bool(rgb.any()),
        "hasGt": False,
        "scalarLabel": scalar_label,
        "bounds": {"min": pos.min(0).tolist(), "max": pos.max(0).tolist()},
    }
    with open(out_base + ".meta.json", "w") as f:
        json.dump(meta, f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_laz", required=True)
    ap.add_argument("--out", required=True, help="output base (no extension)")
    ap.add_argument("--voxel", type=float, default=1.0, help="voxel downsample size (m)")
    ap.add_argument("--max-points", type=int, default=400000)
    ap.add_argument("--search-radius", type=float, default=3.0,
                    help="eigen-feature neighbourhood radius in metres")
    args = ap.parse_args()

    from mpcseg.common.io import read_las

    pc = read_las(args.in_laz)
    print(f"read {len(pc):,} pts; rgb={pc.color is not None}")
    coord = pc.coord
    rgb = pc.color

    c, col = _voxel_downsample(coord, rgb, args.voxel)
    if len(c) > args.max_points:
        sel = np.random.default_rng(0).choice(len(c), args.max_points, replace=False)
        c = c[sel]
        col = None if col is None else col[sel]
    print(f"downsampled to {len(c):,} pts (voxel {args.voxel} m)")

    cls, hag, gmask = assign_mine_classes(c, search_radius=args.search_radius)
    uniq, cnt = np.unique(cls, return_counts=True)
    print("class hist:", {MINE_NAMES[int(u)]: int(n) for u, n in zip(uniq, cnt)})

    # three.js is y-up: map (x, y, z) -> (x, z, y); center XY
    pos = np.stack([c[:, 0] - c[:, 0].mean(), c[:, 2], c[:, 1] - c[:, 1].mean()], axis=1)
    rgb_u8 = (np.clip(col, 0, 1) * 255).astype(np.uint8) if col is not None \
        else np.full((len(c), 3), 150, np.uint8)

    write_web(args.out, pos, hag, rgb_u8, cls, MINE_NAMES, MINE_COLORS,
              "height above ground (m)")
    print(f"wrote {args.out}.bin (+ .meta.json) — {len(c):,} pts")


if __name__ == "__main__":
    main()
