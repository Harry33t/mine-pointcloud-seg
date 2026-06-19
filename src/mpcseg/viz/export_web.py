"""Export a LAS (with pred/gt/entropy) to the web viewer's binary format.

Writes <out>.bin and <out>.meta.json consumed by web/src/data/loadPointCloud.ts.
Binary layout (little-endian, concatenated). Float32 sections come first so their
byte offsets stay 4-byte aligned for any N:
    position Float32[N*3] | entropy Float32[N] | rgb Uint8[N*3] | pred Uint8[N] | gt Uint8[N]
Coordinates are remapped LAS (x, y, z) -> three.js (x, z, y) so height is up.
gt 255 = unknown/ignore.

Example:
    python -m mpcseg.viz.export_web --in outputs/scene_viz.las \
        --out web/public/data/scene --max-points 300000
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

CLASS_NAMES = ["other", "ground", "vegetation", "building", "water", "bridge", "permanent"]
CLASS_COLORS = ["#9e9e9e", "#8d6e63", "#43a047", "#e53935", "#1e88e5", "#fb8c00", "#8e24aa"]


def export(in_las: str, out_base: str, max_points: int = 300000, seed: int = 0) -> None:
    import laspy

    las = laspy.read(in_las)
    n = len(las.x)
    coord = np.stack([las.x, las.y, las.z], axis=1).astype(np.float32)

    dims = set(las.point_format.dimension_names)
    has_color = {"red", "green", "blue"} <= dims
    if has_color:
        rgb = np.stack([las.red, las.green, las.blue], axis=1).astype(np.float32)
        rgb = (rgb / (65535.0 if rgb.max() > 255 else 255.0) * 255).astype(np.uint8)
    else:
        rgb = np.full((n, 3), 180, np.uint8)

    get = lambda k: np.asarray(las[k]) if k in dims else None
    pred = get("pred_class")
    gt = get("gt_class")
    ent = get("entropy")
    pred = np.zeros(n) if pred is None else pred
    has_gt = gt is not None
    gt = np.full(n, 255) if gt is None else np.where(gt < 0, 255, gt)
    ent = np.zeros(n, np.float32) if ent is None else ent.astype(np.float32)

    if n > max_points:
        rng = np.random.default_rng(seed)
        sel = rng.choice(n, size=max_points, replace=False)
        coord, rgb, pred, gt, ent = coord[sel], rgb[sel], pred[sel], gt[sel], ent[sel]
        n = max_points

    # LAS (x, y, z) -> three.js (x, z, y)
    pos = np.stack([coord[:, 0], coord[:, 2], coord[:, 1]], axis=1).astype(np.float32)

    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    with open(out_base + ".bin", "wb") as f:
        f.write(pos.tobytes())                                    # Float32[N*3]
        f.write(ent.astype(np.float32).tobytes())                 # Float32[N]
        f.write(rgb.astype(np.uint8).tobytes())                   # Uint8[N*3]
        f.write(np.clip(pred, 0, 254).astype(np.uint8).tobytes()) # Uint8[N]
        f.write(np.clip(gt, 0, 255).astype(np.uint8).tobytes())   # Uint8[N]

    meta = {
        "numPoints": int(n),
        "classNames": CLASS_NAMES,
        "classColors": CLASS_COLORS,
        "hasColor": bool(has_color),
        "hasGt": bool(has_gt),
        "bounds": {
            "min": pos.min(axis=0).tolist(),
            "max": pos.max(axis=0).tolist(),
        },
    }
    with open(out_base + ".meta.json", "w") as f:
        json.dump(meta, f)
    print(f"Wrote {n:,} pts -> {out_base}.bin (+ .meta.json)  hasGt={has_gt}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_las", required=True)
    ap.add_argument("--out", required=True, help="output base path (no extension)")
    ap.add_argument("--max-points", type=int, default=300000)
    args = ap.parse_args()
    export(args.in_las, args.out, args.max_points)


if __name__ == "__main__":
    main()
