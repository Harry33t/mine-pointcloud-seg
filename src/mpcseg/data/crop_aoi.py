"""Crop a large LAS/LAZ point cloud to a small AOI and/or tile it into a grid.

OpenTopography mine tiles are huge — crop a small area-of-interest first to get the
pipeline running, then scale up.

Examples
--------
Crop one AOI:
    python -m mpcseg.data.crop_aoi --in data/raw/mckinley/tile.laz \
        --out data/raw/mckinley/aoi.laz --bbox 700000 3900000 700500 3900500

Tile into a 250 m grid:
    python -m mpcseg.data.crop_aoi --in data/raw/mckinley/aoi.laz \
        --out-dir data/raw/mckinley/tiles --tile 250
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mpcseg.common.io import crop_bbox, read_las, write_las


def tile_grid(in_path: str, out_dir: str, tile: float) -> list[Path]:
    """Split a cloud into square tiles of side ``tile`` (CRS units). Returns paths."""
    pc = read_las(in_path)
    xmin, ymin = pc.coord[:, 0].min(), pc.coord[:, 1].min()
    xmax, ymax = pc.coord[:, 0].max(), pc.coord[:, 1].max()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    nx = int(np.ceil((xmax - xmin) / tile))
    ny = int(np.ceil((ymax - ymin) / tile))
    for i in range(nx):
        for j in range(ny):
            bx, by = xmin + i * tile, ymin + j * tile
            sub = crop_bbox(pc, (bx, by, bx + tile, by + tile))
            if len(sub) == 0:
                continue
            out = out_dir / f"tile_{i:03d}_{j:03d}.laz"
            write_las(out, sub)
            written.append(out)
            print(f"  {out.name}: {len(sub):,} pts")
    print(f"Wrote {len(written)} tiles to {out_dir}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", help="single cropped output (requires --bbox)")
    ap.add_argument("--out-dir", help="output dir for tiling (requires --tile)")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    ap.add_argument("--tile", type=float, help="tile side length in CRS units")
    args = ap.parse_args()

    if args.bbox and args.out:
        pc = read_las(args.in_path)
        sub = crop_bbox(pc, tuple(args.bbox))
        write_las(args.out, sub)
        print(f"Cropped {len(sub):,} pts -> {args.out}")
    elif args.tile and args.out_dir:
        tile_grid(args.in_path, args.out_dir, args.tile)
    else:
        ap.error("use either --bbox + --out, or --tile + --out-dir")


if __name__ == "__main__":
    main()
