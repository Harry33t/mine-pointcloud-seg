"""Helper for obtaining a McKinley Mine (open-pit coal mine) AOI from OpenTopography.

Dataset: McKinley Mine, New Mexico — surface coal mine LiDAR, 46.15 pts/m^2,
CC BY 4.0, collection OT.112024.6341.2.
  https://portal.opentopography.org/datasetMetadata?otCollectionID=OT.112024.6341.2

IMPORTANT: the points carry only standard ASPRS classification (ground /
unclassified / water / ...), NOT semantic mine features (highwall, bench, spoil,
haul road). Treat this as raw, effectively-unlabelled mine data — which is exactly
the motivation for the self-supervised / weak-label approach.

Access options
--------------
A) Web clip (no coding): use the portal's "Point Cloud" clip-to-AOI tool, draw a
   small box (~0.25-1 km^2: one highwall + bench + haul road), download the LAZ,
   then crop/tile with ``mpcseg.data.crop_aoi``.

B) Bulk tiles: create a free MyOpenTopo account, grab the tile-index shapefile from
   the dataset page, and download only the tiles intersecting your AOI. Paste the
   per-tile URL(s) below to fetch them.

This script handles (B) given a direct tile URL, and prints guidance otherwise.
"""
from __future__ import annotations

import argparse
from pathlib import Path

COLLECTION_ID = "OT.112024.6341.2"
METADATA_URL = (
    "https://portal.opentopography.org/datasetMetadata?otCollectionID=" + COLLECTION_ID
)


def _download(url: str, out: Path) -> Path:
    import requests
    from tqdm import tqdm

    out.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    print(f"Downloaded -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tile-url", help="direct LAZ tile URL from the OT tile index")
    ap.add_argument("--dest", default="data/raw/mckinley")
    args = ap.parse_args()

    if not args.tile_url:
        print("McKinley Mine on OpenTopography:")
        print(f"  {METADATA_URL}")
        print()
        print("No --tile-url given. Easiest path:")
        print("  1. Open the page above, use the Point Cloud clip tool, draw a small")
        print("     AOI (~0.25-1 km^2), download the LAZ into", args.dest)
        print("  2. Crop/tile it:  python -m mpcseg.data.crop_aoi --in <file>.laz ...")
        print("  3. Convert:       python -m mpcseg.data.to_pointcept --no-label ...")
        return

    name = args.tile_url.split("/")[-1] or "mckinley_tile.laz"
    _download(args.tile_url, Path(args.dest) / name)


if __name__ == "__main__":
    main()
