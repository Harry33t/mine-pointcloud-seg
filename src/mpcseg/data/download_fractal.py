"""Download a subset of the FRACTAL aerial-LiDAR semantic-segmentation dataset.

FRACTAL (IGN France) is fully open on the Hugging Face Hub — no form, no email. It
is ultra-large (100k patches of 50x50 m, 81.6 GB total), so this script pulls only a
few shards. One shard (~0.9-1 GB) extracts to a few hundred LAS patches — plenty to
validate the train -> evaluate -> Potree pipeline end to end.

Repo: https://huggingface.co/datasets/IGNF/FRACTAL
Layout: data/{train,val,test}/{split}-NN.zip  (each zip holds LAS patches)
Classes (7): 0 other | 1 ground | 2 vegetation | 3 building | 4 water | 5 bridge |
             6 permanent structure. Channels: near-infrared, red, green, blue.
License: open (etalab / CC-BY style; check the dataset card for the exact terms).

Behind a restrictive network? Set a mirror before running:
    # bash
    export HF_ENDPOINT=https://hf-mirror.com
    # PowerShell
    $env:HF_ENDPOINT = "https://hf-mirror.com"

Examples:
    # one test shard, download + extract to data/raw/fractal
    python -m mpcseg.data.download_fractal --split test --shards 0 --extract

    # first two val shards, download only (no extract)
    python -m mpcseg.data.download_fractal --split val --shards 0 1
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

REPO_ID = "IGNF/FRACTAL"
SPLITS = ("train", "val", "test")
FRACTAL_CLASSES = [
    "other", "ground", "vegetation", "building", "water", "bridge", "permanent_structure",
]


def download_shards(split: str, shards: list[int], dest: str) -> list[Path]:
    """Download given shard zips for a split. Returns local zip paths."""
    from huggingface_hub import hf_hub_download

    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for n in shards:
        filename = f"data/{split}/{split}-{n:02d}.zip"
        print(f"Downloading {REPO_ID}:{filename} ...")
        local = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            local_dir=str(dest),
        )
        paths.append(Path(local))
        print(f"  -> {local}")
    return paths


def extract(zip_paths: list[Path], out_dir: str) -> None:
    """Extract shard zips (LAS patches) into out_dir/<split>/."""
    for zp in zip_paths:
        split = zp.parent.name  # data/<split>/<file>.zip
        target = Path(out_dir) / split
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp) as z:
            z.extractall(target)
        n = len(list(target.glob("*.las"))) + len(list(target.glob("*.laz")))
        print(f"Extracted {zp.name} -> {target}  ({n} point-cloud files so far)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=SPLITS)
    ap.add_argument("--shards", type=int, nargs="+", default=[0],
                    help="shard indices to fetch, e.g. 0 1 2")
    ap.add_argument("--dest", default="data/raw/fractal")
    ap.add_argument("--extract", action="store_true",
                    help="unzip shards into <dest>/extracted/<split>/")
    args = ap.parse_args()

    zips = download_shards(args.split, args.shards, args.dest)
    if args.extract:
        extract(zips, str(Path(args.dest) / "extracted"))
        print("\nNext: convert a patch to Pointcept format, e.g.")
        print("  python -m mpcseg.data.to_pointcept \\")
        print(f"    --in data/raw/fractal/extracted/{args.split}/<patch>.las \\")
        print("    --out-root data/processed/fractal --scene <patch>")


if __name__ == "__main__":
    main()
