"""Fetch + extract the DALES aerial-LiDAR benchmark.

DALES requires accepting a download form, so it can't be fully scripted. Workflow:

  1. Request access + get the download link from the official page:
       https://go.udayton.edu/dales3d
     (project page: https://sites.google.com/a/udayton.edu/vasari1/research/earth-vision/dales)
  2. Either let this script download from the URL you were given, or point it at the
     archive you already downloaded.

DALES facts (for reference): aerial LiDAR, 8 classes (ground, vegetation, cars,
trucks, poles, power lines, fences, buildings), ~505 M labelled points, 40 tiles of
0.5 km (29 train / 11 test). Annotated data licensed CC BY-NC 3.0.

Examples:
    python -m mpcseg.data.download_dales --url "<link-from-form>" --dest data/raw/dales
    python -m mpcseg.data.download_dales --archive ~/Downloads/DALESObjects.tar.gz --dest data/raw/dales
"""
from __future__ import annotations

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path

DALES_INFO_URL = "https://go.udayton.edu/dales3d"


def _download(url: str, out: Path) -> Path:
    import requests
    from tqdm import tqdm

    out.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    return out


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    elif archive.name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive) as t:
            t.extractall(dest)
    else:
        raise ValueError(f"unknown archive type: {archive.name}")
    print(f"Extracted to {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", help="download link obtained from the DALES form")
    ap.add_argument("--archive", help="path to an already-downloaded DALES archive")
    ap.add_argument("--dest", default="data/raw/dales")
    ap.add_argument("--keep-archive", action="store_true")
    args = ap.parse_args()

    dest = Path(args.dest)
    if not args.url and not args.archive:
        print("DALES needs a form-gated link. Request access here:")
        print(f"  {DALES_INFO_URL}")
        print("Then re-run with --url <link>  or  --archive <file>.")
        return

    if args.archive:
        archive = Path(args.archive)
    else:
        archive = dest / "dales_download"
        _download(args.url, archive)

    _extract(archive, dest)
    if args.url and not args.keep_archive:
        archive.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
