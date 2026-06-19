"""Convert a LAS/LAZ file into a Potree static page via PotreeConverter.

PotreeConverter 2.x emits 3 files (octree.bin, metadata.json, hierarchy.bin) plus,
with --generate-page, a ready-to-serve HTML page. Serve over http:// (browsers block
file://), e.g.  python -m http.server  in the output dir.

PotreeConverter is a separate prebuilt binary (Windows: needs the VS2015+ x64
redistributable). Point this script at it via --converter or the POTREE_CONVERTER
env var. Download: https://github.com/potree/PotreeConverter/releases  (>= 2.1.2)

Example:
    python -m mpcseg.viz.make_potree --in outputs/scene_viz.las \
        --out web/pointclouds/scene --converter C:/tools/PotreeConverter/PotreeConverter.exe
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def find_converter(explicit: str | None) -> str:
    cand = explicit or os.environ.get("POTREE_CONVERTER") or shutil.which("PotreeConverter")
    if not cand or not Path(cand).exists():
        raise FileNotFoundError(
            "PotreeConverter not found. Pass --converter or set POTREE_CONVERTER. "
            "Download >= 2.1.2 from github.com/potree/PotreeConverter/releases"
        )
    return cand


def make_potree(in_las: str, out_dir: str, converter: str | None = None,
                page_name: str = "demo") -> Path:
    exe = find_converter(converter)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [exe, str(in_las), "-o", str(out), "--generate-page", page_name]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Potree page generated in {out} (serve over http://, e.g. python -m http.server)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_las", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--converter")
    ap.add_argument("--page-name", default="demo")
    args = ap.parse_args()
    make_potree(args.in_las, args.out, args.converter, args.page_name)


if __name__ == "__main__":
    main()
