"""Turn a mine DEM GeoTIFF (+ optional orthophoto) into a web-viewable point cloud.

Builds one point per DEM cell at its real (UTM) location, classifies cells by slope
(floor/bench vs slope/spoil vs steep highwall), drapes orthophoto RGB, bakes a
hillshade so the terrain reads as solid 3D in the browser, computes headline stats
(relief, cut volume, class mix) and a few 3D annotations (pit floor / rim / highwall).

    python -m mpcseg.data.process_dem --in output_be.tif --ortho output_op.tif \
        --out web/public/data/mine --vexag 1.5 --max-points 700000
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

NAMES = ["floor / bench", "slope / spoil", "steep highwall"]
COLORS = ["#c9b27a", "#fb8c00", "#e53935"]


def _read_dem(path):
    from osgeo import gdal

    gdal.UseExceptions()
    ds = gdal.Open(path)
    b = ds.GetRasterBand(1)
    z = b.ReadAsArray().astype(np.float64)
    nod = b.GetNoDataValue()
    if nod is not None:
        z[z == nod] = np.nan
    gt = ds.GetGeoTransform()
    projected = abs(gt[1]) > 0.01  # pixel-size heuristic (no PROJ dependency)
    return z, gt, projected


def _sample_ortho(path, X, Y):
    from osgeo import gdal

    gdal.UseExceptions()
    ds = gdal.Open(path)
    gt = ds.GetGeoTransform()
    arr = ds.ReadAsArray()
    if arr.ndim == 2:
        arr = arr[None]
    _, H, W = arr.shape
    col = np.clip(((X - gt[0]) / gt[1]).astype(int), 0, W - 1)
    row = np.clip(((Y - gt[3]) / gt[5]).astype(int), 0, H - 1)
    rgb = np.stack([arr[b][row, col] for b in range(3)], axis=1).astype(np.float64)
    if rgb.max() > 255:
        rgb = rgb / rgb.max() * 255
    return rgb.astype(np.uint8)


def _hillshade(z, dx, dy, vexag, az=315.0, alt=45.0):
    """Lambert hillshade 0..1 (NaN-safe)."""
    zf = np.where(np.isnan(z), np.nanmin(z), z) * vexag
    dzdy, dzdx = np.gradient(zf, dy, dx)
    slope = np.pi / 2 - np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(-dzdx, dzdy)
    azr, altr = np.radians(360 - az + 90), np.radians(alt)
    hs = np.sin(altr) * np.sin(slope) + np.cos(altr) * np.cos(slope) * np.cos(azr - aspect)
    return np.clip(hs, 0, 1)


def process(in_tif, out_base, ortho=None, vexag=1.5, max_points=700000):
    z, gt, projected = _read_dem(in_tif)
    H, W = z.shape
    if projected:
        dx_m, dy_m = abs(gt[1]), abs(gt[5])
    else:
        lat0 = gt[3] + gt[5] * H / 2
        dx_m = abs(gt[1]) * 111320.0 * np.cos(np.radians(lat0))
        dy_m = abs(gt[5]) * 110540.0
    relief = float(np.nanmax(z) - np.nanmin(z))
    print(f"DEM {W}x{H} | {'projected' if projected else 'geographic'} | pixel {dx_m:.2f} m "
          f"| relief {relief:.1f} m")

    dzdy, dzdx = np.gradient(z, dy_m, dx_m)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    shade = _hillshade(z, dx_m, dy_m, vexag)

    cc, rr = np.meshgrid(np.arange(W), np.arange(H))
    Xg = gt[0] + (cc + 0.5) * gt[1]
    Yg = gt[3] + (rr + 0.5) * gt[5]

    valid = ~np.isnan(z)
    X, Y, zz, sl, sh = Xg[valid], Yg[valid], z[valid], slope[valid], shade[valid]

    # cut volume below the rim (90th-pct elevation), over valid cells
    z_rim = float(np.nanpercentile(zz, 90))
    cell_area = dx_m * dy_m
    volume = float(np.clip(z_rim - zz, 0, None).sum() * cell_area)

    if len(X) > max_points:
        sel = np.random.default_rng(0).choice(len(X), max_points, replace=False)
        X, Y, zz, sl, sh = X[sel], Y[sel], zz[sel], sl[sel], sh[sel]

    cls = np.zeros(len(X), np.int64)
    cls[sl > 12] = 1
    cls[sl > 28] = 2
    class_pct = [round(float((cls == i).mean() * 100), 1) for i in range(3)]

    rgb = _sample_ortho(ortho, X, Y) if ortho else np.full((len(X), 3), 150, np.uint8)
    zmin = float(zz.min())
    relief_pt = (zz - zmin).astype(np.float32)
    xc, yc = X.mean(), Y.mean()
    pos = np.stack([X - xc, (zz - zmin) * vexag, Y - yc], axis=1).astype(np.float32)

    # 3D annotations (in display coords)
    def disp(i):
        return [float(X[i] - xc), float((zz[i] - zmin) * vexag), float(Y[i] - yc)]
    ann = [
        {"label": "pit floor", "pos": disp(int(np.argmin(zz)))},
        {"label": "rim", "pos": disp(int(np.argmax(zz)))},
        {"label": f"highwall · {relief:.0f} m relief", "pos": disp(int(np.argmax(sl)))},
    ]

    n = len(X)
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    with open(out_base + ".bin", "wb") as f:
        f.write(pos.tobytes())                    # Float32[N*3]
        f.write(relief_pt.tobytes())              # Float32[N]
        f.write(sh.astype(np.float32).tobytes())  # Float32[N] shade
        f.write(rgb.astype(np.uint8).tobytes())   # Uint8[N*3]
        f.write(np.clip(cls, 0, 254).astype(np.uint8).tobytes())
        f.write(np.full(n, 255, np.uint8).tobytes())
    meta = {
        "numPoints": int(n),
        "classNames": NAMES,
        "classColors": COLORS,
        "hasColor": bool(ortho),
        "hasGt": False,
        "hasShade": True,
        "classLabel": "Geometry",
        "scalarLabel": "relief — elevation above pit floor (m)",
        "scalarShort": "Relief",
        "stats": {
            "Relief": f"{relief:.0f} m",
            "AOI": f"{W * dx_m:.0f} × {H * dy_m:.0f} m",
            "Resolution": f"{dx_m:.1f} m DEM",
            "Cut volume": f"≈ {volume / 1e6:.1f} M m³ below rim",
            "Floor / Slope / Highwall": f"{class_pct[0]} / {class_pct[1]} / {class_pct[2]} %",
        },
        "annotations": ann,
        "bounds": {"min": pos.min(0).tolist(), "max": pos.max(0).tolist()},
    }
    with open(out_base + ".meta.json", "w") as f:
        json.dump(meta, f)
    print(f"wrote {n:,} pts -> {out_base}.bin | relief {relief:.0f} m | "
          f"volume {volume/1e6:.1f}M m3 | classes {class_pct} %")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_tif", required=True)
    ap.add_argument("--ortho", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vexag", type=float, default=1.5)
    ap.add_argument("--max-points", type=int, default=700000)
    args = ap.parse_args()
    process(args.in_tif, args.out, args.ortho, args.vexag, args.max_points)


if __name__ == "__main__":
    main()
