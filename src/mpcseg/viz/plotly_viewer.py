"""Self-contained interactive 3D viewer (single HTML) from a LAS with scalar fields.

Reads a LAS written by mpcseg (coord + RGB + pred_class / gt_class / entropy extra
dims), downsamples for responsiveness, and emits one offline HTML with buttons to
switch the point colouring between RGB / prediction / ground-truth / uncertainty.

This is the lowest-effort interactive viewer (no server, no PotreeConverter) — open
the HTML in any browser. For the polished, million-point version use Potree
(mpcseg.viz.make_potree).

Example:
    python -m mpcseg.viz.plotly_viewer --in scene.las --out scene.html --max-points 60000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# 7-class FRACTAL palette (index = class id)
CLASS_COLORS = [
    "#9e9e9e",  # 0 other
    "#8d6e63",  # 1 ground
    "#43a047",  # 2 vegetation
    "#e53935",  # 3 building
    "#1e88e5",  # 4 water
    "#fb8c00",  # 5 bridge
    "#8e24aa",  # 6 permanent_structure
]
CLASS_NAMES = ["other", "ground", "vegetation", "building", "water", "bridge", "permanent"]


def _read(path: str):
    import laspy

    las = laspy.read(path)
    coord = np.stack([las.x, las.y, las.z], axis=1).astype(np.float32)
    dims = set(las.point_format.dimension_names)
    rgb = None
    if {"red", "green", "blue"} <= dims:
        rgb = np.stack([las.red, las.green, las.blue], axis=1).astype(np.float32)
        rgb = rgb / (65535.0 if rgb.max() > 255 else 255.0)
    get = lambda n: np.asarray(las[n]) if n in dims else None
    return coord, rgb, get("pred_class"), get("gt_class"), get("entropy")


def _class_color_strings(labels: np.ndarray) -> list[str]:
    lab = labels.astype(int)
    return [CLASS_COLORS[c] if 0 <= c < len(CLASS_COLORS) else "#000000" for c in lab]


def build_html(in_las: str, out_html: str, max_points: int = 60000, seed: int = 0) -> Path:
    import plotly.graph_objects as go

    coord, rgb, pred, gt, ent = _read(in_las)
    n = len(coord)
    if n > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(n, size=max_points, replace=False)
        coord = coord[keep]
        rgb = None if rgb is None else rgb[keep]
        pred = None if pred is None else pred[keep]
        gt = None if gt is None else gt[keep]
        ent = None if ent is None else ent[keep]

    x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]
    base = dict(x=x, y=y, z=z, mode="markers", type="scatter3d")
    marker = lambda **kw: dict(size=1.4, **kw)

    traces, buttons = [], []

    def add(name, color, is_rgb=False):
        if color is None:
            return
        if is_rgb:
            cols = ["rgb(%d,%d,%d)" % tuple((c * 255).astype(int)) for c in color]
            tr = go.Scatter3d(**base, marker=marker(color=cols), name=name)
        elif name in ("prediction", "ground-truth"):
            tr = go.Scatter3d(**base, marker=marker(color=_class_color_strings(color)),
                              name=name)
        else:  # entropy (continuous)
            tr = go.Scatter3d(**base, marker=marker(
                color=color, colorscale="Viridis", showscale=True,
                colorbar=dict(title="entropy")), name=name)
        traces.append(tr)

    add("RGB", rgb, is_rgb=True)
    add("prediction", pred)
    add("ground-truth", gt)
    add("uncertainty", ent)

    # visibility toggle buttons
    for i, tr in enumerate(traces):
        vis = [j == i for j in range(len(traces))]
        buttons.append(dict(label=tr.name, method="update",
                            args=[{"visible": vis}, {"title": f"colour: {tr.name}"}]))
    for i, tr in enumerate(traces):
        tr.visible = (i == 0)

    fig = go.Figure(data=traces)
    fig.update_layout(
        updatemenus=[dict(buttons=buttons, direction="right", x=0.5, xanchor="center",
                          y=1.08, yanchor="top", showactive=True)],
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
        title=f"colour: {traces[0].name if traces else ''}",
    )
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs=True)  # self-contained, works offline
    print(f"Wrote viewer ({len(x):,} pts) -> {out_html}")
    return Path(out_html)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_las", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-points", type=int, default=60000)
    args = ap.parse_args()
    build_html(args.in_las, args.out, args.max_points)


if __name__ == "__main__":
    main()
