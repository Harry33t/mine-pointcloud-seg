"""[B2] Leave-one-site-out (LOCO) cross-site transfer matrix.

Train on site A, test on site B for every (A, B) pair -> a site x site mIoU heatmap.
Rows = train site, columns = test site; diagonal = in-domain (oracle), off-diagonal =
transfer. Use closed-label scoring (shared classes per pair) so drops reflect real
failure, not ontology mismatch. Report per-class IoU alongside mIoU.

Reality check (from the methods research): if all sites share one sensor, expect a
single-digit-to-~15 mIoU off-diagonal drop concentrated in a few site-specific
classes — NOT the -22..-37 mIoU cross-*sensor* drops in the autonomous-driving
literature. Treat any single off-diagonal cell as high variance.

Status: matrix assembly + heatmap implemented; the per-pair training driver is a stub.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def build_matrix(pairwise_miou: dict[tuple[str, str], float], sites: list[str]) -> np.ndarray:
    """pairwise_miou[(train, test)] -> M[i, j] for sites[i]=train, sites[j]=test."""
    n = len(sites)
    M = np.full((n, n), np.nan)
    idx = {s: i for i, s in enumerate(sites)}
    for (tr, te), v in pairwise_miou.items():
        M[idx[tr], idx[te]] = v
    return M


def summarise(M: np.ndarray) -> dict:
    """Mean diagonal (ceiling), mean off-diagonal (transfer), generalisation gap."""
    diag = np.diag(M)
    off = M[~np.eye(M.shape[0], dtype=bool)]
    d, o = float(np.nanmean(diag)), float(np.nanmean(off))
    return {"mean_diag": d, "mean_offdiag": o, "gap": d - o}


def plot_heatmap(M: np.ndarray, sites: list[str], out_png: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.4 * len(sites) + 1, 1.4 * len(sites)))
    im = ax.imshow(M, vmin=0, vmax=max(1.0, np.nanmax(M)), cmap="viridis")
    ax.set_xticks(range(len(sites)), sites, rotation=45, ha="right")
    ax.set_yticks(range(len(sites)), sites)
    ax.set_xlabel("test site")
    ax.set_ylabel("train site")
    for i in range(len(sites)):
        for j in range(len(sites)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                        color="w" if M[i, j] < np.nanmax(M) * 0.6 else "k", fontsize=8)
    fig.colorbar(im, ax=ax, label="mIoU")
    ax.set_title("LOCO cross-site transfer")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")


# --- TODO ---------------------------------------------------------------------
def run_loco(config, sites, out_dir):  # noqa: D401 - stub
    """Train per site, cross-evaluate all pairs, assemble + plot. TODO."""
    raise NotImplementedError("wire to Pointcept train/test per site")
