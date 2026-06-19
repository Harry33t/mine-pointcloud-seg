"""[B4] Bi-temporal change detection (erosion / deposition) via M3C2 (py4dgeo).

M3C2 (Lague et al. 2013) computes signed surface change between two co-registered
epochs along local normals, with a per-point level-of-detection (LoD) so significant
change is distinguishable from noise. Output (signed distance) can be written back as
a LAS scalar field and shown in the Potree viewer alongside the segmentation.

For a mine demo, run between two survey epochs of the same pit (or two Kijkduin
epochs as a prototype). M3C2 is preferred over a raster DEM-of-difference here: it is
pip-installable, needs no GDAL/grid-alignment plumbing, captures full 3D change, and
reports uncertainty.

Status: scaffold — wraps py4dgeo's M3C2; tune normal/projection scales per dataset.
"""
from __future__ import annotations

import numpy as np


def run_m3c2(
    coord_epoch1: np.ndarray,
    coord_epoch2: np.ndarray,
    corepoints: np.ndarray | None = None,
    normal_radii: tuple[float, ...] = (1.0,),
    cyl_radius: float = 0.5,
    max_distance: float = 10.0,
) -> dict:
    """Signed M3C2 distance from epoch1 -> epoch2 at corepoints.

    Returns dict with 'distances' (N,), 'lod' (N,) level-of-detection, and
    'significant' boolean mask (|distance| > lod).
    """
    import py4dgeo

    epoch1 = py4dgeo.Epoch(coord_epoch1.astype(np.float64))
    epoch2 = py4dgeo.Epoch(coord_epoch2.astype(np.float64))
    core = corepoints.astype(np.float64) if corepoints is not None else coord_epoch1

    m3c2 = py4dgeo.M3C2(
        epochs=(epoch1, epoch2),
        corepoints=core,
        normal_radii=normal_radii,
        cyl_radius=cyl_radius,
        max_distance=max_distance,
    )
    distances, uncertainties = m3c2.run()
    lod = np.asarray(uncertainties["lodetection"])
    distances = np.asarray(distances)
    return {
        "distances": distances,
        "lod": lod,
        "significant": np.abs(distances) > lod,
        "corepoints": core,
    }


def net_volume_change(result: dict, cell_area: float) -> float:
    """Rough net volume change = sum(significant signed distance) * cell_area.

    cell_area = footprint per corepoint (e.g. corepoint spacing^2). Positive =
    net deposition, negative = net erosion. Demo-grade estimate.
    """
    d = result["distances"][result["significant"]]
    return float(np.nansum(d) * cell_area)
