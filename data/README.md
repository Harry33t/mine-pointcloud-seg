# Data menu — mine-pointcloud-seg

All open / CC-BY. Download into `data/raw/`, convert to Pointcept format under `data/processed/`.

| Dataset | Use | Source |
|---|---|---|
| OpenTopography open-pit coal mine LiDAR (McKinley / Centralia / John Henry) | Real mine point clouds; LOCO cross-mine | portal.opentopography.org (search mine name) |
| DALES | Self-supervised pre-training corpus + ALS segmentation benchmark | go.udayton.edu/dales3d |
| Hessigheim 3D (H3D) | Cross-sensor gap (photogrammetry mesh + LiDAR) — stretch | ifp.uni-stuttgart.de/benchmark/hessigheim |
| Kijkduin 4D + py4dgeo (M3C2) | Bi-temporal change (DoD) module | PANGAEA + github.com/3dgeo-heidelberg/py4dgeo |

> Note: there is no public *labelled* mine point cloud — this scarcity is itself the motivation
> for the self-supervised / weak-label approach. Weak labels come from geometric features
> (height / curvature / normals) + a few manual points → pseudo-label self-training.

Tip: OpenTopography tiles can be huge — crop a small AOI first, get the pipeline running, then scale up.
