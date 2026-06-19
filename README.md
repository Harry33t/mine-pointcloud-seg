# mine-pointcloud-seg

**Label-Efficient Semantic Segmentation of Post-Mining LiDAR Point Clouds**

Self-supervised pre-training + weak labels for semantic segmentation of
open-pit mine and landform LiDAR point clouds. The goal is to reach near-fully-supervised
accuracy with only **5–10% labels**, evaluate **cross-site (leave-one-site-out) transfer**,
and report **calibrated per-point uncertainty** — the three pain points of point-cloud
methods on mine sites (annotation cost, weak cross-site generalisation, no uncertainty).

## Highlights
- **Self-supervised pre-training** on unlabelled mine / terrain point clouds (self-distillation, PTv3-native).
- **Label-efficiency curve**: mIoU at 1 / 5 / 10 / 100% labels (self-supervised vs from scratch).
- **Leave-one-site-out (LOCO)** cross-mine transfer matrix.
- **Per-point uncertainty / entropy** heatmaps.
- **DEM-of-difference (DoD)** change layers for erosion / deposition (landform input layers).
- Interactive 3D viewer (RGB ↔ prediction ↔ ground truth).

## Data (open)
- OpenTopography open-pit mine LiDAR · DALES (ALS benchmark) · Hessigheim H3D · Kijkduin 4D.

## Stack
PyTorch · Pointcept (PTv3) · Sonata (self-supervised) · Open3D · py4dgeo · Potree.

## Status
Work in progress. See project board for the build schedule.

## License
TBD.
