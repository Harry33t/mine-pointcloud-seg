# Pointcept configs

Custom dataset/training configs for this project, kept in the repo and copied (or
symlinked) into `Pointcept/configs/` on the GPU box at run time.

## How configs work

Pointcept configs are Python files that compose a `dict`-style config (backbone,
dataset, augmentations, optimizer, schedule). Start from an **outdoor / intensity**
template (no RGB), since ALS/mine data has intensity but usually no colour:

- `semantic_kitti/semseg-pt-v3m1-0-base.py` — outdoor PTv3, uses `strength`
- `s3dis/semseg-pt-v3m1-0-rpe.py` — indoor PTv3 reference (for the smoke test)

## To add a DALES / mine config

1. Register a dataset class (or reuse `DefaultDataset`) pointing at
   `data/processed/<dataset>` with the npy keys produced by `mpcseg.data.to_pointcept`
   (`coord`, `strength`, optional `color`/`normal`, `segment`).
2. Set `num_classes`, the class names, and `ignore_index = -1`.
3. For single-GPU 24 GB: lower `batch_size` (~4–6), keep `enable_flash=True`,
   reduce `grid_size` / chunk range if OOM.
4. To init from Sonata self-supervised weights, set the backbone `pretrained` /
   `weight` path to the downloaded Sonata checkpoint (see docs/setup_gpu.md).

## Building blocks that consume these runs

- B1 label-efficiency: train per fraction with `mpcseg.evaluate.label_efficiency`
  writing `segment_<frac>.npy`, point the config's label key at it.
- B2 LOCO: one config per train site, cross-evaluate via `mpcseg.evaluate.loco_matrix`.

Files will be added here as each building block is wired. Placeholder for now.
