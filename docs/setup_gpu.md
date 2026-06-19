# GPU environment setup (AutoDL / Linux, single GPU)

The segmentation backbone (Point Transformer V3 via **Pointcept**) needs compiled
CUDA extensions (`pointops`, FlashAttention, spconv). These do **not** build cleanly
on native Windows — use a Linux box (AutoDL 4090 / 48 GB card) or WSL2/Docker.

Target stack (verified current, 2026): Python 3.10 · PyTorch 2.5.0 · CUDA 12.4.

## Option A — conda (recommended on AutoDL)

```bash
# 0. clone Pointcept next to this repo
git clone https://github.com/Pointcept/Pointcept.git
cd Pointcept

# 1. create env
conda create -n pcseg python=3.10 -y
conda activate pcseg

# 2. PyTorch 2.5.0 + CUDA 12.4
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124

# 3. core deps
conda install -c pyg pytorch-cluster pytorch-scatter pytorch-sparse -y
pip install torch-geometric
pip install spconv-cu124
pip install h5py pyyaml tensorboard tensorboardx wandb yapf addict einops scipy plyfile termcolor timm open3d

# 4. compiled point ops — set arch to your GPU (RTX 4090 = 8.9)
cd libs/pointops
TORCH_CUDA_ARCH_LIST="8.9" python setup.py install
cd ../..

# 5. FlashAttention (PTv3 default; or disable with enable_flash=false in config)
pip install flash-attn --no-build-isolation
```

## Option B — official Docker image

```bash
docker pull pointcept/pointcept:v1.6.0-pytorch2.5.0-cuda12.4-cudnn9-devel
```

## Pretrained weights (Sonata self-supervised, PTv3-native)

- Repo: https://github.com/facebookresearch/sonata · paper https://arxiv.org/abs/2503.16429
- Weights on the Pointcept HuggingFace org. License: **CC-BY-NC 4.0 (non-commercial)** —
  fine for a research demo, not for commercial deployment.
- ⚠️ Sonata weights are pretrained on **indoor** scenes; expect a domain gap to aerial
  mine/terrain. We fine-tune from them rather than pretraining from scratch.

## Verified working recipe (AutoDL 4090D, 2026-06-18)

The base AutoDL image already had **torch 2.5.1+cu124** + CUDA toolkit 12.4 — no torch
reinstall needed. Beyond the deps above, these were also required for Pointcept v1.7
to import and run (install into base env):

```bash
pip install peft wandb open3d ftfy regex sharedarray   # train.py / dataset imports
pip install laspy lazrs                                  # for data conversion (mpcseg)
export WANDB_MODE=disabled                               # skip wandb login
```

**No-flash-attn memory caps (single 24 GB, until FlashAttention is built):** PTv3's
plain (non-flash) attention materialises the score matrix, so peak memory is large and
scales with points/scene. To fit 24 GB use: `enable_flash=False`, `batch_size=1`,
`enc/dec_patch_size=256`, `GridSample grid_size>=0.2`, a `SphereCrop point_max~50000`,
and `empty_cache=True`. See `configs/pointcept/fractal_semseg-pt-v3m1-0-smoke.py`.
Building FlashAttention lifts these limits (bigger batch, full-resolution scenes).

## Quick smoke test

```bash
cd Pointcept
sh scripts/train.sh -p python -g 1 -d s3dis -c semseg-pt-v3m1-0-rpe -n smoke
```

If that trains an epoch, the GPU env is good. Custom mine/DALES configs go in
`../mine-pointcloud-seg/configs/pointcept/` (symlink or copy into Pointcept/configs).

## VRAM / time notes (single GPU)

- 48 GB (RTX 6000 Ada): batch ~10, ~9 h for 120 epochs on ~140 M-point ALS set.
- 24 GB (RTX 4090): batch ~4–6, smaller tiles; ~8–15 h. Lower `batch_size` in the
  config and reduce `grid_size` / chunk range if you hit OOM.
- PTv3 is weak on sparse low-vegetation/bare-earth classes — consider running a
  KPConv baseline (also in Pointcept) for comparison on those classes.
