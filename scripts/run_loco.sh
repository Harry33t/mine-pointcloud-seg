#!/bin/bash
# [B2] LOCO: train one SpUNet per site, then cross-evaluate into a transfer matrix.
export PATH=/root/miniconda3/bin:/usr/local/cuda-12.4/bin:$PATH
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/root/Pointcept:/root/mine-pointcloud-seg/src
cd /root/Pointcept
LOG=/root/autodl-tmp/loco.log
echo "START $(date)" > "$LOG"

for k in 0 1 2 3; do
  rm -rf "exp/fractal/loco_site_${k}"
  LOCO_SITE=${k} sh scripts/train.sh -p python -g 1 -d fractal \
    -c semseg-spunet-loco -n "loco_site_${k}" \
    > "/root/autodl-tmp/loco_site_${k}.log" 2>&1
  echo "site_${k} trained $(date)" >> "$LOG"
done
echo "TRAIN_DONE $(date)" >> "$LOG"

python -m mpcseg.evaluate.loco_eval \
  --config /root/Pointcept/configs/fractal/semseg-spunet-loco.py \
  --exp-root /root/Pointcept/exp/fractal \
  --loco-root /root/autodl-tmp/fractal/loco \
  --sites 4 --out /root/autodl-tmp/fractal/loco/loco_matrix.png >> "$LOG" 2>&1
echo "ALL_DONE $(date)" >> "$LOG"
