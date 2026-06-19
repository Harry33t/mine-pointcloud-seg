#!/bin/bash
# Sequential label-efficiency sweep (B1): train PTv3 at 1/5/10/100% labels.
# Run on the GPU box (detached): setsid nohup bash run_labeleff.sh > master.log 2>&1 &
set -u
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=/root/miniconda3/bin:$CUDA_HOME/bin:$PATH
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/root/Pointcept:/root/mine-pointcloud-seg/src
cd /root/Pointcept

for p in 1 5 10 100; do
  rm -rf "exp/fractal/labeleff_${p}"
  echo "=== START labeleff_${p} $(date) ==="
  LABEL_PCT=${p} sh scripts/train.sh -p python -g 1 -d fractal \
    -c semseg-pt-v3m1-2-labeleff -n "labeleff_${p}" \
    > "/root/autodl-tmp/labeleff_${p}.log" 2>&1
  echo "=== DONE labeleff_${p} $(date) ==="
done
echo ALL_LABELEFF_DONE
