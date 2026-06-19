#!/bin/bash
# #3 SSL-vs-scratch finetune sweep: SpUNet at 1/5/10/100% labels, two arms each
# (random init vs MSC-pretrained init). Run after MSC pretraining finishes.
set -u
export PATH=/root/miniconda3/bin:/usr/local/cuda-12.4/bin:$PATH
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/root/Pointcept:/root/mine-pointcloud-seg/src
cd /root/Pointcept

MSC=/root/Pointcept/exp/fractal/msc_pretrain/model/model_last.pth

for p in 1 5 10 100; do
  rm -rf "exp/fractal/ssl_scratch_${p}"
  echo "=== START scratch_${p} $(date) ==="
  LABEL_PCT=${p} sh scripts/train.sh -p python -g 1 -d fractal \
    -c semseg-spunet-v1m1-ssl -n "ssl_scratch_${p}" \
    > "/root/autodl-tmp/ssl_scratch_${p}.log" 2>&1
  echo "=== DONE scratch_${p} $(date) ==="

  rm -rf "exp/fractal/ssl_msc_${p}"
  echo "=== START msc_${p} $(date) ==="
  LABEL_PCT=${p} SSL_WEIGHT=${MSC} sh scripts/train.sh -p python -g 1 -d fractal \
    -c semseg-spunet-v1m1-ssl -n "ssl_msc_${p}" \
    > "/root/autodl-tmp/ssl_msc_${p}.log" 2>&1
  echo "=== DONE msc_${p} $(date) ==="
done
echo ALL_SSL_DONE
