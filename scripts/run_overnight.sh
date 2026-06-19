#!/bin/bash
# Unattended overnight pipeline (#3): MSC pretrain -> SSL-vs-scratch finetune sweep
# -> collect/plot -> shutdown. Failures in one run don't abort the chain (no set -e).
export PATH=/root/miniconda3/bin:/usr/local/cuda-12.4/bin:$PATH
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/root/Pointcept:/root/mine-pointcloud-seg/src
cd /root/Pointcept
LOG=/root/autodl-tmp/overnight.log
echo "START $(date)" > "$LOG"

# 1) MSC pretraining (in-domain SSL on 4000 unlabeled aerial patches)
rm -rf exp/fractal/msc_pretrain
MSC_EPOCH=32 sh scripts/train.sh -p python -g 1 -d fractal \
  -c pretrain-msc-v1m1-spunet -n msc_pretrain > /root/autodl-tmp/msc_pretrain.log 2>&1
echo "PRETRAIN_DONE $(date)" >> "$LOG"
MSC=/root/Pointcept/exp/fractal/msc_pretrain/model/model_last.pth

# 2) finetune sweep: scratch vs MSC-init at 1/5/10/100% labels
for p in 1 5 10 100; do
  rm -rf "exp/fractal/ssl_scratch_${p}"
  LABEL_PCT=${p} sh scripts/train.sh -p python -g 1 -d fractal \
    -c semseg-spunet-v1m1-ssl -n "ssl_scratch_${p}" \
    > "/root/autodl-tmp/ssl_scratch_${p}.log" 2>&1
  echo "scratch_${p} done $(date)" >> "$LOG"

  rm -rf "exp/fractal/ssl_msc_${p}"
  LABEL_PCT=${p} SSL_WEIGHT="${MSC}" sh scripts/train.sh -p python -g 1 -d fractal \
    -c semseg-spunet-v1m1-ssl -n "ssl_msc_${p}" \
    > "/root/autodl-tmp/ssl_msc_${p}.log" 2>&1
  echo "msc_${p} done $(date)" >> "$LOG"
done
echo "FINETUNE_DONE $(date)" >> "$LOG"

# 3) collect + plot
mkdir -p /root/autodl-tmp/fractal/ssl
python -m mpcseg.evaluate.collect_ssl --log-dir /root/autodl-tmp \
  --out /root/autodl-tmp/fractal/ssl/ssl_curve.png >> "$LOG" 2>&1
echo "ALL_DONE $(date)" >> "$LOG"

# 4) shutdown early (saves cost); the 7h backstop covers any hang
sync
/usr/bin/shutdown -h now 2>/dev/null || /usr/bin/shutdown 2>/dev/null || poweroff
