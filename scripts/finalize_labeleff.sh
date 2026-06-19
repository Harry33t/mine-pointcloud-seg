#!/bin/bash
# Wait for the label-efficiency sweep to finish, then plot the curve and dump
# per-class IoU for each fraction.
export PATH=/root/miniconda3/bin:/usr/local/cuda-12.4/bin:$PATH
export PYTHONPATH=/root/Pointcept:/root/mine-pointcloud-seg/src

for i in $(seq 1 60); do
  grep -q ALL_LABELEFF_DONE /root/autodl-tmp/labeleff_master.log && break
  sleep 10
done

echo "=== MASTER ==="
cat /root/autodl-tmp/labeleff_master.log

cd /root/Pointcept
echo "=== CURVE (overall mIoU) ==="
python -m mpcseg.evaluate.collect_labeleff --log-dir /root/autodl-tmp \
  --out /root/autodl-tmp/fractal/labeleff/curve.png

for p in 1 5 10 100; do
  echo "=== ${p}% final per-class IoU ==="
  grep -E 'Class_[0-9] -' "/root/autodl-tmp/labeleff_${p}.log" | tail -7
done
echo FINALIZE_DONE
