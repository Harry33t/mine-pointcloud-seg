#!/bin/bash
# Health snapshot of the label-efficiency sweep — detects stalls.
echo "=== $(date) ==="
echo "MASTER:"; cat /root/autodl-tmp/labeleff_master.log 2>/dev/null
echo "BEST:"
for p in 1 5 10 100; do
  v=$(grep -oE "Currently Best mIoU: [0-9.]+" "/root/autodl-tmp/labeleff_${p}.log" 2>/dev/null | tail -1)
  echo "  ${p}%: ${v:-pending}"
done
if pgrep -f run_labeleff >/dev/null; then echo "SWEEP: alive"; else echo "SWEEP: not running"; fi
echo -n "GPU: "; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "LOG-ACTIVITY:"
for p in 1 5 10 100; do
  f="/root/autodl-tmp/labeleff_${p}.log"
  [ -f "$f" ] && echo "  ${p}%: mtime $(stat -c %y "$f" | cut -d. -f1), $(grep -cE 'Train: ' "$f") train-iters"
done
