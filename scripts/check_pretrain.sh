#!/bin/bash
# Health snapshot of the MSC pretraining run.
echo "=== $(date) ==="
echo -n "last: "; grep -E "Train: .[0-9]+/40" /root/autodl-tmp/msc_pretrain.log 2>/dev/null | tail -1
echo -n "errors: "; grep -cE "Traceback|OutOfMemory" /root/autodl-tmp/msc_pretrain.log 2>/dev/null
if pgrep -f msc_pretrain >/dev/null || pgrep -f "n msc_pretrain" >/dev/null; then echo "PROC: alive"; else echo "PROC: not found"; fi
echo -n "gpu: "; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
if [ -f /root/Pointcept/exp/fractal/msc_pretrain/model/model_last.pth ]; then
  echo "CKPT: saved ($(stat -c %y /root/Pointcept/exp/fractal/msc_pretrain/model/model_last.pth | cut -d. -f1))"
else
  echo "CKPT: not yet"
fi
