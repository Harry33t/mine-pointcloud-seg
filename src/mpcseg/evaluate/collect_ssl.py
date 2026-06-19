"""[#3] Collect the SSL-vs-scratch finetune results and plot the comparison curve.

Reads ssl_scratch_<pct>.log and ssl_msc_<pct>.log and overlays the two label-
efficiency curves (random init vs MSC-pretrained init).

    python -m mpcseg.evaluate.collect_ssl --log-dir /root/autodl-tmp \
        --out /root/autodl-tmp/fractal/ssl/ssl_curve.png
"""
from __future__ import annotations

import argparse
import os

from mpcseg.evaluate.collect_labeleff import miou_from_log
from mpcseg.evaluate.label_efficiency import plot_curve

PCTS = [1, 5, 10, 100]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-dir", default="/root/autodl-tmp")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results = {"scratch": {}, "msc-init": {}}
    print(f"{'labels %':>9} | {'scratch':>8} | {'msc-init':>8}")
    for pct in PCTS:
        s = miou_from_log(os.path.join(args.log_dir, f"ssl_scratch_{pct}.log"))
        m = miou_from_log(os.path.join(args.log_dir, f"ssl_msc_{pct}.log"))
        if s is not None:
            results["scratch"][pct / 100.0] = s
        if m is not None:
            results["msc-init"][pct / 100.0] = m
        print(f"{pct:>9} | {('%.4f' % s) if s is not None else '   -':>8} | "
              f"{('%.4f' % m) if m is not None else '   -':>8}")

    if results["scratch"] or results["msc-init"]:
        plot_curve(results, args.out)
    else:
        print("No SSL results found yet.")


if __name__ == "__main__":
    main()
