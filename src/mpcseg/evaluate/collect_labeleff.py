"""[B1] Collect val mIoU from the label-efficiency sweep logs and plot the curve.

Parses each labeleff_<pct>.log produced by scripts/run_labeleff.sh, extracts the
best-model test mIoU, and draws the label-efficiency curve (mIoU vs % labels).

Run after the sweep finishes:
    python -m mpcseg.evaluate.collect_labeleff \
        --log-dir /root/autodl-tmp --out /root/autodl-tmp/fractal/labeleff/curve.png
"""
from __future__ import annotations

import argparse
import os
import re

from mpcseg.evaluate.label_efficiency import plot_curve

PCTS = [1, 5, 10, 100]
# "Val result: mIoU/mAcc/allAcc 0.4679/0.5817/0.9442"  (PreciseEvaluator)
_TEST = re.compile(r"Val result:\s*mIoU/mAcc/allAcc\s*([0-9.]+)")
# "Currently Best mIoU: 0.4602"
_BEST = re.compile(r"Currently Best mIoU:\s*([0-9.]+)")
# "Class_5 - bridge Result: iou/accuracy 0.2414/0.3851"
_CLS = re.compile(r"Class_(\d+) - (\w+) Result: iou/accuracy ([0-9.]+)")


def per_class_from_log(path: str) -> dict[str, float]:
    """Final-evaluation per-class IoU {class_name: iou} from a run log."""
    if not os.path.exists(path):
        return {}
    text = open(path, encoding="utf-8", errors="replace").read()
    out: dict[str, float] = {}
    for _id, name, iou in _CLS.findall(text):  # later lines overwrite -> keep final
        out[name] = float(iou)
    return out


def plot_per_class(per_class: dict[int, dict[str, float]], out_png: str) -> None:
    """per_class[pct] = {name: iou}. One line per class, IoU vs % labels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pcts = sorted(per_class)
    names = list(next(iter(per_class.values())).keys()) if per_class else []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in names:
        ys = [per_class[p].get(name, float("nan")) for p in pcts]
        ax.plot(pcts, ys, marker="o", label=name)
    ax.set_xscale("log")
    ax.set_xticks(pcts)
    ax.set_xticklabels([str(p) for p in pcts])
    ax.set_xlabel("labelled fraction (%)")
    ax.set_ylabel("per-class IoU")
    ax.set_title("Label efficiency by class — rare classes need more labels")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")


def miou_from_log(path: str) -> float | None:
    if not os.path.exists(path):
        return None
    text = open(path, encoding="utf-8", errors="replace").read()
    m = _TEST.findall(text)
    if m:
        return float(m[-1])
    b = _BEST.findall(text)
    return float(b[-1]) if b else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-dir", default="/root/autodl-tmp")
    ap.add_argument("--prefix", default="labeleff_")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    curve = {}
    print(f"{'labels %':>9} | {'mIoU':>6}")
    for pct in PCTS:
        log = os.path.join(args.log_dir, f"{args.prefix}{pct}.log")
        miou = miou_from_log(log)
        if miou is None:
            print(f"{pct:>9} | (missing)")
            continue
        curve[pct / 100.0] = miou
        print(f"{pct:>9} | {miou:.4f}")

    if curve:
        plot_curve({"from-scratch": curve}, args.out)
    else:
        print("No mIoU values found — is the sweep finished?")

    per_class = {}
    for pct in PCTS:
        pc = per_class_from_log(os.path.join(args.log_dir, f"{args.prefix}{pct}.log"))
        if pc:
            per_class[pct] = pc
    if per_class:
        out_pc = os.path.join(os.path.dirname(args.out), "labeleff_per_class.png")
        plot_per_class(per_class, out_pc)


if __name__ == "__main__":
    main()
