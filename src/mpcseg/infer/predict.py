"""Run a trained Pointcept PTv3 model on scenes and export per-point outputs to LAS.

Produces, per scene, full-resolution arrays aligned to the original coord.npy:
  pred    : argmax class
  gt      : ground-truth class (segment.npy)
  entropy : softmax predictive entropy (per-point uncertainty)
and writes them as LAS 1.4 extra dimensions (via mpcseg.viz.export_las_scalars) so
PotreeConverter can show a switchable RGB / prediction / ground-truth / uncertainty
viewer.

Runs on the GPU box (needs Pointcept + the trained checkpoint). Example:
    python -m mpcseg.infer.predict \
        --config /root/Pointcept/configs/fractal/semseg-pt-v3m1-1-flash.py \
        --checkpoint /root/Pointcept/exp/fractal/flash30/model/model_best.pth \
        --out-dir /root/autodl-tmp/fractal/viz --num-scenes 3
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def _load_model(config_path: str, checkpoint_path: str):
    import torch
    from pointcept.engines.defaults import default_config_parser
    from pointcept.models import build_model

    cfg = default_config_parser(config_path, None)
    model = build_model(cfg.model).cuda().eval()

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load] {len(missing)} missing keys (e.g. {missing[:2]})")
    if unexpected:
        print(f"[load] {len(unexpected)} unexpected keys (e.g. {unexpected[:2]})")
    return cfg, model


def _entropy(prob: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return -(prob * np.log(prob + eps)).sum(axis=1)


def predict_scenes(config_path: str, checkpoint_path: str, out_dir: str,
                   num_scenes: int = 3, scene_names: list[str] | None = None) -> list[str]:
    import torch
    from torch.nn.functional import softmax

    from pointcept.datasets import build_dataset, point_collate_fn
    from mpcseg.viz.export_las_scalars import export_scalars

    cfg, model = _load_model(config_path, checkpoint_path)
    val = build_dataset(cfg.data.val)
    data_root = cfg.data.val["data_root"]
    os.makedirs(out_dir, exist_ok=True)

    # pick scenes by name or by first-N index
    if scene_names:
        names = set(scene_names)
        indices = [i for i in range(len(val)) if val.get_data_name(i) in names]
    else:
        indices = list(range(min(num_scenes, len(val))))

    written: list[str] = []
    for idx in indices:
        name = val.get_data_name(idx)
        batch = point_collate_fn([val[idx]])
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda(non_blocking=True)

        # fp32 inference: avoids a spconv implicit-gemm tuner failure seen under autocast
        with torch.no_grad():
            logits = model(batch)["seg_logits"].float()      # (N_grid, C)
        prob = softmax(logits, dim=1).cpu().numpy()
        pred_grid = prob.argmax(axis=1).astype(np.int64)
        ent_grid = _entropy(prob)

        inverse = batch["inverse"].cpu().numpy()              # (N_full,) -> gridded idx
        pred_full = pred_grid[inverse]
        ent_full = ent_grid[inverse]

        scene_dir = os.path.join(data_root, "val", name)
        gt = np.load(os.path.join(scene_dir, "segment.npy")).reshape(-1)
        out_las = os.path.join(out_dir, f"{name}.las")
        export_scalars(scene_dir, out_las, pred=pred_full, gt=gt, entropy=ent_full)
        written.append(out_las)
        print(f"[{idx}] {name}: {len(pred_full):,} pts, mean entropy {ent_full.mean():.3f}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-scenes", type=int, default=3)
    ap.add_argument("--scenes", nargs="*", help="explicit scene names (overrides --num-scenes)")
    args = ap.parse_args()
    predict_scenes(args.config, args.checkpoint, args.out_dir,
                   num_scenes=args.num_scenes, scene_names=args.scenes)


if __name__ == "__main__":
    main()
