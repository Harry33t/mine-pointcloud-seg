"""Convert the released Sonata checkpoint into a Pointcept backbone-init weight.

Sonata (facebook/sonata `sonata.pth`) is an encoder-only PTv3 checkpoint with keys
`embedding.*` / `enc.*`. Pointcept's CheckpointLoader loads `cfg.weight`'s "state_dict"
into the segmentor (strict=False), whose backbone params are prefixed `backbone.`.

This script:
  * prefixes every key with `backbone.`
  * drops `embedding.stem.linear.*` (input dim differs from ours -> would shape-clash;
    the stem is reinitialised and finetuned) and `embedding.mask_token` (finetune has none)
so the result loads cleanly: encoder + embedding-norm transfer, decoder + seg_head are
trained from scratch.

Usage:
    python scripts/prep_sonata_init.py --in /root/autodl-tmp/sonata.pth \
        --out /root/autodl-tmp/sonata_backbone_init.pth
"""
from __future__ import annotations

import argparse

DROP = {
    "embedding.mask_token",
    "embedding.stem.linear.weight",
    "embedding.stem.linear.bias",
}


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ck = torch.load(args.in_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    new = {f"backbone.{k}": v for k, v in sd.items() if k not in DROP}
    torch.save({"state_dict": new}, args.out)
    print(f"in: {len(sd)} params -> out: {len(new)} params (dropped {len(sd) - len(new)})")
    print("sample out keys:", list(new)[:3])


if __name__ == "__main__":
    main()
