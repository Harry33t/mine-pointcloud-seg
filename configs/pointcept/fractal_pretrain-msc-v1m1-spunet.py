# In-domain self-supervised pretraining (MSC, #3) on unlabeled aerial patches.
# MSC + SpUNet (the tested combo; PTv3 returns a Point obj that MSC can't index).
# Adapted from configs/scannet/pretrain-msc-v1m1-0-spunet-base.py for FRACTAL:
#   * no normals  -> color-only features (in_channels=3), reconstruct_normal=False
#   * color already in [0,1] -> drop NormalizeColor
#   * scales tuned for 50 m aerial tiles (grid 0.1 m, mask patch 2 m, match radius 0.5 m)
# Run (smoke = MSC_EPOCH=2; full default = 400):
#   MSC_EPOCH=400 sh scripts/train.sh -p python -g 1 -d fractal -c pretrain-msc-v1m1-spunet -n msc_pretrain
# Output backbone checkpoint: exp/fractal/<name>/model/model_last.pth

_base_ = ["../_base_/default_runtime.py"]

import os as _os
MSC_EPOCH = int(_os.environ.get("MSC_EPOCH", "400"))
del _os

batch_size = 4    # two views/sample + contrastive matching is memory-heavy on 24 GB
num_worker = 12
mix_prob = 0
empty_cache = False
enable_amp = False
evaluate = False
enable_wandb = False
find_unused_parameters = False

model = dict(
    type="MSC-v1m1",
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=3,  # color only
        num_classes=0,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ),
    backbone_in_channels=3,
    backbone_out_channels=96,
    mask_grid_size=2.0,       # 2 m mask patches over 50 m tiles
    mask_rate=0.4,
    view1_mix_prob=0.8,
    view2_mix_prob=0,
    matching_max_k=8,
    matching_max_radius=0.5,  # 0.5 m correspondence radius (aerial scale)
    matching_max_pair=8192,
    nce_t=0.4,
    contrast_weight=1,
    reconstruct_weight=1,
    reconstruct_color=True,
    reconstruct_normal=False,
)

epoch = MSC_EPOCH
eval_epoch = MSC_EPOCH    # no eval during pretraining; keep epoch % eval_epoch == 0
optimizer = dict(type="SGD", lr=0.1, momentum=0.8, weight_decay=0.0001, nesterov=True)
scheduler = dict(
    type="OneCycleLR",
    max_lr=0.1,
    pct_start=0.02,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=10000.0,
)

dataset_type = "DefaultDataset"
data_root = "/root/autodl-tmp/fractal/pretrain"
names = ["other", "ground", "vegetation", "building", "water", "bridge", "permanent"]

data = dict(
    num_classes=7,
    ignore_index=-1,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="Copy", keys_dict={"coord": "origin_coord"}),
            dict(type="Update", keys_dict={"index_valid_keys": ["coord", "color", "origin_coord"]}),
            dict(
                type="ContrastiveViewsGenerator",
                view_keys=("coord", "color", "origin_coord"),
                view_trans_cfg=[
                    # ContrastiveViewsGenerator does NOT copy index_valid_keys into the
                    # per-view dict, so set it here or GridSample/SphereCrop use the
                    # default (which omits origin_coord) and desync it from coord.
                    dict(type="Update", keys_dict={"index_valid_keys": ["coord", "color", "origin_coord"]}),
                    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=1),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=1),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=1),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    dict(type="GridSample", grid_size=0.1, hash_type="fnv", mode="train",
                         return_grid_coord=True),
                    dict(type="SphereCrop", point_max=40000, mode="random"),
                    dict(type="CenterShift", apply_z=False),
                ],
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "view1_origin_coord", "view1_grid_coord", "view1_coord", "view1_color",
                    "view2_origin_coord", "view2_grid_coord", "view2_coord", "view2_color",
                ),
                offset_keys_dict=dict(view1_offset="view1_coord", view2_offset="view2_coord"),
                view1_feat_keys=("view1_color",),
                view2_feat_keys=("view2_color",),
            ),
        ],
        test_mode=False,
    ),
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=None),
]
