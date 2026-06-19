# FRACTAL finetune from Sonata self-supervised weights (B1 SSL-vs-scratch, #3).
# Backbone architecture MUST match the Sonata checkpoint (PT-v3m2 encoder dims). The
# encoder + embedding-norm load from the prepared init weight; decoder + seg_head are
# trained from scratch. Input stem is reinitialised (our in_channels=7 != Sonata's 9).
#
# Prepare the init weight once:
#   python scripts/prep_sonata_init.py --in sonata.pth --out sonata_backbone_init.pth
# Then sweep label fractions (compare against the PT-v3m1 from-scratch curve):
#   for p in 1 5 10 100; do LABEL_PCT=$p sh scripts/train.sh -p python -g 1 \
#       -d fractal -c semseg-pt-v3m2-3-sonata -n sonata_$p; done

_base_ = ["../_base_/default_runtime.py"]

import os as _os
LABEL_PCT = int(_os.environ.get("LABEL_PCT", "100"))
del _os

# load Sonata encoder weights as initialisation (strict=False via CheckpointLoader)
weight = "/root/autodl-tmp/sonata_backbone_init.pth"

# misc
batch_size = 2
num_worker = 12
mix_prob = 0
clip_grad = 3.0          # finetuning stability (as in Sonata ft configs)
empty_cache = False
enable_amp = True
enable_wandb = False

# model — PT-v3m2 with Sonata's encoder dimensions
model = dict(
    type="DefaultSegmentorV2",
    num_classes=7,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m2",
        in_channels=7,                       # coord(3)+color(3)+strength(1); stem reinit
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),         # <- Sonata dims
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 96, 192, 384),
        dec_num_head=(4, 6, 12, 24),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        enc_mode=False,
        freeze_encoder=False,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    freeze_backbone=False,
)

# scheduler
epoch = 20
eval_epoch = 20
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.02)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0002)]

# dataset (identical to the label-efficiency config)
dataset_type = "DefaultDataset"
labeleff_root = "/root/autodl-tmp/fractal/labeleff"
full_root = "/root/autodl-tmp/fractal/pointcept_full"
ignore_index = -1
names = [
    "other", "ground", "vegetation", "building",
    "water", "bridge", "permanent_structure",
]

grid = 0.1
point_max = 80000
feat_keys = ("coord", "color", "strength")

data = dict(
    num_classes=7,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split=f"train_{LABEL_PCT}",
        data_root=labeleff_root,
        transform=[
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="GridSample", grid_size=grid, hash_type="fnv", mode="train",
                 return_grid_coord=True),
            dict(type="SphereCrop", point_max=point_max, mode="random"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment"),
                 feat_keys=feat_keys),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=full_root,
        transform=[
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(type="GridSample", grid_size=grid, hash_type="fnv", mode="train",
                 return_grid_coord=True, return_inverse=True),
            dict(type="ToTensor"),
            dict(type="Collect",
                 keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                 feat_keys=feat_keys),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=full_root,
        transform=[
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(type="GridSample", grid_size=grid, hash_type="fnv", mode="train",
                 return_inverse=True),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(type="GridSample", grid_size=grid, hash_type="fnv",
                          mode="test", return_grid_coord=True),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(type="Collect", keys=("coord", "grid_coord", "index"),
                     feat_keys=feat_keys),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
            ],
        ),
        ignore_index=ignore_index,
    ),
)
