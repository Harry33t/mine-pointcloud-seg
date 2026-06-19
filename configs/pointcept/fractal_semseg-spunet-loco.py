# [B2] Per-site SpUNet training for the leave-one-site-out transfer matrix.
# LOCO_SITE selects which site to train on; cross-evaluation across sites is done
# afterwards by mpcseg.evaluate.loco_eval.
#   LOCO_SITE=0 sh scripts/train.sh -p python -g 1 -d fractal -c semseg-spunet-loco -n loco_site_0

_base_ = ["../_base_/default_runtime.py"]

import os as _os
LOCO_SITE = int(_os.environ.get("LOCO_SITE", "0"))
del _os

batch_size = 8
num_worker = 12
mix_prob = 0
empty_cache = False
enable_amp = True
enable_wandb = False

model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=3,
        num_classes=7,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

epoch = 20
eval_epoch = 20
optimizer = dict(type="SGD", lr=0.05, momentum=0.9, weight_decay=0.0001, nesterov=True)
scheduler = dict(
    type="OneCycleLR",
    max_lr=0.05,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=10000.0,
)

dataset_type = "DefaultDataset"
data_root = f"/root/autodl-tmp/fractal/loco/site_{LOCO_SITE}"
ignore_index = -1
names = ["other", "ground", "vegetation", "building", "water", "bridge", "permanent_structure"]

grid = 0.1
point_max = 80000
feat_keys = ("color",)

data = dict(
    num_classes=7,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
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
        data_root=data_root,
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
        data_root=data_root,
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
            aug_transform=[[dict(type="RandomScale", scale=[1, 1])]],
        ),
        ignore_index=ignore_index,
    ),
)
