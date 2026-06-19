# SpUNet finetune for the SSL-vs-scratch comparison (#3). One config, two arms:
#   scratch:   (no SSL_WEIGHT)            from random init
#   msc-init:  SSL_WEIGHT=<msc ckpt>      load the MSC-pretrained SpUNet encoder
# across label fractions via LABEL_PCT. Backbone matches the MSC pretrain (SpUNet,
# in_channels=3 color). Compare against the PTv3 label-efficiency curve qualitatively;
# the apples-to-apples comparison is scratch-SpUNet vs MSC-SpUNet here.
#   LABEL_PCT=5 SSL_WEIGHT=/root/.../model_last.pth sh scripts/train.sh -p python \
#       -g 1 -d fractal -c semseg-spunet-v1m1-ssl -n ssl_msc_5

_base_ = ["../_base_/default_runtime.py"]

import os as _os
LABEL_PCT = int(_os.environ.get("LABEL_PCT", "100"))
_w = _os.environ.get("SSL_WEIGHT", "").strip()
del _os

weight = _w if _w else None  # CheckpointLoader loads backbone.* (strict=False)

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
        in_channels=3,  # color only — matches the MSC-pretrained backbone
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
labeleff_root = "/root/autodl-tmp/fractal/labeleff"
full_root = "/root/autodl-tmp/fractal/pointcept_full"
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
