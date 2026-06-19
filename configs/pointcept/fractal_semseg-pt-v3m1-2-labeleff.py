# FRACTAL label-efficiency run (B1) — PTv3 + FlashAttention, single 4090D.
# One config, four runs: set LABEL_PCT in {1,5,10,100} before launching.
#   for p in 1 5 10 100; do LABEL_PCT=$p sh scripts/train.sh -p python -g 1 \
#       -d fractal -c semseg-pt-v3m1-2-labeleff -n labeleff_$p; done
# Train labels come from the fraction dirs built by mpcseg.evaluate.label_efficiency;
# val/test use the full-label split. 20 epochs to keep the 4-run sweep affordable.

_base_ = ["../_base_/default_runtime.py"]

# read label fraction from env; del the module name so Pointcept's Config (which
# deepcopies every top-level var) doesn't choke trying to pickle a module object.
import os as _os
LABEL_PCT = int(_os.environ.get("LABEL_PCT", "100"))
del _os

# misc
batch_size = 2
num_worker = 12
mix_prob = 0
empty_cache = False
enable_amp = True
enable_wandb = False

# model (identical to the flash baseline)
model = dict(
    type="DefaultSegmentorV2",
    num_classes=7,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=7,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
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
        enc_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("FRACTAL",),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler
epoch = 20
eval_epoch = 20
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
param_dicts = [dict(keyword="block", lr=0.0002)]

# dataset — train from the fraction dir, val/test from the full-label split
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
