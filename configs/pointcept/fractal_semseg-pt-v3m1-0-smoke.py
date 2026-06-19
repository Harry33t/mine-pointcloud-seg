# FRACTAL aerial-LiDAR semantic segmentation — PTv3, single-GPU smoke config.
# Deploy as Pointcept/configs/fractal/semseg-pt-v3m1-0-base.py and run:
#   sh scripts/train.sh -p python -g 1 -d fractal -c semseg-pt-v3m1-0-base -n smoke
# Adapted from configs/nuscenes/semseg-pt-v3m1-0-base.py. FlashAttention is OFF
# (deferred build); batch + epochs are tiny just to validate the pipeline end-to-end.

_base_ = ["../_base_/default_runtime.py"]

# misc
batch_size = 1            # single 24 GB 4090D, no flash-attn (build flash for larger bs)
num_worker = 8
mix_prob = 0              # disable Mixup for the smoke run
empty_cache = True       # clear cache each iter — avoids fragmentation OOM (no flash)
enable_amp = True
enable_wandb = False      # override default_runtime (avoids wandb login prompt)

# model
model = dict(
    type="DefaultSegmentorV2",
    num_classes=7,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=7,    # coord(3) + color(3) + strength(1)
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        # small attention windows: non-flash softmax memory is ~quadratic in patch_size
        enc_patch_size=(256, 256, 256, 256, 256),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(256, 256, 256, 256),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=False,   # deferred — build later for speed/memory
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

# scheduler — tiny for a smoke run
epoch = 2
eval_epoch = 2
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

# dataset
dataset_type = "DefaultDataset"
data_root = "/root/autodl-tmp/fractal/pointcept"
ignore_index = -1
names = [
    "other", "ground", "vegetation", "building",
    "water", "bridge", "permanent_structure",
]

grid = 0.2          # coarse voxels: caps points/scene so the biggest patch fits 24 GB
point_max = 50000   # hard per-scene point cap (no flash-attn -> bound peak memory)
feat_keys = ("coord", "color", "strength")

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
            aug_transform=[
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
            ],
        ),
        ignore_index=ignore_index,
    ),
)
