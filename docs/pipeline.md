# Pipeline overview

End-to-end flow and where each module lives. Building blocks B1–B4 map to PLAN.md.

```
                 ┌─────────────────── data/raw ───────────────────┐
                 │  DALES tiles (labelled ALS, validate pipeline)  │
                 │  McKinley Mine AOI clip (raw, unlabelled mine)  │
                 │  Kijkduin 4D epochs (bi-temporal, for B4)       │
                 └────────────────────────┬───────────────────────┘
                                          │  src/mpcseg/data
                       crop_aoi → to_pointcept (npy dict)
                                          │
                                          ▼
                 data/processed/<dataset>/<scene>/{coord,strength,color,segment}.npy
                                          │
          ┌───────────────────────────────┼────────────────────────────────┐
          │                               │                                  │
  weak labels (no GT)              supervised finetune                 self-supervised
  src/mpcseg/weaklabels       Pointcept PTv3 + Sonata init            (Sonata weights /
  geometric features →        configs/pointcept/*.py                   Pointcept MSC)
  pseudo-labels + self-train          │
          │                           ▼
          └──────────────►   trained checkpoint + per-point logits
                                          │
        ┌─────────────────────┬──────────┴──────────┬────────────────────┐
        ▼                     ▼                      ▼                    ▼
  [B1] label-eff curve   [B2] LOCO matrix   [B3] uncertainty        [B4] change (M3C2)
  evaluate/label_        evaluate/loco_     uncertainty/temp_       change/m3c2_
  efficiency.py          matrix.py          scaling.py + entropy    change.py
        └─────────────────────┴──────────────┬───────┴────────────────────┘
                                              ▼
                            viz/export_las_scalars.py  (pred / gt / entropy
                            as LAS 1.4 extra dims)  →  viz/make_potree.py
                                              ▼
                            Potree static page: switch RGB ↔ pred ↔ GT ↔ uncertainty
```

## Module map

| Path | Role | Status |
|---|---|---|
| `src/mpcseg/common/io.py` | laspy/Open3D read/write, bbox crop | implemented |
| `src/mpcseg/data/crop_aoi.py` | crop/tile large LAZ to AOI | implemented |
| `src/mpcseg/data/to_pointcept.py` | LAS/PLY → Pointcept npy dict | implemented |
| `src/mpcseg/data/download_dales.py` | fetch/extract DALES (after form) | helper |
| `src/mpcseg/data/download_mckinley.py` | OpenTopography AOI clip | helper |
| `src/mpcseg/weaklabels/geometric_pseudolabels.py` | features → pseudo-labels | scaffold |
| `src/mpcseg/uncertainty/temperature_scaling.py` | calibration + per-point entropy | implemented |
| `src/mpcseg/evaluate/label_efficiency.py` | B1 curve | scaffold |
| `src/mpcseg/evaluate/loco_matrix.py` | B2 site×site matrix | scaffold |
| `src/mpcseg/change/m3c2_change.py` | B4 py4dgeo M3C2 | scaffold |
| `src/mpcseg/viz/export_las_scalars.py` | write scalar fields to LAS 1.4 | implemented |
| `src/mpcseg/viz/make_potree.py` | run PotreeConverter | implemented |

"scaffold" = interface + docstrings + TODO, wired but not yet validated against real data.
