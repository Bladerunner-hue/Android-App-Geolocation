# Experimental ML (not production)

**Production spine is dense `fusion_v0` only.** Everything here is demoted for
research / CI smoke / historical comparison. Never import these as the default
train or TFLite path.

| Module | Role | Release? |
|--------|------|----------|
| `moe_kickstart.py` | Raw-media sparse MoE prototype | No |
| `train_moe_legacy.py` | Old `ml.train` entry for MoE | No |
| `synthetic_bootstrap.py` | Fake TFRecords for pipeline wiring | CI smoke only |
| `sound_bootstrap.py` | ESC-10/FSD event lists (not vibes) | Aux only |
| `vibe_fusion/` | Incomplete legacy dense shim | No |
| `test_vibe_fusion.py` | Legacy contract tests | Optional |

## Run experimental MoE (optional)

```bash
# synthetic TFRecords for smoke only — never for release weights
python -m ml.experiments.synthetic_bootstrap
python -m ml.experiments.train_moe_legacy --cpu --epochs 2 --bootstrap
```

## Production (always)

```bash
python -m ml.prepare_fusion_dataset ...
python -m ml.validate_fusion_dataset --manifest ...
python -m ml.train_fusion_v0 --manifest ...
python -m ml.export_tflite --weights ...
```

Optional MoE capacity upgrade (same I/O as fusion_v0) stays behind an explicit
flag / separate train entry when implemented — never replace the default import.
