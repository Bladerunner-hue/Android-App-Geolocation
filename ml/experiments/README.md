# Experimental ML (not production)

**Production spine is dense `fusion_v0` only.** Everything here is demoted for
research / CI smoke / historical comparison. Never import these as the default
train or TFLite path.

| Module | Role | Release? |
|--------|------|----------|
| `fusion_moe_v0.py` + `sparse_moe.py` | Same I/O as fusion_v0, top-k MoE FFN | Optional capacity |
| `train_moe_v0.py` | Explicit MoE train entry (same NPZ) | Not default |
| `moe_kickstart.py` | Raw-media sparse MoE prototype | No |
| `train_moe_legacy.py` | Old raw-media MoE train | No |
| `synthetic_bootstrap.py` | Fake TFRecords for pipeline wiring | CI smoke only |
| `sound_bootstrap.py` | ESC-10/FSD event lists (not vibes) | Aux only |
| `vibe_fusion/` | Incomplete legacy dense shim | No |
| `test_*` | Experimental contract tests | Optional |

## Optional capacity upgrade (same TFLite I/O)

```bash
python -m ml.experiments.train_moe_v0 \
  --manifest path/to/manifest.json \
  --weights-out ml/artifacts/fusion-moe-v0.weights.h5 \
  --init-from-dense ml/artifacts/fusion-v0.weights.h5

python -m ml.export_tflite \
  --weights ml/artifacts/fusion-moe-v0.weights.h5 \
  --architecture moe \
  --parity-data ... --out ...
```

## Legacy raw-media MoE smoke

```bash
python -m ml.experiments.synthetic_bootstrap
python -m ml.experiments.train_moe_legacy --cpu --epochs 2 --bootstrap
```

## Production (always default)

```bash
python -m ml.prepare_fusion_dataset ...
python -m ml.validate_fusion_dataset --manifest ...
python -m ml.train_fusion_v0 --manifest ...
python -m ml.export_tflite --weights ...   # --architecture dense (default)
```

Never make experimental modules the default import for train or TFLite.
