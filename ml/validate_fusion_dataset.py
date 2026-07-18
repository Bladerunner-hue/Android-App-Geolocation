"""Validate fusion NPZ splits + manifest (leakage, shapes, class coverage)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Set

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.fusion_v0 import AUDIO_DIM, CONTEXT_DIM, IMAGE_DIM, NUM_VIBES  # noqa: E402


REQUIRED = (
    "image_embedding",
    "audio_embedding",
    "context",
    "modality_mask",
    "vibe_label",
)


def _load(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def validate_npz(path: Path, split: str) -> Dict[str, Any]:
    d = _load(path)
    for k in REQUIRED:
        if k not in d:
            raise AssertionError(f"{split}: missing {k}")
    n = d["vibe_label"].shape[0]
    assert d["image_embedding"].shape == (n, IMAGE_DIM), d["image_embedding"].shape
    assert d["audio_embedding"].shape == (n, AUDIO_DIM)
    assert d["context"].shape == (n, CONTEXT_DIM)
    assert d["modality_mask"].shape == (n, 3)
    assert d["vibe_label"].dtype in (np.int32, np.int64)
    if n:
        assert d["vibe_label"].min() >= 0
        assert d["vibe_label"].max() < NUM_VIBES
        # masked modalities zeroed
        m = d["modality_mask"]
        img = d["image_embedding"]
        aud = d["audio_embedding"]
        for i in range(n):
            if m[i, 0] == 0:
                assert np.allclose(img[i], 0), f"{split} row {i} image not zeroed"
            if m[i, 1] == 0:
                assert np.allclose(aud[i], 0), f"{split} row {i} audio not zeroed"
            assert m[i, 2] == 1.0, "time mask must be 1"
    groups: Set[str] = set()
    if "split_group" in d and n:
        groups = set(str(x) for x in d["split_group"].tolist())
    return {"n": n, "groups": groups, "labels": set(int(x) for x in d["vibe_label"].tolist())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()
    man = json.loads(args.manifest.read_text())
    base = args.manifest.parent
    results = {}
    for key, split in (
        ("train_npz", "train"),
        ("validation_npz", "validation"),
        ("test_npz", "test"),
    ):
        path = base / man[key]
        results[split] = validate_npz(path, split)
        print(f"{split}: n={results[split]['n']} labels={sorted(results[split]['labels'])}")

    # leakage
    g_train = results["train"]["groups"]
    g_val = results["validation"]["groups"]
    g_test = results["test"]["groups"]
    assert g_train.isdisjoint(g_val), "train/val session leakage"
    assert g_train.isdisjoint(g_test), "train/test session leakage"
    assert g_val.isdisjoint(g_test), "val/test session leakage"

    if results["train"]["n"] > 0 and len(results["train"]["labels"]) < NUM_VIBES:
        print(
            f"WARNING: train missing class ids: "
            f"{set(range(NUM_VIBES)) - results['train']['labels']}"
        )
    print("OK: dataset valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
