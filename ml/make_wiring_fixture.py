"""
Create a tiny synthetic *feature* NPZ fixture for pipeline wiring (not accuracy).

Uses deterministic random embeddings + context12 — never claims vibe truth.
Human labels are fake class IDs only for trainer/export smoke tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.context12 import context12, modality_mask  # noqa: E402
from ml.fusion_v0 import AUDIO_DIM, IMAGE_DIM, VIBE_LABELS  # noqa: E402
from datetime import datetime, timezone, timedelta


def main() -> int:
    out = Path("ml/data_sample/fusion_wiring")
    splits = out / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    def make_split(name: str, n: int, session_prefix: str):
        imgs, auds, ctxs, masks, labels, weights = [], [], [], [], [], []
        sids, groups, sources = [], [], []
        for i in range(n):
            vibe = i % len(VIBE_LABELS)
            has_aud = i % 3 != 0
            has_photo = True
            m = modality_mask(has_photo, has_aud)
            img = rng.normal(0, 1, IMAGE_DIM).astype(np.float32) * m[0]
            # class-correlated bias so training can move off chance
            img[vibe * 10 : vibe * 10 + 10] += 2.0
            aud = (
                rng.normal(0, 1, AUDIO_DIM).astype(np.float32) * m[1]
                if has_aud
                else np.zeros(AUDIO_DIM, np.float32)
            )
            if has_aud:
                aud[vibe * 5 : vibe * 5 + 5] += 1.5
            ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)
            c = context12(ts, 60, 48.0 + i * 0.01, 2.0, 10.0)
            imgs.append(img)
            auds.append(aud)
            ctxs.append(c)
            masks.append(m)
            labels.append(vibe)
            weights.append(1.0)
            sids.append(f"{name}_{i}")
            # sessions of 5
            groups.append(f"{session_prefix}_{i // 5}")
            sources.append("human_self")
        path = splits / f"{name}.npz"
        np.savez_compressed(
            path,
            image_embedding=np.stack(imgs),
            audio_embedding=np.stack(auds),
            context=np.stack(ctxs),
            modality_mask=np.stack(masks),
            vibe_label=np.asarray(labels, np.int32),
            sample_weight=np.asarray(weights, np.float32),
            sample_id=np.asarray(sids, dtype="U64"),
            split_group=np.asarray(groups, dtype="U64"),
            label_source=np.asarray(sources, dtype="U24"),
        )
        return path, n

    # disjoint session prefixes
    _, ntr = make_split("train", 140, "sess_tr")
    _, nva = make_split("validation", 35, "sess_va")
    _, nte = make_split("test", 35, "sess_te")
    # parity = first 8 train rows
    tr = np.load(splits / "train.npz")
    parity = out / "eval" / "parity.npz"
    parity.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        parity,
        image_embedding=tr["image_embedding"][:8],
        audio_embedding=tr["audio_embedding"][:8],
        context=tr["context"][:8],
        modality_mask=tr["modality_mask"][:8],
        vibe_label=tr["vibe_label"][:8],
    )
    man = {
        "schema_version": 1,
        "split_strategy": "per_user_grouped_session_temporal_v1",
        "train_npz": "splits/train.npz",
        "validation_npz": "splits/validation.npz",
        "test_npz": "splits/test.npz",
        "counts": {"train": ntr, "validation": nva, "test": nte},
        "context_revision": "context12-v1",
        "vibe_labels": list(VIBE_LABELS),
        "note": "WIRING FIXTURE ONLY — not an accuracy benchmark",
        "extractors": {"image": {"id": "synthetic"}, "audio": {"id": "synthetic"}},
        "model_spec": {},
    }
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    print(f"wiring fixture → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
