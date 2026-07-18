"""
Build leakage-safe NPZ splits for fusion_v0 from a memories JSONL export.

Input JSONL rows (one memory + training label per line):
  {
    "sample_id": "...",
    "session_id": "...",
    "photo_path": "optional path",
    "audio_path": "optional path",
    "primary_vibe": "serene",
    "label_source": "human_self",
    "consent_for_training": true,
    "captured_at_utc": "2026-07-01T12:00:00Z",
    "utc_offset_minutes": 120,
    "latitude": 48.85,
    "longitude": 2.35,
    "accuracy_m": 12.0,
    "sample_weight": 1.0
  }

Never fabricates vibes for public sound datasets — those belong in sound bootstrap.

Usage:
  python -m ml.prepare_fusion_dataset \\
    --input data/personal/train_mode_export.jsonl \\
    --out-dir /secure/geojournal/fusion-v0 \\
    --skip-encoders   # if embeddings already in row as lists
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.context12 import CONTEXT_REVISION, context12, modality_mask  # noqa: E402
from ml.fusion_v0 import AUDIO_DIM, IMAGE_DIM, VIBE_LABELS  # noqa: E402

VIBE_TO_ID = {v: i for i, v in enumerate(VIBE_LABELS)}


def _parse_ts(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def session_hash(session_id: str) -> str:
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]


def temporal_session_splits(
    rows: List[Dict[str, Any]],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Dict[str, str]:
    """
    Group by session_id, sort sessions by min capture time, assign
    oldest 70% train / next 15% val / latest 15% test.
    Returns session_id -> split.
    """
    sessions: Dict[str, datetime] = {}
    for r in rows:
        sid = str(r["session_id"])
        ts = _parse_ts(str(r["captured_at_utc"]))
        if sid not in sessions or ts < sessions[sid]:
            sessions[sid] = ts
    ordered = sorted(sessions.items(), key=lambda kv: kv[1])
    n = len(ordered)
    if n == 0:
        return {}
    n_train = max(1, int(n * train_frac))
    n_val = max(0, int(n * val_frac))
    if n_train + n_val >= n and n > 1:
        n_val = max(0, n - n_train - 1)
    assign: Dict[str, str] = {}
    for i, (sid, _) in enumerate(ordered):
        if i < n_train:
            assign[sid] = "train"
        elif i < n_train + n_val:
            assign[sid] = "validation"
        else:
            assign[sid] = "test"
    # ensure at least one test if possible
    if n >= 3 and "test" not in assign.values():
        assign[ordered[-1][0]] = "test"
    return assign


def extract_features(
    rows: List[Dict[str, Any]],
    skip_encoders: bool = False,
) -> List[Dict[str, Any]]:
    image_enc = None
    audio_enc = None
    if not skip_encoders:
        from ml.encoders import AudioEncoder, ImageEncoder, zero_audio, zero_image

        image_enc = ImageEncoder()
        try:
            audio_enc = AudioEncoder()
        except Exception as e:
            print(f"WARNING: YAMNet unavailable ({e}); audio will be zeros")
            audio_enc = None
    else:
        from ml.encoders import zero_audio, zero_image

    out = []
    for r in rows:
        if not r.get("consent_for_training", False):
            continue
        vibe = r.get("primary_vibe")
        if vibe not in VIBE_TO_ID:
            continue
        label_source = str(r.get("label_source", ""))
        if not label_source.startswith("human"):
            # refuse non-human vibe labels for personal classifier
            continue

        has_photo = bool(r.get("photo_path") or r.get("image_embedding"))
        has_audio = bool(r.get("audio_path") or r.get("audio_embedding"))

        if r.get("image_embedding") is not None:
            img = np.asarray(r["image_embedding"], dtype=np.float32)
        elif has_photo and image_enc is not None:
            img = image_enc.embed_path(Path(r["photo_path"]))
        else:
            img = zero_image()
            has_photo = False

        if r.get("audio_embedding") is not None:
            aud = np.asarray(r["audio_embedding"], dtype=np.float32)
        elif has_audio and audio_enc is not None:
            try:
                aud = audio_enc.embed_path(Path(r["audio_path"]))
            except Exception as e:
                print(f"audio fail {r.get('sample_id')}: {e}")
                aud = zero_audio()
                has_audio = False
        else:
            aud = zero_audio()
            has_audio = False

        if img.shape != (IMAGE_DIM,) or aud.shape != (AUDIO_DIM,):
            raise ValueError(f"Bad emb shape sample={r.get('sample_id')}")

        mask = modality_mask(has_photo, has_audio)
        img = img * mask[0]
        aud = aud * mask[1]
        ctx = context12(
            _parse_ts(str(r["captured_at_utc"])),
            int(r.get("utc_offset_minutes", 0)),
            r.get("latitude"),
            r.get("longitude"),
            r.get("accuracy_m"),
        )
        out.append(
            {
                **r,
                "image_embedding": img,
                "audio_embedding": aud,
                "context": ctx,
                "modality_mask": mask,
                "vibe_label": VIBE_TO_ID[vibe],
                "sample_weight": float(r.get("sample_weight", 1.0)),
                "split_group": session_hash(str(r["session_id"])),
            }
        )
    return out


def stack_split(rows: List[Dict[str, Any]], path: Path) -> int:
    if not rows:
        # empty split with correct shapes
        np.savez_compressed(
            path,
            image_embedding=np.zeros((0, IMAGE_DIM), np.float32),
            audio_embedding=np.zeros((0, AUDIO_DIM), np.float32),
            context=np.zeros((0, 12), np.float32),
            modality_mask=np.zeros((0, 3), np.float32),
            vibe_label=np.zeros((0,), np.int32),
            sample_weight=np.zeros((0,), np.float32),
            sample_id=np.asarray([], dtype="U64"),
            split_group=np.asarray([], dtype="U64"),
            label_source=np.asarray([], dtype="U24"),
        )
        return 0
    np.savez_compressed(
        path,
        image_embedding=np.stack([r["image_embedding"] for r in rows]).astype(np.float32),
        audio_embedding=np.stack([r["audio_embedding"] for r in rows]).astype(np.float32),
        context=np.stack([r["context"] for r in rows]).astype(np.float32),
        modality_mask=np.stack([r["modality_mask"] for r in rows]).astype(np.float32),
        vibe_label=np.asarray([r["vibe_label"] for r in rows], dtype=np.int32),
        sample_weight=np.asarray([r["sample_weight"] for r in rows], dtype=np.float32),
        sample_id=np.asarray([str(r.get("sample_id", "")) for r in rows], dtype="U64"),
        split_group=np.asarray([r["split_group"] for r in rows], dtype="U64"),
        label_source=np.asarray(
            [str(r.get("label_source", "")) for r in rows], dtype="U24"
        ),
    )
    return len(rows)


def write_manifest(
    out_dir: Path,
    counts: Dict[str, int],
    encoder_meta: Dict[str, Any],
) -> Path:
    tax_hash = hashlib.sha256(json.dumps(VIBE_LABELS).encode()).hexdigest()
    manifest = {
        "schema_version": 1,
        "split_strategy": "per_user_grouped_session_temporal_v1",
        "train_npz": "splits/train.npz",
        "validation_npz": "splits/validation.npz",
        "test_npz": "splits/test.npz",
        "counts": counts,
        "extractors": encoder_meta,
        "context_revision": CONTEXT_REVISION,
        "taxonomy_revision": f"sha256:{tax_hash}",
        "vibe_labels": list(VIBE_LABELS),
        "model_spec": {
            "name": "fusion_v0",
            "image_dim": IMAGE_DIM,
            "audio_dim": AUDIO_DIM,
        },
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--skip-encoders",
        action="store_true",
        help="Use precomputed image_embedding/audio_embedding lists in JSONL",
    )
    args = ap.parse_args()

    raw = load_jsonl(args.input)
    print(f"loaded {len(raw)} rows")
    feats = extract_features(raw, skip_encoders=args.skip_encoders)
    print(f"accepted {len(feats)} human-consent training rows")
    if not feats:
        print("ERROR: no rows accepted", file=sys.stderr)
        return 1

    assign = temporal_session_splits(feats)
    for r in feats:
        r["split"] = assign[str(r["session_id"])]

    # disjoint session assertion
    by_split: Dict[str, set] = {"train": set(), "validation": set(), "test": set()}
    for r in feats:
        by_split[r["split"]].add(r["split_group"])
    assert by_split["train"].isdisjoint(by_split["validation"])
    assert by_split["train"].isdisjoint(by_split["test"])
    assert by_split["validation"].isdisjoint(by_split["test"])

    splits_dir = args.out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "validation", "test"):
        part = [r for r in feats if r["split"] == split]
        n = stack_split(part, splits_dir / f"{split if split != 'validation' else 'validation'}.npz")
        # file names: train.npz, validation.npz, test.npz
        counts[split] = n
        print(f"{split}: {n}")

    # ensure train has all classes if possible
    train_labels = {r["vibe_label"] for r in feats if r["split"] == "train"}
    if len(train_labels) < len(VIBE_LABELS):
        missing = [VIBE_LABELS[i] for i in range(len(VIBE_LABELS)) if i not in train_labels]
        print(f"WARNING: train missing classes: {missing}")

    enc_meta = {
        "image": {
            "id": "keras/MobileNetV3Small-ImageNet-224-pool-avg",
            "revision": "pretrained-imagenet",
        },
        "audio": {
            "id": "tfhub/google/yamnet/1-mean-pool",
            "revision": "tfhub-google-yamnet-1",
        },
    }
    if args.skip_encoders:
        enc_meta["note"] = "embeddings provided in JSONL; extractors not re-run"

    man = write_manifest(args.out_dir, counts, enc_meta)
    print(f"manifest → {man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
