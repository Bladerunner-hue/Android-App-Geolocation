"""
Public sound data helpers for AUXILIARY sound-event recognition only.

Never map ESC/FSD labels into the 7-class personal vibe taxonomy.
ESC-50 full is CC BY-NC — exclude from commercial shipping weights.
ESC-10 (esc10==true) is CC BY. FSD50K: prefer CC0, then CC-BY after legal review.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_esc10_rows(esc50_root: Path) -> List[Dict[str, Any]]:
    """Select ESC-10 subset from meta/esc50.csv (preserve fold)."""
    meta = esc50_root / "meta" / "esc50.csv"
    rows = []
    with meta.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("esc10", "")).lower() not in {"true", "1", "yes"}:
                continue
            rows.append(
                {
                    "source_id": row["filename"],
                    "fold": int(row["fold"]),
                    "category": row["category"],
                    "target": int(row["target"]),
                    "license_id": "CC-BY",  # ESC-10
                    "audio_path": str(esc50_root / "audio" / row["filename"]),
                    "split_group": f"esc10_fold_{row['fold']}",
                    "task": "sound_event",
                    "note": "Benchmark only — not vibe ground truth",
                }
            )
    return rows


def allowed_fsd_license(raw: str) -> Optional[str]:
    value = raw.lower().replace("http://", "https://").rstrip("/")
    if "publicdomain/zero/1.0" in value or "cc0" in value:
        return "CC0-1.0"
    if "creativecommons.org/licenses/by/" in value and "/by-nc/" not in value:
        return "CC-BY"
    return None


def load_fsd50k_cc0_rows(root: Path, include_cc_by: bool = False) -> List[Dict[str, Any]]:
    info_path = root / "FSD50K.metadata" / "dev_clips_info_FSD50K.json"
    gt_path = root / "FSD50K.ground_truth" / "dev.csv"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    rows = []
    with gt_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            source_id = str(row["fname"])
            metadata = info[source_id]
            license_id = allowed_fsd_license(str(metadata["license"]))
            if license_id is None:
                continue
            if license_id == "CC-BY" and not include_cc_by:
                continue
            rows.append(
                {
                    "source_id": source_id,
                    "split": row["split"],
                    "labels": row["labels"].split(","),
                    "license_id": license_id,
                    "license_url": metadata["license"],
                    "creator": metadata.get("uploader"),
                    "source_url": f"https://freesound.org/s/{source_id}/",
                    "audio_path": str(root / "FSD50K.dev_audio" / f"{source_id}.wav"),
                    "task": "sound_event",
                    "note": "Auxiliary events only — never fabricate vibe labels",
                }
            )
    return rows


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="List public sound rows for bootstrap")
    ap.add_argument("--esc50-root", type=Path, default=None)
    ap.add_argument("--fsd50k-root", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--include-cc-by", action="store_true")
    args = ap.parse_args()
    all_rows: List[Dict[str, Any]] = []
    if args.esc50_root:
        all_rows.extend(load_esc10_rows(args.esc50_root))
    if args.fsd50k_root:
        all_rows.extend(
            load_fsd50k_cc0_rows(args.fsd50k_root, include_cc_by=args.include_cc_by)
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(all_rows)} sound-event rows → {args.out}")


if __name__ == "__main__":
    main()
