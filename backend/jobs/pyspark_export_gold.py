"""
PySpark Gold export — planned data plane for GeoJournal fusion_v0.

Medallion:
  Bronze: raw app events / Train Mode export (paths, geo, human vibes, consent)
  Silver: session windows, dwell, license-safe sound-bootstrap joins
  Gold:   training-ready rows + deterministic session-temporal split metadata

Hard rules:
  - **PySpark entry** (not Scala-first). Optional Scala later only if needed.
  - No TensorFlow imports in this ETL module.
  - Never collect photo/audio bytes to the driver.
  - Gold feeds ml.prepare_fusion_dataset / NPZ manifests; TF train stays in pyenv.
  - Public sound rows are sound-event only — never fabricated vibe labels.

See docs/CONFIRMATION.md for accepted feedback + status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


def stable_split(sample_id: str, train=0.90, val=0.07) -> str:
    """Deterministic train/val/test from sample id hash (no random seed drift)."""
    h = int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest(), 16) % 10_000
    r = h / 10_000.0
    if r < train:
        return "train"
    if r < train + val:
        return "val"
    return "test"


def class_weights_sqrt_inv(counts: Dict[str, int], num_classes: int = 7) -> Dict[str, float]:
    inv = {}
    for i in range(num_classes):
        c = float(counts.get(str(i), 1) or 1)
        inv[str(i)] = 1.0 / (c ** 0.5)
    s = sum(inv.values())
    return {k: v * num_classes / s for k, v in inv.items()}


def run_local_parquet_stub(
    bronze_jsonl: Path,
    gold_dir: Path,
) -> Dict[str, Any]:
    """
    Pure-Python stub when Spark is not installed — same schema as Spark Gold.
    Swap for SparkSession path below when cluster/local Spark is available.
    """
    gold_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    if bronze_jsonl.exists():
        for line in bronze_jsonl.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    else:
        # Demo bronze rows
        for i in range(50):
            rows.append(
                {
                    "sample_id": f"mem_{i:04d}",
                    "user_id": "demo",
                    "lat": 48.85 + (i % 10) * 0.01,
                    "lon": 2.35 + (i % 7) * 0.01,
                    "hour": (i * 3) % 24,
                    "vibe": i % 7,
                    "photo_path": f"s3://bucket/photos/{i}.jpg",
                    "audio_path": f"s3://bucket/audio/{i}.wav",
                    "dwell_sec": 60 + i * 3,
                }
            )

    # Silver-ish features (would be window/SQL in Spark)
    silver = []
    for r in rows:
        hour = float(r.get("hour", 12))
        import math

        ang = 2 * math.pi * hour / 24.0
        sid = str(r["sample_id"])
        silver.append(
            {
                "sample_id": sid,
                "user_id": r.get("user_id", "unknown"),
                "lat_norm": float(r["lat"]) / 90.0,
                "lon_norm": float(r["lon"]) / 180.0,
                "hour_sin": math.sin(ang),
                "hour_cos": math.cos(ang),
                "dwell_sec": float(r.get("dwell_sec", 0)),
                "vibe": int(r.get("vibe", 0)),
                "photo_path": r.get("photo_path"),
                "audio_path": r.get("audio_path"),
                "split": stable_split(sid),
            }
        )

    counts: Dict[str, int] = {}
    for r in silver:
        k = str(r["vibe"])
        counts[k] = counts.get(k, 0) + 1

    out_json = gold_dir / "gold_rows.jsonl"
    with out_json.open("w") as f:
        for r in silver:
            f.write(json.dumps(r) + "\n")

    manifest = {
        "schema_version": 1,
        "rows": len(silver),
        "class_counts": counts,
        "class_weights": class_weights_sqrt_inv(counts),
        "splits": {
            s: sum(1 for r in silver if r["split"] == s)
            for s in ("train", "val", "test")
        },
        "gold_path": str(out_json),
        "note": (
            "Parquet/TFRecord writer should run executor-side. "
            "Do not collect image/audio bytes to the driver."
        ),
    }
    (gold_dir / "manifest_gold.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return manifest


def run_spark(bronze_path: str, gold_path: str) -> None:
    """
    Real Spark path (requires pyspark). Pure DataFrame/SQL — no pandas UDFs.
    """
    from pyspark.sql import SparkSession, Window
    from pyspark.sql import functions as F

    spark = (
        SparkSession.builder.appName("geoai-gold-export")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    bronze = spark.read.json(bronze_path)

    # Trajectory-ish features: visits per place bucket
    w = Window.partitionBy("user_id").orderBy("hour")
    silver = (
        bronze.withColumn("lat_norm", F.col("lat") / F.lit(90.0))
        .withColumn("lon_norm", F.col("lon") / F.lit(180.0))
        .withColumn("hour_sin", F.sin(F.col("hour") * F.lit(2 * 3.14159265 / 24)))
        .withColumn("hour_cos", F.cos(F.col("hour") * F.lit(2 * 3.14159265 / 24)))
        .withColumn("visit_idx", F.row_number().over(w))
        .withColumn(
            "split_hash",
            F.xxhash64(F.col("sample_id")) % F.lit(10000),
        )
        .withColumn(
            "split",
            F.when(F.col("split_hash") < 9000, "train")
            .when(F.col("split_hash") < 9700, "val")
            .otherwise("test"),
        )
    )
    silver.persist()
    silver.write.mode("overwrite").parquet(gold_path)

    counts = (
        silver.groupBy("vibe")
        .count()
        .collect()
    )
    class_counts = {str(r["vibe"]): int(r["count"]) for r in counts}
    manifest = {
        "schema_version": 1,
        "gold_path": gold_path,
        "class_counts": class_counts,
        "class_weights": class_weights_sqrt_inv(class_counts),
        "rows": silver.count(),
    }
    Path(gold_path).mkdir(parents=True, exist_ok=True)
    (Path(gold_path) / "manifest_gold.json").write_text(json.dumps(manifest, indent=2))
    print(manifest)
    spark.stop()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bronze", type=Path, default=Path("ml/data_sample/bronze_events.jsonl"))
    ap.add_argument("--gold", type=Path, default=Path("ml/data_sample/gold"))
    ap.add_argument("--spark", action="store_true", help="Use PySpark if installed")
    args = ap.parse_args()

    if args.spark:
        run_spark(str(args.bronze), str(args.gold))
    else:
        run_local_parquet_stub(args.bronze, args.gold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
