"""marimo notebook: fusion_v0 data validation + live-data prep.

Exploration / validation only — never the production train entry.
After a clean manifest is written, run:

  python -m ml.train_fusion_v0 --manifest <path> ...

Edit:  marimo edit ml/notebooks/validate_fusion_live.py
Run:   marimo run  ml/notebooks/validate_fusion_live.py
"""

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np

    try:
        import polars as pl
    except ImportError:
        pl = None

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    return Path, ROOT, json, mo, np, pl, sys


@app.cell
def _(Path, ROOT, mo):
    """UI params — change these → only downstream cells re-run."""
    gold_path = mo.ui.text(
        value=str(ROOT / "ml/data_sample/gold/gold_rows.jsonl"),
        label="Gold / Bronze JSONL path",
        full_width=True,
    )
    manifest_path = mo.ui.text(
        value=str(ROOT / "ml/data_sample/fusion_wiring/manifest.json"),
        label="Existing fusion manifest (optional)",
        full_width=True,
    )
    split_seed = mo.ui.slider(0, 100, value=42, label="Split seed", show_value=True)
    min_per_class = mo.ui.slider(1, 50, value=5, label="Min samples / class", show_value=True)
    mo.vstack(
        [
            mo.md("## fusion_v0 validation (marimo-reactive)"),
            mo.md(
                "Production train stays on CLI: `python -m ml.train_fusion_v0`. "
                "This notebook explores balance, missing modalities, and schema."
            ),
            gold_path,
            manifest_path,
            split_seed,
            min_per_class,
        ]
    )
    return gold_path, manifest_path, min_per_class, split_seed


@app.cell
def _(Path, gold_path, json, mo, pl):
    """Lazy load Gold/Bronze JSONL (Polars if available, else pure JSON)."""
    path = Path(gold_path.value)
    load_status = {"ok": False, "n": 0, "error": None, "rows": None, "df": None}

    if not path.is_file():
        load_status["error"] = f"File not found: {path}"
        mo.md(f"**Load:** missing `{path}` — drop a consented Bronze/Gold JSONL path above.")
    else:
        try:
            lines = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            load_status["rows"] = lines
            load_status["n"] = len(lines)
            load_status["ok"] = True
            if pl is not None and lines:
                load_status["df"] = pl.DataFrame(lines)
            mo.md(f"**Load OK:** `{path.name}` — **{load_status['n']}** rows")
        except Exception as e:  # noqa: BLE001
            load_status["error"] = str(e)
            mo.md(f"**Load error:** {e}")
    return (load_status,)


@app.cell
def _(load_status, mo, pl):
    """Reactive class-balance + missing-modality matrix."""
    if not load_status["ok"] or not load_status["rows"]:
        mo.md("_No rows — class balance skipped._")
        balance = None
        modality_stats = None
    else:
        rows = load_status["rows"]
        vibes = [r.get("primary_vibe") or r.get("vibe") for r in rows]
        consent = [bool(r.get("consent_for_training", False)) for r in rows]
        has_img = [
            bool(r.get("image_embedding") or r.get("image_path") or r.get("photo_path"))
            for r in rows
        ]
        has_aud = [
            bool(r.get("audio_embedding") or r.get("audio_path") or r.get("wav_path"))
            for r in rows
        ]
        has_geo = [
            r.get("latitude") is not None and r.get("longitude") is not None for r in rows
        ]

        if pl is not None:
            df = pl.DataFrame(
                {
                    "vibe": vibes,
                    "consent": consent,
                    "image": has_img,
                    "audio": has_aud,
                    "geo": has_geo,
                }
            )
            balance = (
                df.group_by("vibe")
                .agg(pl.len().alias("n"), pl.col("consent").sum().alias("consented"))
                .sort("n", descending=True)
            )
            modality_stats = {
                "image_frac": float(df["image"].mean()),
                "audio_frac": float(df["audio"].mean()),
                "geo_frac": float(df["geo"].mean()),
                "consent_frac": float(df["consent"].mean()),
            }
            mo.vstack(
                [
                    mo.md("### Class balance"),
                    mo.ui.table(balance),
                    mo.md(
                        f"### Modalities\n"
                        f"- image: **{modality_stats['image_frac']:.1%}**\n"
                        f"- audio: **{modality_stats['audio_frac']:.1%}**\n"
                        f"- geo: **{modality_stats['geo_frac']:.1%}**\n"
                        f"- consent_for_training: **{modality_stats['consent_frac']:.1%}**"
                    ),
                ]
            )
        else:
            from collections import Counter

            balance = Counter(vibes)
            modality_stats = {
                "image_frac": sum(has_img) / max(len(rows), 1),
                "audio_frac": sum(has_aud) / max(len(rows), 1),
                "geo_frac": sum(has_geo) / max(len(rows), 1),
                "consent_frac": sum(consent) / max(len(rows), 1),
            }
            mo.md(
                f"### Class balance (install polars for table)\n```\n{dict(balance)}\n```\n"
                f"### Modalities\n{modality_stats}"
            )
    return balance, modality_stats


@app.cell
def _(Path, json, manifest_path, mo, min_per_class):
    """Call production validate_fusion_dataset logic when a manifest exists."""
    man = Path(manifest_path.value)
    validation = {"ran": False, "pass": None, "detail": ""}

    if not man.is_file():
        mo.md(
            f"_No manifest at `{man}` — run prepare or wiring fixture first._\n\n"
            "```bash\n"
            "python -m ml.make_wiring_fixture\n"
            "python -m ml.validate_fusion_dataset --manifest ml/data_sample/fusion_wiring/manifest.json\n"
            "```"
        )
    else:
        try:
            from ml.validate_fusion_dataset import validate_npz

            meta = json.loads(man.read_text(encoding="utf-8"))
            base = man.parent
            details = []
            ok = True
            for key, split in (
                ("train_npz", "train"),
                ("validation_npz", "validation"),
                ("test_npz", "test"),
            ):
                if key not in meta:
                    details.append(f"missing key {key}")
                    ok = False
                    continue
                r = validate_npz(base / meta[key], split)
                details.append(f"{split}: n={r['n']} labels={sorted(r['labels'])}")
                if r["n"] and len(r["labels"]) < 2:
                    details.append(f"  warn: {split} has few classes")
            # min per class heuristic from train only
            train_key = meta.get("train_npz")
            if train_key:
                import numpy as np

                z = np.load(base / train_key, allow_pickle=False)
                labels = z["vibe_label"]
                for c in range(7):
                    n_c = int((labels == c).sum())
                    if n_c < int(min_per_class.value):
                        details.append(
                            f"  warn: class {c} has {n_c} < min_per_class={min_per_class.value}"
                        )
            validation = {"ran": True, "pass": ok, "detail": "\n".join(details)}
            status = "PASS" if ok else "FAIL"
            mo.md(f"### validate_fusion_dataset — **{status}**\n```\n{validation['detail']}\n```")
        except Exception as e:  # noqa: BLE001
            validation = {"ran": True, "pass": False, "detail": str(e)}
            mo.md(f"### Validation error\n```\n{e}\n```")
    return (validation,)


@app.cell
def _(ROOT, gold_path, mo, split_seed):
    """Prepare-training helper: show exact CLI the train script expects.

    Does not run training inside marimo — production loops stay on CLI/SLURM.
    """
    out_dir = ROOT / "ml" / "artifacts" / "marimo_prep"
    gold_in = gold_path.value
    cmd = f"""# 1) Prepare NPZ + manifest (from consented Gold only)
#    temporal/session splits are deterministic in prepare; seed is for train.
python -m ml.prepare_fusion_dataset \\
  --input {gold_in} \\
  --out-dir {out_dir}

# 2) Validate schema / leakage
python -m ml.validate_fusion_dataset --manifest {out_dir}/manifest.json

# 3) Train dense fusion_v0 (NOT inside marimo; seed={split_seed.value})
python -m ml.train_fusion_v0 \\
  --manifest {out_dir}/manifest.json \\
  --weights-out {out_dir}/fusion_v0.weights.h5 \\
  --seed {split_seed.value}

# 4) Export TFLite for Kotlin FusionV0Interpreter
python -m ml.export_tflite \\
  --weights {out_dir}/fusion_v0.weights.h5 \\
  --parity-data {out_dir}/eval/parity.npz \\
  --out {out_dir}/fusion_v0.tflite
"""
    mo.vstack(
        [
            mo.md("### One-click CLI (copy after filters look good)"),
            mo.md(f"```bash\n{cmd}\n```"),
            mo.md(
                "**Hard rules:** human Train Mode labels only · "
                "`consent_for_training=true` · never invent 7-class vibes from ESC/Places · "
                "single `SCHEMA_VERSION` · dense fusion_v0 is the release path."
            ),
        ]
    )
    return cmd, out_dir


@app.cell
def _(Path, ROOT, mo, np):
    """Optional: PCA of perceptual / image embeddings from a train NPZ (if present)."""
    candidates = [
        ROOT / "ml/data_sample/fusion_wiring/splits/train.npz",
        ROOT / "ml/artifacts/marimo_prep/splits/train.npz",
    ]
    plot_path = next((p for p in candidates if p.is_file()), None)
    if plot_path is None:
        mo.md("_No train NPZ yet for embedding projection — run prepare first._")
    else:
        z = np.load(plot_path, allow_pickle=False)
        X = z["image_embedding"].astype(np.float64)
        y = z["vibe_label"]
        # cheap PCA (2-D) without sklearn dependency
        Xc = X - X.mean(axis=0, keepdims=True)
        # covariance via SVD on sample
        n = min(len(Xc), 500)
        U, S, Vt = np.linalg.svd(Xc[:n], full_matrices=False)
        coords = Xc @ Vt[:2].T
        mo.md(
            f"### Image embedding PCA (2-D) — `{plot_path.name}` n={len(X)}\n"
            f"PC1 var share ≈ **{(S[0]**2 / (S**2).sum()):.1%}**, "
            f"PC2 ≈ **{(S[1]**2 / (S**2).sum()):.1%}**  \n"
            f"Label range: {int(y.min())}–{int(y.max())} · "
            f"coords shape {coords.shape}"
        )
    return (plot_path,)


if __name__ == "__main__":
    app.run()
