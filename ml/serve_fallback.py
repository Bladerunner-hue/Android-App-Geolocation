"""
Hybrid inference: TensorFlow SavedModel / in-process MoE first,
fallback to Ollama (local) or Grok API for rich CoT captions.

Wire this into FastAPI as /vision/analyze.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.config import VIBE_LABELS  # noqa: E402

# Optional TF — allow fallback-only mode on machines without GPU/TF
try:
    import tensorflow as tf

    _HAS_TF = True
except ImportError:
    _HAS_TF = False


@dataclass
class AnalyzeResult:
    source: str  # "moe" | "ollama" | "grok" | "rules"
    vibe: str
    vibe_confidence: float
    cot_steps: List[str]
    caption: str
    insight_embedding: Optional[List[float]] = None
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "vibe": self.vibe,
            "vibe_confidence": self.vibe_confidence,
            "cot_steps": self.cot_steps,
            "caption": self.caption,
            "insight_embedding": self.insight_embedding,
        }


COT_SLOT_NAMES = ["scene", "sound", "geo_context", "valence"]


class GeoAIServing:
    def __init__(
        self,
        savedmodel_dir: Optional[Path] = None,
        prefer_moe: bool = True,
        ollama_model: str = "llama3.2",
        ollama_url: str = "http://127.0.0.1:11434",
        grok_api_key: Optional[str] = None,
        grok_model: str = "grok-2-latest",
    ):
        self.prefer_moe = prefer_moe
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url.rstrip("/")
        self.grok_api_key = grok_api_key or os.getenv("XAI_API_KEY") or os.getenv(
            "GROK_API_KEY"
        )
        self.grok_model = grok_model
        self._fn = None
        if prefer_moe and _HAS_TF and savedmodel_dir and Path(savedmodel_dir).exists():
            model = tf.saved_model.load(str(savedmodel_dir))
            self._fn = model.signatures.get("serving_default") or model.serving_default
            print(f"Loaded SavedModel from {savedmodel_dir}")
        elif prefer_moe and _HAS_TF:
            # Optional experimental MoE (not production fusion_v0).
            try:
                from ml.experiments.moe_kickstart import build_geoai_moe, setup_runtime

                setup_runtime(mixed_precision=False)
                self._model = build_geoai_moe(hidden=128, num_blocks=1)
                self._fn = "inprocess"
                print("Using in-process experimental GeoAIMoE (not fusion_v0)")
            except Exception as e:
                print(f"Experimental MoE unavailable (OK for production spine): {e}")
                self._model = None
        else:
            self._model = None

    def _moe_predict(
        self,
        image: np.ndarray,
        audio_mel: np.ndarray,
        geo: np.ndarray,
    ) -> Optional[AnalyzeResult]:
        if self._fn is None and not hasattr(self, "_model"):
            return None
        try:
            if self._fn == "inprocess":
                out = self._model.infer(
                    tf.constant(image),
                    tf.constant(audio_mel),
                    tf.constant(geo),
                )
                vibe_id = int(out["vibe_id"][0].numpy())
                conf = float(out["vibe_prob"][0, vibe_id].numpy())
                cot_ids = out["cot_ids"][0].numpy().tolist()
                emb = out["insight_embedding"][0].numpy().tolist()
            else:
                res = self._fn(
                    image=tf.constant(image),
                    audio_mel=tf.constant(audio_mel),
                    geo=tf.constant(geo),
                )
                # Signature may nest differently
                vibe_id = int(np.array(res["vibe_id"])[0])
                probs = np.array(res["vibe_prob"])[0]
                conf = float(probs[vibe_id])
                cot_ids = np.array(res["cot_ids"])[0].tolist()
                emb = np.array(res["insight_embedding"])[0].tolist()

            vibe = VIBE_LABELS[vibe_id] if 0 <= vibe_id < len(VIBE_LABELS) else "unknown"
            cot_steps = [
                f"{COT_SLOT_NAMES[i]}={cot_ids[i]}" for i in range(min(4, len(cot_ids)))
            ]
            caption = (
                f"This place feels {vibe} (p={conf:.2f}). "
                f"Reasoning: {'; '.join(cot_steps)}."
            )
            return AnalyzeResult(
                source="moe",
                vibe=vibe,
                vibe_confidence=conf,
                cot_steps=cot_steps,
                caption=caption,
                insight_embedding=emb,
            )
        except Exception as e:
            print(f"MoE inference failed: {e}")
            return None

    def _build_prompt(
        self,
        lat: float,
        lon: float,
        image_tags: Optional[List[str]] = None,
        audio_summary: Optional[str] = None,
        hour: Optional[int] = None,
    ) -> str:
        tags = ", ".join(image_tags or ["unknown scene"])
        audio = audio_summary or "ambient audio not classified"
        when = f"hour={hour}" if hour is not None else "time unknown"
        return (
            "You are a privacy-first personal geo memory assistant.\n"
            "Reason step-by-step (Chain of Thought) then give a short journal caption.\n"
            f"Location: lat={lat:.5f}, lon={lon:.5f}, {when}.\n"
            f"Image tags: {tags}.\n"
            f"Sound vibe: {audio}.\n"
            "Vibe classes: serene, energetic, chaotic, nostalgic, tense, social, contemplative.\n"
            "Reply JSON only: "
            '{"vibe":"...","cot_steps":["..."],"caption":"..."}\n'
        )

    def _ollama(self, prompt: str) -> Optional[AnalyzeResult]:
        body = json.dumps(
            {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.ollama_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data.get("response", "{}")
            parsed = json.loads(text) if isinstance(text, str) else text
            return AnalyzeResult(
                source="ollama",
                vibe=str(parsed.get("vibe", "contemplative")),
                vibe_confidence=0.55,
                cot_steps=list(parsed.get("cot_steps", [])),
                caption=str(parsed.get("caption", text)),
                raw=parsed,
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as e:
            print(f"Ollama fallback failed: {e}")
            return None

    def _grok(self, prompt: str) -> Optional[AnalyzeResult]:
        if not self.grok_api_key:
            return None
        body = json.dumps(
            {
                "model": self.grok_model,
                "messages": [
                    {"role": "system", "content": "Return strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.4,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            "https://api.x.ai/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.grok_api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"]
            # Strip markdown fences if present
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
                text = text.rsplit("```", 1)[0]
            parsed = json.loads(text)
            return AnalyzeResult(
                source="grok",
                vibe=str(parsed.get("vibe", "contemplative")),
                vibe_confidence=0.65,
                cot_steps=list(parsed.get("cot_steps", [])),
                caption=str(parsed.get("caption", text)),
                raw=parsed,
            )
        except Exception as e:
            print(f"Grok fallback failed: {e}")
            return None

    @staticmethod
    def _rules(
        lat: float,
        lon: float,
        audio_summary: Optional[str] = None,
    ) -> AnalyzeResult:
        """Always-on deterministic fallback — never blank UX."""
        vibe = "serene"
        if audio_summary and "crowd" in audio_summary.lower():
            vibe = "energetic"
        elif audio_summary and "traffic" in audio_summary.lower():
            vibe = "tense"
        cot = [
            f"geo=({lat:.3f},{lon:.3f})",
            f"sound={audio_summary or 'n/a'}",
            "model=rules",
            f"valence={vibe}",
        ]
        return AnalyzeResult(
            source="rules",
            vibe=vibe,
            vibe_confidence=0.35,
            cot_steps=cot,
            caption=f"Memory near ({lat:.4f}, {lon:.4f}) — provisional vibe: {vibe}.",
        )

    def analyze(
        self,
        *,
        image: Optional[np.ndarray] = None,
        audio_mel: Optional[np.ndarray] = None,
        geo: Optional[np.ndarray] = None,
        lat: float = 0.0,
        lon: float = 0.0,
        image_tags: Optional[List[str]] = None,
        audio_summary: Optional[str] = None,
        hour: Optional[int] = None,
    ) -> AnalyzeResult:
        # 1) Custom MoE
        if self.prefer_moe and image is not None and audio_mel is not None and geo is not None:
            # Ensure batch dim
            if image.ndim == 3:
                image = image[None, ...]
            if audio_mel.ndim == 3:
                audio_mel = audio_mel[None, ...]
            if geo.ndim == 1:
                geo = geo[None, ...]
            hit = self._moe_predict(
                image.astype(np.float32),
                audio_mel.astype(np.float32),
                geo.astype(np.float32),
            )
            if hit is not None:
                return hit

        prompt = self._build_prompt(lat, lon, image_tags, audio_summary, hour)

        # 2) Ollama local
        hit = self._ollama(prompt)
        if hit is not None:
            return hit

        # 3) Grok cloud
        hit = self._grok(prompt)
        if hit is not None:
            return hit

        # 4) Rules
        return self._rules(lat, lon, audio_summary)


def demo() -> None:
    svc = GeoAIServing(prefer_moe=True)
    # Synthetic tensors
    image = np.random.rand(1, 224, 224, 3).astype(np.float32)
    audio = np.random.randn(1, 96, 64, 1).astype(np.float32) * 0.1
    geo = np.zeros((1, 8), dtype=np.float32)
    geo[0, 0], geo[0, 1] = 0.4, -0.2
    r = svc.analyze(
        image=image,
        audio_mel=audio,
        geo=geo,
        lat=48.8566,
        lon=2.3522,
        image_tags=["cafe", "street"],
        audio_summary="soft chatter, cups",
        hour=18,
    )
    print(json.dumps(r.to_dict(), indent=2))


if __name__ == "__main__":
    demo()
