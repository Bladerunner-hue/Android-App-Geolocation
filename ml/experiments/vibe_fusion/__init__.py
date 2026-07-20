"""Legacy vibe_fusion shim — experimental only. Prefer ml.fusion_v0."""

from ml.experiments.vibe_fusion.model import VIBE_LABELS, build_vibe_fusion, presence_masks

__all__ = ["VIBE_LABELS", "build_vibe_fusion", "presence_masks"]
