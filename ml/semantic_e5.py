"""
Multilingual semantic embeddings for captions / journal text.

Default model: intfloat/multilingual-e5-large-instruct → 1024-D
(Use with memory_semantic_embeddings.embedding vector(1024).)

E5-instruct retrieval prefixes (required for good quality):
  query:   "Instruct: Given a web search query, retrieve relevant passages\\nQuery: {text}"
  or short: "query: {text}"  for e5-large without instruct variants
  passage: "passage: {text}"

This is a *different space* from fusion_v0 perceptual 128-D. Never mix them.

Example:
  from ml.semantic_e5 import E5Embedder
  e = E5Embedder()
  v = e.embed_passage("café in Lisbon at dusk")
  assert len(v) == 1024
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Sequence

import numpy as np

# 1024 for e5-large / e5-large-instruct; 768 for e5-base
DEFAULT_MODEL = "intfloat/multilingual-e5-large-instruct"
DEFAULT_DIM = 1024


class E5Embedder:
    """Thin wrapper around sentence-transformers E5 (float32; INT8 via ONNX later)."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: Optional[str] = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "pip install sentence-transformers  # for E5 semantic embeddings"
            ) from exc
        self.model_id = model_id
        self.model = SentenceTransformer(model_id, device=device)
        # Most E5 large models report 1024
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        """Batch embed documents/captions (passage prefix)."""
        prepared = [self._passage(t) for t in texts]
        emb = self.model.encode(
            prepared,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return emb.astype(np.float32)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        prepared = [self._query(t) for t in texts]
        emb = self.model.encode(
            prepared,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return emb.astype(np.float32)

    def embed_passage(self, text: str) -> List[float]:
        return self.embed_passages([text])[0].tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_queries([text])[0].tolist()

    def _passage(self, text: str) -> str:
        t = (text or "").strip()
        if "instruct" in self.model_id.lower():
            # instruct models: passage without long instruct prefix is OK;
            # still use passage: for symmetry with e5 family
            return f"passage: {t}"
        return f"passage: {t}"

    def _query(self, text: str) -> str:
        t = (text or "").strip()
        if "instruct" in self.model_id.lower():
            return (
                "Instruct: Retrieve journal memories that match this description\n"
                f"Query: {t}"
            )
        return f"query: {t}"


def upsert_sql_literal(vec: Sequence[float]) -> str:
    """Format for psql: '[0.1,0.2,...]'::vector"""
    inner = ",".join(f"{float(x):.8f}" for x in vec)
    return f"'[{inner}]'::vector"


def main() -> None:
    p = argparse.ArgumentParser(description="Smoke-test E5 embedder")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--text", default="rainy park bench in Lisbon")
    args = p.parse_args()
    e = E5Embedder(model_id=args.model)
    v = e.embed_passage(args.text)
    print(f"model={e.model_id} dim={len(v)} first5={v[:5]}")


if __name__ == "__main__":
    main()
