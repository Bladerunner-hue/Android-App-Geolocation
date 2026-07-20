"""
Semantic embeddings via the direct OpenAI-compatible E5 service.

Default (live probe confirmed):
  POST http://127.0.0.1:6100/v1/embeddings
  model: intfloat/e5-large-v2
  dim:   1024

Health: GET http://127.0.0.1:6100/health

E5 prefixes (required):
  query:   "query: {text}"
  passage: "passage: {text}"

This is a different space from fusion_v0 perceptual 128-D. Never mix them.
Store under memory_semantic_embeddings (vector 1024) with model_id set.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from typing import List, Optional, Sequence, Union

import numpy as np

DEFAULT_BASE_URL = os.environ.get("GEO_E5_BASE_URL", "http://127.0.0.1:6100")
DEFAULT_MODEL = os.environ.get("GEO_SEMANTIC_MODEL", "intfloat/e5-large-v2")
DEFAULT_DIM = 1024


class E5Embedder:
    """HTTP client for the direct E5 service (OpenAI embeddings API shape)."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model_id: str = DEFAULT_MODEL,
        timeout_s: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.timeout_s = timeout_s
        self.dim = DEFAULT_DIM
        self._probe_health()

    def _probe_health(self) -> None:
        url = f"{self.base_url}/health"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"E5 service not reachable at {url}. "
                f"Expected health JSON with dim=1024. ({exc})"
            ) from exc
        if body.get("status") != "ok":
            raise RuntimeError(f"E5 health not ok: {body}")
        dim = int(body.get("dim") or 0)
        if dim and dim != DEFAULT_DIM:
            raise RuntimeError(f"E5 dim={dim}, schema expects {DEFAULT_DIM}")
        if body.get("model"):
            # Prefer live model name when service reports it
            self.model_id = str(body["model"])
        self.dim = dim or DEFAULT_DIM

    def _post_embeddings(self, inputs: Union[str, List[str]]) -> np.ndarray:
        url = f"{self.base_url}/v1/embeddings"
        payload = json.dumps({"model": self.model_id, "input": inputs}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"E5 embeddings HTTP {exc.code}: {detail}") from exc

        data = body.get("data") or []
        if not data:
            raise RuntimeError(f"E5 empty data: {body!r}")
        # OpenAI shape: list of {index, embedding}
        data_sorted = sorted(data, key=lambda x: int(x.get("index", 0)))
        mat = np.asarray([row["embedding"] for row in data_sorted], dtype=np.float32)
        if mat.ndim != 2 or mat.shape[1] != self.dim:
            raise RuntimeError(f"Unexpected embedding shape {mat.shape}, want (*, {self.dim})")
        return mat

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        prepared = [self._passage(t) for t in texts]
        return self._post_embeddings(list(prepared))

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        prepared = [self._query(t) for t in texts]
        return self._post_embeddings(list(prepared))

    def embed_passage(self, text: str) -> List[float]:
        return self.embed_passages([text])[0].tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_queries([text])[0].tolist()

    def _passage(self, text: str) -> str:
        t = (text or "").strip()
        if t.lower().startswith("passage:") or t.lower().startswith("query:"):
            return t
        return f"passage: {t}"

    def _query(self, text: str) -> str:
        t = (text or "").strip()
        if t.lower().startswith("query:") or t.lower().startswith("passage:"):
            return t
        return f"query: {t}"


def main() -> None:
    p = argparse.ArgumentParser(description="Smoke-test direct E5 HTTP embedder")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--text", default="café in Lisbon at dusk")
    args = p.parse_args()
    e = E5Embedder(base_url=args.base_url, model_id=args.model)
    v = e.embed_passage(args.text)
    q = e.embed_query(args.text)
    print(f"model={e.model_id} dim={len(v)} passage_first3={v[:3]}")
    print(f"query_first3={q[:3]}")


if __name__ == "__main__":
    main()
