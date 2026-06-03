"""In-process sentence-transformers embedders — drop-in for nvidia_client's embed_*.

Mirrors the interface of rag_api/nvidia_client.py so the rest of the pipeline can
use either embedder by duck typing (no shared base class):

    embed_passages(texts: list[str]) -> (N, dim) float32, L2-normalized
    embed_query(text: str)          -> (1, dim) float32, L2-normalized

CPU-only by design (device="cpu"): the dev laptop and Render's free tier have no
GPU. Models cache to ~/.cache/huggingface/ on first use (left at the default).
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

# Per-model load options + query/passage instruction prefixes.
#
#   E5 (intfloat/e5-small-v2) is trained with asymmetric "query:" / "passage:"
#   prefixes and degrades noticeably for retrieval without them (per the
#   intfloat/e5 model card), so we apply them.
#
#   BGE-small and GTE-small are used here without prefixes: their model cards
#   only define an optional *query-side* instruction for retrieval, and applying
#   it asymmetrically is an experiment knob we deliberately leave off to keep
#   query/passage embeddings symmetric across the three configs.
#
#   Alibaba-NLP/gte-small-en-v1.5 ships custom modeling code, so it needs
#   trust_remote_code=True to load via sentence-transformers.
_MODEL_CONFIG: dict[str, dict] = {
    "intfloat/e5-small-v2": {
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "trust_remote_code": False,
    },
    "BAAI/bge-small-en-v1.5": {
        "query_prefix": "",
        "passage_prefix": "",
        "trust_remote_code": False,
    },
    "Alibaba-NLP/gte-small-en-v1.5": {
        "query_prefix": "",
        "passage_prefix": "",
        "trust_remote_code": True,
    },
}

_DEFAULT_CONFIG = {"query_prefix": "", "passage_prefix": "", "trust_remote_code": False}


class LocalEmbedder:
    """sentence-transformers embedder matching nvidia_client's embed_* interface."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        cfg = _MODEL_CONFIG.get(model_name, _DEFAULT_CONFIG)
        self._query_prefix: str = cfg["query_prefix"]
        self._passage_prefix: str = cfg["passage_prefix"]
        self.model = SentenceTransformer(
            model_name,
            device="cpu",
            trust_remote_code=cfg["trust_remote_code"],
        )

    @property
    def dim(self) -> int:
        """Embedding dimension (e.g. 384 for the small models)."""
        return int(self.model.get_sentence_embedding_dimension())

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Embed documents/passages. Returns (N, dim) float32, L2-normalized."""
        inputs = [self._passage_prefix + t for t in texts]
        vecs = self.model.encode(
            inputs, normalize_embeddings=True, convert_to_numpy=True, device="cpu"
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query. Returns (1, dim) float32, L2-normalized."""
        inputs = [self._query_prefix + text]
        vecs = self.model.encode(
            inputs, normalize_embeddings=True, convert_to_numpy=True, device="cpu"
        )
        return np.asarray(vecs, dtype=np.float32)
