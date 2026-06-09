"""Torch-free ONNX embedder — a drop-in for nvidia_client's / LocalEmbedder's
embed_* interface, used for the e5-small-v2 local path on a 512 MB host.

Runs an ONNX-exported intfloat/e5-small-v2 via onnxruntime, with the Rust
`tokenizers` lib (no torch, no transformers, no sentence-transformers). Replicates
sentence-transformers' e5 pipeline exactly — asymmetric "query: "/"passage: "
prefixes, masked mean-pooling, L2-normalization — verified to cosine 1.0000
(max abs diff ~6e-7) against sentence-transformers e5.

    embed_passages(texts) -> (N, 384) float32, L2-normalized
    embed_query(text)     -> (1, 384) float32, L2-normalized
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# Bundled export (committed under rag_api/onnx/, copied into the image).
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "onnx" / "e5-small-v2"
_E5_DIM = 384
_MAX_TOKENS = 512
# Texts per ONNX forward pass. Bounds peak memory: BERT attention is O(batch *
# seq^2), so embedding all ~700 chunks at once would allocate multi-GB activations.
_EMBED_BATCH = 16
# e5 is trained with asymmetric instruction prefixes (per the model card).
_PREFIX = {"query": "query: ", "passage": "passage: "}


class OnnxEmbedder:
    """onnxruntime + tokenizers embedder matching the embed_* interface."""

    def __init__(self, model_dir: Path | str = DEFAULT_MODEL_DIR,
                 model_name: str = "intfloat/e5-small-v2") -> None:
        model_dir = Path(model_dir)
        self.model_name = model_name
        self._tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tok.enable_truncation(max_length=_MAX_TOKENS)
        # Single-threaded: free tier has 0.1 CPU; avoids onnxruntime over-spawning.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        # Disable the growing CPU memory arena — caps peak RSS (the arena would
        # otherwise retain the largest forward's multi-GB allocation).
        opts.enable_cpu_mem_arena = False
        self._sess = ort.InferenceSession(str(model_dir / "model.onnx"), sess_options=opts)
        self._inputs = {i.name for i in self._sess.get_inputs()}

    @property
    def dim(self) -> int:
        return _E5_DIM

    def _embed(self, texts: list[str], kind: str) -> np.ndarray:
        prefixed = [_PREFIX[kind] + t for t in texts]
        # Process in fixed-size batches so peak memory stays bounded regardless of
        # how many texts are passed (ingestion/migration embed ~700 at once).
        parts = [self._forward(prefixed[i:i + _EMBED_BATCH])
                 for i in range(0, len(prefixed), _EMBED_BATCH)]
        return (np.vstack(parts).astype(np.float32) if parts
                else np.zeros((0, _E5_DIM), dtype=np.float32))

    def _forward(self, prefixed: list[str]) -> np.ndarray:
        """One ONNX forward over a small batch → masked mean-pooled, L2-normalized."""
        encs = self._tok.encode_batch(prefixed)
        maxlen = max(len(e.ids) for e in encs)
        ids = np.zeros((len(encs), maxlen), dtype=np.int64)
        mask = np.zeros((len(encs), maxlen), dtype=np.int64)
        for i, e in enumerate(encs):
            n = len(e.ids)
            ids[i, :n] = e.ids
            mask[i, :n] = e.attention_mask

        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        last_hidden = self._sess.run(None, feed)[0]  # (B, seq, 384)

        m = mask[..., None].astype(np.float32)
        mean = (last_hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        norms = np.linalg.norm(mean, axis=1, keepdims=True)
        return (mean / np.where(norms == 0, 1.0, norms)).astype(np.float32)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Embed documents/passages. Returns (N, 384) float32, L2-normalized."""
        return self._embed(list(texts), "passage")

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query. Returns (1, 384) float32, L2-normalized."""
        return self._embed([text], "query")
