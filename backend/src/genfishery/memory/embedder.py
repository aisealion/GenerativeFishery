"""Local sentence-transformers embedder (project decision: local model, no
per-call embedding API/key -- memory writes and retrievals happen far too
often per round to pay network latency/cost for each one).

The model is loaded lazily and cached at module scope since constructing a
`SentenceTransformer` is expensive; every `AgentMemoryBank` in a process
shares the same loaded model.
"""

from collections.abc import Callable

import numpy as np

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_embedder() -> Callable[[str], np.ndarray]:
    """Returns a `Callable[[str], np.ndarray]` suitable for Concordia's
    `AssociativeMemoryBank(sentence_embedder=...)` and for our own direct
    query-embedding in `AgentMemoryBank.retrieve`. Embeddings are L2-normalized
    so a plain dot product equals cosine similarity everywhere they're used.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(_MODEL_NAME)

    def embed(text: str) -> np.ndarray:
        return _model.encode(text, normalize_embeddings=True)

    return embed
