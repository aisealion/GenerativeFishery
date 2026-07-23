"""Shared test fixtures."""

import numpy as np

_VOCAB = ["fish", "lake", "punish", "vote", "weather", "storm", "effort", "stock"]


def fake_embedder(text: str) -> np.ndarray:
    """Deterministic bag-of-words embedder for tests -- avoids loading the
    real (heavyweight) sentence-transformers model just to test ranking logic.
    """
    lowered = text.lower()
    vector = np.array([1.0 if word in lowered else 0.0 for word in _VOCAB])
    norm = np.linalg.norm(vector)
    if norm == 0:
        # Give every memory a small shared component so unrelated text isn't
        # a zero vector (cosine with an all-zero vector is undefined otherwise).
        vector = np.full(len(_VOCAB), 1e-6)
        norm = np.linalg.norm(vector)
    return vector / norm
