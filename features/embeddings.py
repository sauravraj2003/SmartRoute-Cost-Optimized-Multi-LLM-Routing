"""Optional sentence-transformer embeddings with a deterministic fallback."""
from __future__ import annotations

import hashlib
import re


class PromptEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 dimensions: int = 64, use_transformer: bool = False) -> None:
        self.model_name, self.dimensions = model_name, dimensions
        self._model = None
        if use_transformer:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(model_name)
            except ImportError as exc:
                raise RuntimeError("Install sentence-transformers to enable semantic embeddings") from exc

    @property
    def supports_semantic_similarity(self) -> bool:
        return self._model is not None

    def encode(self, text: str) -> list[float]:
        if self._model is not None:
            return self._model.encode(text, normalize_embeddings=True).tolist()
        # Stable signed feature hashing keeps offline extraction reproducible.
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[a-zA-Z_][\w]*|[^\s\w]", text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            vector[value % self.dimensions] += 1.0 if value & (1 << 8) else -1.0
        norm = sum(x * x for x in vector) ** .5 or 1.0
        return [x / norm for x in vector]
