"""Combines scalar, taxonomy and optional embedding features."""
from __future__ import annotations

import os
from dotenv import load_dotenv

from .embeddings import PromptEmbedder
from .scalar_features import extract_scalar_features
from .task_type import task_type_features

load_dotenv()


class FeaturePipeline:
    def __init__(self, embedder: PromptEmbedder | None = None, include_embeddings: bool = True,
                 include_scalar: bool = True, include_task_type: bool = True) -> None:
        self.include_embeddings = include_embeddings
        self.include_scalar = include_scalar
        self.include_task_type = include_task_type
        self.embedder = (embedder or PromptEmbedder(
            use_transformer=(os.getenv("SMARTROUTE_USE_SENTENCE_TRANSFORMER", "false").lower()
                             in {"1", "true", "yes"}))) if include_embeddings else None

    def transform_one(self, prompt: str) -> dict[str, float]:
        features: dict[str, float] = {}
        if self.include_scalar:
            features.update(extract_scalar_features(prompt))
        if self.include_task_type:
            features.update(task_type_features(prompt))
        if self.include_embeddings:
            assert self.embedder is not None
            features.update({f"embedding_{i}": value for i, value in enumerate(self.embedder.encode(prompt))})
        return features

    def transform(self, prompts: list[str]) -> list[dict[str, float]]:
        return [self.transform_one(prompt) for prompt in prompts]
