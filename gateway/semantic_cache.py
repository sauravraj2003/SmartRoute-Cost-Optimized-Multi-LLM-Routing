"""SQLite cache for responses verified by the request's exact test suite."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from features.embeddings import PromptEmbedder


class SemanticCache:
    """Cache only verified successes; use exact or embedding-similar prompts.

    Similarity is scoped by a caller supplied signature containing the exact tests,
    conversation context, model pool, and generation settings. This prevents reuse
    across incompatible test suites or request configurations.
    """

    def __init__(self, path: str | Path = "cache/semantic.sqlite3", ttl_seconds: int = 86_400,
                 max_entries: int = 10_000, embedder: PromptEmbedder | None = None) -> None:
        self.path = Path(path)
        self.ttl_seconds = max(0, int(ttl_seconds))
        self.max_entries = max(1, int(max_entries))
        self.embedder = embedder or PromptEmbedder(
            dimensions=256,
            use_transformer=(os.getenv("SMARTROUTE_USE_SENTENCE_TRANSFORMER", "false").lower()
                             in {"1", "true", "yes"}),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("""CREATE TABLE IF NOT EXISTS verified_responses (
                signature TEXT NOT NULL, prompt_hash TEXT NOT NULL, prompt TEXT NOT NULL,
                vector TEXT NOT NULL, payload TEXT NOT NULL, created_at REAL NOT NULL,
                PRIMARY KEY (signature, prompt_hash))""")
            connection.execute("CREATE INDEX IF NOT EXISTS cache_signature_time "
                               "ON verified_responses(signature, created_at DESC)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    @staticmethod
    def _hash(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            return 0.0
        denominator = math.sqrt(sum(x * x for x in left) * sum(y * y for y in right))
        return sum(x * y for x, y in zip(left, right)) / denominator if denominator else 0.0

    def get(self, signature: str, prompt: str, similarity_threshold: float = .96) -> dict[str, Any] | None:
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        now = time.time()
        vector = self.embedder.encode(prompt)
        prompt_hash = self._hash(prompt)
        with self._connect() as connection:
            connection.execute("DELETE FROM verified_responses WHERE created_at < ?",
                               (now - self.ttl_seconds,))
            rows = connection.execute(
                "SELECT prompt_hash, vector, payload FROM verified_responses "
                "WHERE signature=? ORDER BY created_at DESC LIMIT 1000", (signature,)).fetchall()
        best: tuple[float, str] | None = None
        for stored_hash, stored_vector, payload in rows:
            if stored_hash == prompt_hash:
                return json.loads(payload)
            # Feature hashing is deterministic but lexical; only a sentence
            # embedding is suitable for approximate semantic reuse.
            if not self.embedder.supports_semantic_similarity:
                continue
            similarity = self._cosine(vector, json.loads(stored_vector))
            if similarity >= similarity_threshold and (best is None or similarity > best[0]):
                best = (similarity, payload)
        return json.loads(best[1]) if best else None

    def put(self, signature: str, prompt: str, payload: dict[str, Any]) -> None:
        if payload.get("verified") is not True or payload.get("actual_success") is not True:
            raise ValueError("Only responses that passed their test suite can be cached")
        prompt_hash = self._hash(prompt)
        vector = json.dumps(self.embedder.encode(prompt), separators=(",", ":"))
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO verified_responses "
                "(signature, prompt_hash, prompt, vector, payload, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (signature, prompt_hash, prompt, vector,
                 json.dumps(payload, separators=(",", ":")), now))
            connection.execute("DELETE FROM verified_responses WHERE rowid IN "
                               "(SELECT rowid FROM verified_responses ORDER BY created_at DESC LIMIT -1 OFFSET ?)",
                               (self.max_entries,))


def cache_signature(messages: list[dict[str, str]], tests: str, entry_point: str,
                    mode: str, max_tokens: int, temperature: float,
                    models: list[dict[str, Any]], success_threshold: float,
                    max_cost_usd: float | None) -> str:
    """Stable cache scope for the context and execution contract of a request."""
    context = messages[:-1] if messages and messages[-1].get("role") == "user" else messages
    material = {
        "context": context, "tests": tests, "entry_point": entry_point, "mode": mode,
        "max_tokens": max_tokens, "temperature": temperature,
        "models": [{key: model.get(key) for key in ("id", "model_name", "provider", "tier",
                  "input_cost_per_million_tokens", "output_cost_per_million_tokens")}
                   for model in models],
        "success_threshold": success_threshold, "max_cost_usd": max_cost_usd,
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
