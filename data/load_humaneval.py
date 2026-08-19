"""Load HumanEval from a local JSONL file or Hugging Face datasets."""
from __future__ import annotations

from typing import Any

from .dataset_utils import normalize_task


def load_humaneval(path: str | None = None, split: str = "test", limit: int | None = None,
                   cache_dir: str | None = None) -> list[dict[str, Any]]:
    if path:
        from .dataset_utils import load_jsonl
        rows = load_jsonl(path)
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install datasets or provide a local HumanEval JSONL path") from exc
        data = load_dataset("openai/openai_humaneval", split=split, cache_dir=cache_dir)
        rows = [normalize_task(dict(row), "humaneval") for row in data]
    for i, row in enumerate(rows):
        row["dataset"] = "humaneval"
        row["prompt_id"] = row["prompt_id"] or f"HumanEval/{i}"
    return rows[:limit] if limit is not None else rows
