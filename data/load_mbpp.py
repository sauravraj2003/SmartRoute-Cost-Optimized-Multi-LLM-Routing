"""Load MBPP from a local JSONL file or Hugging Face datasets."""
from __future__ import annotations

from typing import Any

from .dataset_utils import normalize_task


def load_mbpp(path: str | None = None, split: str = "test", limit: int | None = None,
              cache_dir: str | None = None) -> list[dict[str, Any]]:
    if path:
        from .dataset_utils import load_jsonl
        rows = load_jsonl(path)
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install datasets or provide a local MBPP JSONL path") from exc
        data = load_dataset("google-research-datasets/mbpp", "full", split=split, cache_dir=cache_dir)
        rows = []
        for row in data:
            setup = row.get("test_setup_code", "") or ""
            tests = ([setup] if setup else []) + list(row.get("test_list", []))
            rows.append(normalize_task({"prompt_id": row.get("task_id"), "prompt": row.get("text"),
                                        "tests": "\n".join(tests), "metadata": row}, "mbpp"))
    for i, row in enumerate(rows):
        row["dataset"] = "mbpp"
        row["prompt_id"] = row["prompt_id"] or f"mbpp/{i}"
    return rows[:limit] if limit is not None else rows
