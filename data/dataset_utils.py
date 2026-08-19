"""Dataset normalization and deterministic task-level splitting."""
from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any


def normalize_task(row: dict[str, Any], dataset: str = "unknown") -> dict[str, Any]:
    prompt = row.get("prompt") or row.get("text") or row.get("question") or row.get("task")
    if not prompt:
        raise ValueError("Task row is missing prompt/text/question/task")
    return {
        "prompt_id": str(row.get("prompt_id") or row.get("task_id") or row.get("id") or ""),
        "prompt": str(prompt),
        "tests": _as_test_source(row.get("tests", row.get("test", ""))),
        "entry_point": row.get("entry_point", ""),
        "dataset": str(row.get("dataset", dataset)),
        "metadata": row.get("metadata", {}),
    }


def _as_test_source(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(str(part) for part in value)
    return str(value)


def split_tasks(tasks: Iterable[dict[str, Any]], train: float = .7, validation: float = .15,
                seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    if train <= 0 or validation < 0 or train + validation >= 1:
        raise ValueError("Require train > 0, validation >= 0, train + validation < 1")
    items = list(tasks)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for index, item in enumerate(items):
        metadata = item.get("metadata") or {}
        group = str(item.get("group_id") or metadata.get("group_id") or item.get("prompt_id") or index)
        grouped.setdefault(group, []).append(item)
    keys = list(grouped)
    random.Random(seed).shuffle(keys)
    target_train, target_validation = int(len(items) * train), int(len(items) * validation)
    result = {"train": [], "validation": [], "test": []}
    for key in keys:
        group = grouped[key]
        if len(result["train"]) < target_train:
            split = "train"
        elif len(result["validation"]) < target_validation:
            split = "validation"
        else:
            split = "test"
        result[split].extend(group)
    return result


def load_jsonl(path: str) -> list[dict[str, Any]]:
    import json
    with open(path, encoding="utf-8") as stream:
        return [normalize_task(json.loads(line)) for line in stream if line.strip()]
