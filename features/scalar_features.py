"""Dependency-light scalar prompt features."""
from __future__ import annotations

import re

_DEBUG = re.compile(r"\b(debug|bug|error|exception|traceback|fix)\b", re.I)
_TEST = re.compile(r"\b(test|pytest|unittest|assert)\b", re.I)


def extract_scalar_features(prompt: str) -> dict[str, float]:
    code_blocks = re.findall(r"```[\w+-]*\n?(.*?)```", prompt, re.S)
    code = "\n".join(code_blocks)
    return {
        "char_count": float(len(prompt)),
        "word_count": float(len(prompt.split())),
        "line_count": float(prompt.count("\n") + 1),
        "estimated_tokens": float(max(1, (len(prompt) + 3) // 4)),
        "code_block_count": float(len(code_blocks)),
        "code_char_count": float(len(code)),
        "function_count": float(len(re.findall(r"\bdef\s+\w+", prompt))),
        "class_count": float(len(re.findall(r"\bclass\s+\w+", prompt))),
        "file_count": float(len(re.findall(r"(?:^|\s)[\w./-]+\.(?:py|js|ts|java|go|rs)\b", prompt))),
        "has_stack_trace": float(bool(re.search(r"Traceback \(most recent call last\)|\bat line \d+", prompt))),
        "debug_keyword": float(bool(_DEBUG.search(prompt))),
        "test_keyword": float(bool(_TEST.search(prompt))),
        "asks_for_long_output": float(bool(re.search(r"\b(detailed|comprehensive|entire|full implementation)\b", prompt, re.I))),
    }
