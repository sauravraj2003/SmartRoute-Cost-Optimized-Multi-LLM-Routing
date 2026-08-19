"""Deterministic task taxonomy for coding requests."""
from __future__ import annotations

from enum import Enum


class TaskType(str, Enum):
    GENERATION = "generation"
    DEBUGGING = "debugging"
    EXPLANATION = "explanation"
    REFACTORING = "refactoring"
    TESTING = "testing"
    ALGORITHM = "algorithm"
    CODE_REVIEW = "code_review"
    OTHER = "other"


_KEYWORDS = {
    TaskType.DEBUGGING: ("debug", "bug", "error", "exception", "traceback", "fix"),
    TaskType.EXPLANATION: ("explain", "what does", "how does", "describe"),
    TaskType.REFACTORING: ("refactor", "clean up", "simplify", "optimize"),
    TaskType.TESTING: ("unit test", "test case", "pytest", "write tests"),
    TaskType.CODE_REVIEW: ("review this", "code review", "vulnerability"),
    TaskType.ALGORITHM: ("leetcode", "data structure", "algorithm", "complexity"),
}


def classify_task(prompt: str) -> TaskType:
    """Classify using simple, transparent keyword rules; never calls an LLM."""
    text = prompt.lower()
    for label, terms in _KEYWORDS.items():
        if any(term in text for term in terms):
            return label
    if any(term in text for term in ("write", "implement", "create", "build", "generate")):
        return TaskType.GENERATION
    return TaskType.OTHER
