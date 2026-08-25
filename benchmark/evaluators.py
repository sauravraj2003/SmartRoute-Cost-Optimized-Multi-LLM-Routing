"""Benchmark response evaluators."""
from __future__ import annotations

import re
import textwrap


def extract_python_code(response: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", response, flags=re.I | re.S)
    if blocks:
        return max(blocks, key=len).strip()
    match = re.search(r"(?m)^(?:def |class |import |from ).*", response)
    return response[match.start():].strip() if match else response.strip()


def evaluate_exact(response: str, expected: str) -> dict:
    passed = response.strip() == expected.strip()
    return {"success": passed, "quality_score": float(passed), "passed_tests": int(passed), "total_tests": 1}


def prepare_candidate(response: str, entry_point: str = "", starter_code: str = "") -> str:
    candidate = extract_python_code(response)
    has_starter_signature = bool(entry_point and re.search(
        rf"(?m)^\s*def\s+{re.escape(entry_point)}\s*\(", starter_code))
    if has_starter_signature and not re.search(rf"(?m)^\s*def\s+{re.escape(entry_point)}\s*\(", candidate):
        # HumanEval prompts end inside the requested function. Preserve indentation from that prompt.
        return starter_code + candidate.lstrip("\r\n")
    return textwrap.dedent(candidate)


def evaluate_code(response: str, tests: str, entry_point: str = "", timeout: float = 5.0,
                  starter_code: str = "") -> dict:
    from .sandbox import run_python_tests
    result = run_python_tests(prepare_candidate(response, entry_point, starter_code), tests,
                              entry_point=entry_point, timeout=timeout)
    return {"success": result["success"], "quality_score": result["quality_score"],
            "passed_tests": result.get("passed_tests", int(result["success"])),
            "total_tests": result.get("total_tests", 1), "sandbox": result}
