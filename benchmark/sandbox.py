"""Isolated Docker runner for untrusted candidate Python code."""
from __future__ import annotations

import json
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path


def run_python_tests(code: str, tests: str, entry_point: str = "", timeout: float = 5.0,
                    memory_limit: str = "256m", cpu_limit: float = 1.0,
                    docker_image: str = "python:3.11-slim", require_docker: bool = True) -> dict:
    """Run code only in a no-network, resource-limited disposable container.

    The host fallback is deliberately disabled by default because generated code is untrusted.
    """
    if entry_point and not re.fullmatch(r"[A-Za-z_]\w*", entry_point):
        raise ValueError("entry_point must be a valid Python identifier")
    if not entry_point and re.search(r"(?m)^\s*def\s+check\s*\(", tests):
        raise ValueError("entry_point is required for test suites that define check(candidate)")
    if not shutil.which("docker"):
        if require_docker:
            raise RuntimeError("Docker is required to run untrusted benchmark code")
        return _run_subprocess(code, tests, entry_point, timeout)
    marker = "__SMARTRoute_RESULT__" + secrets.token_hex(16) + "__"
    wrapper = (code + "\nimport json\n_tests = " + repr(tests) + "\n" +
               "_passed = 0\n_total = 1\n_error = None\ntry:\n" +
               "    exec(compile(_tests, '<benchmark-tests>', 'exec'), globals())\n" +
               (f"    check({entry_point})\n" if entry_point else "") +
               "    _passed = _total\nexcept BaseException as exc:\n" +
               "    _error = type(exc).__name__ + ': ' + str(exc)\n" +
               "print(" + repr(marker) + " + json.dumps({'passed_tests': _passed, 'total_tests': _total, 'error': _error}))\n")
    with tempfile.TemporaryDirectory(prefix="smartroute-") as tmp:
        script = Path(tmp) / "candidate.py"
        script.write_text(wrapper, encoding="utf-8")
        container_name = f"smartroute-{secrets.token_hex(8)}"
        command = ["docker", "run", "--rm", "--name", container_name, "--network", "none", "--read-only", "--user", "65534:65534", "--cap-drop", "ALL",
                   "--security-opt", "no-new-privileges", "--pids-limit", "64", "--memory", memory_limit,
                   "--cpus", str(cpu_limit), "--tmpfs", "/tmp:rw,noexec,nosuid,size=32m",
                   "--mount", f"type=bind,src={script},dst=/candidate.py,readonly", docker_image,
                   "python", "-I", "/candidate.py"]
        try:
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.TimeoutExpired:
            return {"success": False, "quality_score": 0.0, "passed_tests": 0, "total_tests": 1, "error": "timeout"}
        except OSError as exc:
            return {"success": False, "quality_score": 0.0, "passed_tests": 0, "total_tests": 1, "error": str(exc)}
        streams: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}

        def drain(name: str, pipe) -> None:
            while chunk := pipe.read(65536):
                streams[name].extend(chunk)
                if len(streams[name]) > 1_000_000:
                    del streams[name][:-1_000_000]

        readers = [threading.Thread(target=drain, args=(name, getattr(proc, name)), daemon=True)
                   for name in ("stdout", "stderr")]
        for reader in readers:
            reader.start()
        try:
            return_code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            subprocess.run(["docker", "rm", "--force", container_name], capture_output=True,
                           timeout=5, check=False)
            return {"success": False, "quality_score": 0.0, "passed_tests": 0,
                    "total_tests": 1, "error": "timeout"}
        for reader in readers:
            reader.join(timeout=2)
    stdout = bytes(streams["stdout"]).decode("utf-8", errors="replace")
    stderr = bytes(streams["stderr"]).decode("utf-8", errors="replace")
    return _parse_result(stdout, stderr, return_code, marker)


def _parse_result(stdout: str, stderr: str, returncode: int, marker: str | None = None) -> dict:
    for line in reversed(stdout.splitlines()):
        if marker is not None and not line.startswith(marker):
            continue
        if marker is not None:
            line = line[len(marker):]
        try:
            data = json.loads(line)
            passed, total = data["passed_tests"], data["total_tests"]
            return {"success": passed == total and returncode == 0, "quality_score": passed / max(1, total),
                    "passed_tests": passed, "total_tests": total, "error": data.get("error") or (stderr[-2000:] or None)}
        except (ValueError, KeyError, TypeError):
            continue
    return {"success": False, "quality_score": 0.0, "passed_tests": 0, "total_tests": 1,
            "error": stderr[-2000:] or stdout[-2000:] or f"runner exited {returncode}"}


def _run_subprocess(code: str, tests: str, entry_point: str, timeout: float) -> dict:
    """Unsafe local mode, available only by explicit opt-in for trusted code."""
    import sys
    wrapper = code + "\n" + tests + (f"\ncheck({entry_point})\n" if entry_point else "")
    with tempfile.TemporaryDirectory(prefix="smartroute-local-") as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(wrapper, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, "-I", str(path)], capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return {"success": False, "quality_score": 0.0, "error": "timeout"}
    return {"success": proc.returncode == 0, "quality_score": float(proc.returncode == 0),
            "error": proc.stderr[-2000:] or None, "passed_tests": int(proc.returncode == 0), "total_tests": 1}
