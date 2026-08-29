import pytest

from benchmark.sandbox import run_python_tests


def test_sandbox_requires_docker_unless_host_fallback_is_explicit(monkeypatch):
    monkeypatch.setattr("benchmark.sandbox.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="Docker is required"):
        run_python_tests("def solve(): return 1", "assert solve() == 1", require_docker=True)
