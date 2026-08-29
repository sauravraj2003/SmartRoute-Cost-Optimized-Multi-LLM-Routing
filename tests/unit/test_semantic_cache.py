import pytest

from gateway.semantic_cache import SemanticCache, cache_signature


def test_cache_only_accepts_verified_success(tmp_path):
    cache = SemanticCache(tmp_path / "responses.sqlite3")
    with pytest.raises(ValueError, match="Only responses"):
        cache.put("scope", "prompt", {"verified": False, "actual_success": False})


def test_cache_reads_exact_verified_response(tmp_path):
    cache = SemanticCache(tmp_path / "responses.sqlite3")
    payload = {"text": "def solve(): return 42", "model": "model-a",
               "verified": True, "actual_success": True}
    cache.put("scope", "write solve", payload)
    assert cache.get("scope", "write solve") == payload
    assert cache.get("other-scope", "write solve") is None


def test_cache_scope_changes_when_tests_change():
    messages = [{"role": "user", "content": "write solve"}]
    models = [{"id": "model-a", "provider": "openai", "tier": 1}]
    first = cache_signature(messages, "assert solve() == 1", "solve", "balanced", 256,
                            .2, models, .7, .01)
    second = cache_signature(messages, "assert solve() == 2", "solve", "balanced", 256,
                             .2, models, .7, .01)
    assert first != second
