import pytest

from features.embeddings import PromptEmbedder
from features.scalar_features import extract_scalar_features
from features.task_type import task_type_features


def test_scalar_features_count_prompt_structure():
    features = extract_scalar_features("Fix this:\n```python\ndef solve():\n    pass\n```")
    assert features["code_block_count"] == 1
    assert features["function_count"] == 1
    assert features["debug_keyword"] == 1


def test_task_features_are_one_hot():
    values = task_type_features("Debug this exception traceback")
    assert sum(values.values()) == 1


def test_hash_embeddings_are_stable_and_normalized():
    first = PromptEmbedder(dimensions=16).encode("write a function")
    assert first == PromptEmbedder(dimensions=16).encode("write a function")
    assert len(first) == 16
    assert sum(value * value for value in first) == pytest.approx(1)


def test_feature_pipeline_can_disable_embedding_and_taxonomy_groups():
    from features.feature_pipeline import FeaturePipeline

    features = FeaturePipeline(include_embeddings=False, include_task_type=False).transform_one("debug code")
    assert "debug_keyword" in features
    assert not any(name.startswith("embedding_") or name.startswith("task_") for name in features)
