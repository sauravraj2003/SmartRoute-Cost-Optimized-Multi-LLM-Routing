import pytest

from router.policy import estimate_cost, select_model


def _models():
    return [
        {"id": "small", "tier": 1, "input_cost_per_million_tokens": 1,
         "output_cost_per_million_tokens": 2, "context_window": 4096},
        {"id": "large", "tier": 2, "input_cost_per_million_tokens": 5,
         "output_cost_per_million_tokens": 10, "context_window": 8192},
    ]


def test_selects_cheapest_model_over_threshold():
    decision = select_model(_models(), {"small": .8, "large": .95}, .7, 100, 100)
    assert decision.model["id"] == "small"
    assert decision.predicted_success == .8


def test_falls_back_to_strongest_feasible_model():
    decision = select_model(_models(), {"small": .4, "large": .6}, .9, 100, 100)
    assert decision.model["id"] == "large"
    assert "no model met" in decision.reason


def test_hard_cost_limit_is_not_silently_violated():
    with pytest.raises(ValueError, match="cost and latency limits"):
        select_model(_models(), {"small": .99, "large": .99}, .5, 100, 100,
                     max_cost_usd=.00001)


def test_long_context_multiplier_applies_to_both_token_rates():
    model = {"input_cost_per_million_tokens": 1, "output_cost_per_million_tokens": 2,
             "long_context_threshold_tokens": 100, "long_context_input_multiplier": 2,
             "long_context_output_multiplier": 3}
    assert estimate_cost(model, 101, 10) == pytest.approx((202 + 60) / 1_000_000)
