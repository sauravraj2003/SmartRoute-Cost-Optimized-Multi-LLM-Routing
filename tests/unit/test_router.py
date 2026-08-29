from router.escalation import escalation_candidates
from router.policy import RoutingPolicy


def test_escalation_advances_in_configured_order_and_skips_attempted():
    models = [{"id": "a", "tier": 1}, {"id": "b", "tier": 2}, {"id": "c", "tier": 3}]
    result = escalation_candidates(models, "a", {"b"}, escalation_order=[1, 2, 3])
    assert [model["id"] for model in result] == ["c"]


def test_routing_policy_selects_model_from_predictions():
    policy = RoutingPolicy([{"id": "m", "tier": 1, "input_cost_per_million_tokens": 1,
                            "output_cost_per_million_tokens": 1}], threshold=.5)
    assert policy.route({"m": .8}).model["id"] == "m"
