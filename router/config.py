"""Load model definitions and apply the separately versioned pricing ledger."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_models_config(models_path: str | Path = "configs/models.yaml",
                       pricing_path: str | Path = "configs/pricing.yaml") -> list[dict[str, Any]]:
    with open(models_path, encoding="utf-8") as stream:
        model_config = yaml.safe_load(stream) or {}
    models = model_config.get("models", [])
    price_path = Path(pricing_path)
    prices_by_id: dict[str, dict[str, Any]] = {}
    if price_path.is_file():
        with price_path.open(encoding="utf-8") as stream:
            pricing = yaml.safe_load(stream) or {}
        for entry in pricing.get("prices", []):
            model_id = str(entry["model_id"])
            if model_id in prices_by_id:
                raise ValueError(f"Duplicate price entry for model {model_id}")
            input_rate, output_rate = float(entry["input"]), float(entry["output"])
            if input_rate < 0 or output_rate < 0:
                raise ValueError(f"Token rates for {model_id} must be non-negative")
            prices_by_id[model_id] = {"input_cost_per_million_tokens": input_rate,
                                      "output_cost_per_million_tokens": output_rate}
    model_ids = {str(model.get("id", "")) for model in models}
    unknown_prices = set(prices_by_id) - model_ids
    if unknown_prices:
        raise ValueError(f"Pricing entries have no matching model definitions: {sorted(unknown_prices)}")
    for model in models:
        if model["id"] in prices_by_id:
            model.update(prices_by_id[model["id"]])
    seen: set[str] = set()
    for model in models:
        model_id = str(model.get("id", ""))
        if not model_id or model_id in seen:
            raise ValueError(f"Model IDs must be present and unique: {model_id!r}")
        seen.add(model_id)
        if int(model.get("tier", 0)) < 1:
            raise ValueError(f"Model {model_id} must have a positive tier")
        if "input_cost_per_million_tokens" not in model or "output_cost_per_million_tokens" not in model:
            raise ValueError(f"Model {model_id} needs input and output rates in models.yaml or pricing.yaml")
        input_rate = float(model["input_cost_per_million_tokens"])
        output_rate = float(model["output_cost_per_million_tokens"])
        if input_rate < 0 or output_rate < 0:
            raise ValueError(f"Token rates for {model_id} must be non-negative")
        if model.get("provider") == "local" and not (
                input_rate > 0 or output_rate > 0):
            raise ValueError(f"Local model {model_id} needs a nonzero normalized cost estimate")
    return models
