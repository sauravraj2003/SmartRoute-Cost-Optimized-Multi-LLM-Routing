"""Train one success classifier per model from benchmark matrix rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from features.feature_pipeline import FeaturePipeline
from router.calibrate import ProbabilityCalibrator


def _classifier():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=250, max_depth=5, learning_rate=.05,
                             subsample=.9, colsample_bytree=.9, eval_metric="logloss",
                             random_state=42, n_jobs=4)
    except ImportError:
        return HistGradientBoostingClassifier(max_iter=150, learning_rate=.08, random_state=42)


def train_router(rows: pd.DataFrame, output_dir: str | Path = "router/models",
                 pipeline: FeaturePipeline | None = None,
                 validation_rows: pd.DataFrame | None = None) -> dict[str, Any]:
    required = {"prompt_id", "prompt", "model_id", "success"}
    if missing := required - set(rows.columns):
        raise ValueError(f"Training data missing columns: {sorted(missing)}")
    if rows.duplicated(["prompt_id", "model_id"]).any():
        raise ValueError("Benchmark matrix contains duplicate prompt_id/model_id rows")
    expected = {(prompt_id, model_id) for prompt_id in rows.prompt_id.unique()
                for model_id in rows.model_id.unique()}
    observed = set(zip(rows.prompt_id, rows.model_id))
    missing_pairs = expected - observed
    if missing_pairs:
        raise ValueError(f"Benchmark matrix is incomplete ({len(missing_pairs)} task/model pairs missing)")
    pipeline = pipeline or FeaturePipeline()
    rows = rows.copy()
    if "total_tests" in rows:
        rows = rows[rows.total_tests.fillna(0).astype(float) > 0]
    if "evaluation_error" in rows:
        rows = rows[rows.evaluation_error.isna() | (rows.evaluation_error == "")]
    if rows.empty:
        raise ValueError("No successfully evaluated benchmark rows are available for training")
    prompts = rows[["prompt_id", "prompt"]].drop_duplicates("prompt_id")
    if validation_rows is not None and not validation_rows.empty:
        validation_rows = validation_rows.copy()
        if "total_tests" in validation_rows:
            validation_rows = validation_rows[validation_rows.total_tests.fillna(0).astype(float) > 0]
        if "evaluation_error" in validation_rows:
            validation_rows = validation_rows[validation_rows.evaluation_error.isna() | (validation_rows.evaluation_error == "")]
        prompts = pd.concat([prompts, validation_rows[["prompt_id", "prompt"]]], ignore_index=True).drop_duplicates("prompt_id")
    feature_rows = {row.prompt_id: pipeline.transform_one(row.prompt) for row in prompts.itertuples()}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    models = {}
    for model_id, group in rows.groupby("model_id"):
        aligned = group.drop_duplicates("prompt_id")
        x = pd.DataFrame([feature_rows[key] for key in aligned.prompt_id])
        y = aligned.success.astype(int)
        if y.nunique() < 2:
            # A constant outcome is a valid empirical prior, and avoids classifier errors.
            estimator = None
            prior = float(y.mean())
        else:
            estimator, prior = _classifier(), float(y.mean())
            estimator.fit(x, y)
        calibrator = None
        if validation_rows is not None and not validation_rows.empty:
            val = validation_rows[validation_rows.model_id == model_id].drop_duplicates("prompt_id")
            val = val[val.prompt_id.isin(feature_rows)]
            if len(val) >= 20 and val.success.astype(int).nunique() == 2:
                val_x = pd.DataFrame([feature_rows[key] for key in val.prompt_id], columns=x.columns)
                raw = estimator.predict_proba(val_x)[:, 1] if estimator is not None else [prior] * len(val)
                method = "isotonic" if len(val) >= 100 else "platt"
                calibrator = ProbabilityCalibrator(method).fit(raw, val.success.astype(int).to_numpy())
        artifact = {"estimator": estimator, "prior": prior, "calibrator": calibrator,
                    "feature_columns": list(x.columns),
                    "model_id": str(model_id), "version": 1}
        joblib.dump(artifact, output / f"{model_id}.joblib")
        models[str(model_id)] = artifact
    feature_parts = []
    if pipeline.include_embeddings:
        feature_parts.append("embeddings")
    if pipeline.include_scalar:
        feature_parts.append("scalar")
    if pipeline.include_task_type:
        feature_parts.append("task_type")
    (output / "manifest.json").write_text(json.dumps({"models": list(models), "features": "+".join(feature_parts),
        "embedding_backend": ("sentence_transformer" if pipeline.embedder is not None
                              and pipeline.embedder.supports_semantic_similarity else "feature_hashing")
        if pipeline.include_embeddings else None,
        "calibrated_models": [key for key, value in models.items() if value["calibrator"] is not None]}, indent=2), encoding="utf-8")
    return models


class SuccessPredictor:
    def __init__(self, model_dir: str | Path = "router/models",
                 pipeline: FeaturePipeline | None = None, apply_calibration: bool = True) -> None:
        self.model_dir = Path(model_dir)
        self.pipeline = pipeline or FeaturePipeline()
        self.apply_calibration = apply_calibration

    def predict(self, prompt: str, model_ids: list[str]) -> dict[str, float]:
        features = self.pipeline.transform_one(prompt)
        predictions = {}
        for model_id in model_ids:
            path = self.model_dir / f"{model_id}.joblib"
            if not path.exists():
                predictions[model_id] = .5
                continue
            artifact = joblib.load(path)
            estimator = artifact["estimator"]
            if estimator is None:
                raw = float(artifact["prior"])
            else:
                x = pd.DataFrame([[features[c] for c in artifact["feature_columns"]]], columns=artifact["feature_columns"])
                raw = float(estimator.predict_proba(x)[0, 1]) if estimator is not None else float(artifact["prior"])
            calibrator = artifact.get("calibrator") if self.apply_calibration else None
            predictions[model_id] = float(calibrator.predict([raw])[0]) if calibrator is not None else raw
        return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="benchmark/results/matrix.csv")
    parser.add_argument("--output", default="router/models")
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--pricing", default="configs/pricing.yaml")
    args = parser.parse_args()
    path = Path(args.input)
    rows = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "model_id" not in rows:
        raise ValueError("Benchmark matrix is missing model_id")
    from router.config import load_models_config
    expected_models = {model["id"] for model in load_models_config(args.models, args.pricing)
                       if model.get("enabled", True)}
    observed_models = set(rows.model_id.astype(str))
    missing_models = expected_models - observed_models
    if missing_models:
        raise ValueError(f"Benchmark matrix is missing configured models: {sorted(missing_models)}")
    from data.dataset_utils import split_tasks
    columns = ["prompt_id", "prompt"] + (["group_id"] if "group_id" in rows else [])
    unique = rows[columns].drop_duplicates("prompt_id").to_dict("records")
    partitions = split_tasks(unique, train=.7, validation=.15, seed=42)
    train_ids = {row["prompt_id"] for row in partitions["train"]}
    validation_ids = {row["prompt_id"] for row in partitions["validation"]}
    train_rows = rows[rows.prompt_id.isin(train_ids)]
    validation_rows = rows[rows.prompt_id.isin(validation_ids)]
    artifacts = train_router(train_rows, args.output, validation_rows=validation_rows)
    Path(args.output, "split.json").write_text(json.dumps({"seed": 42, "ratios": {"train": .7, "validation": .15, "test": .15},
        "counts": {name: len(part) for name, part in partitions.items()}, "task_ids": {name: [r["prompt_id"] for r in part] for name, part in partitions.items()},
        "calibrated_models": [name for name, value in artifacts.items() if value["calibrator"] is not None]}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
