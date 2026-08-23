"""Probability calibration helpers fitted strictly on validation predictions."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    def __init__(self, method: str = "isotonic") -> None:
        if method not in {"isotonic", "platt"}:
            raise ValueError("method must be isotonic or platt")
        self.method, self.model = method, None

    def fit(self, probabilities, outcomes):
        p, y = np.asarray(probabilities, dtype=float), np.asarray(outcomes, dtype=int)
        if len(p) != len(y) or len(p) == 0:
            raise ValueError("probabilities and outcomes must have equal nonzero length")
        if np.unique(y).size < 2:
            raise ValueError("Calibration requires both successful and failed validation examples")
        if self.method == "isotonic":
            self.model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p, y)
        else:
            self.model = LogisticRegression().fit(p.reshape(-1, 1), y)
        return self

    def predict(self, probabilities):
        if self.model is None:
            raise RuntimeError("Call fit before predict")
        p = np.asarray(probabilities, dtype=float)
        if self.method == "isotonic":
            return self.model.predict(p)
        return self.model.predict_proba(p.reshape(-1, 1))[:, 1]
