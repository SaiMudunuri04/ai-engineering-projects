"""Leakage-aware customer churn classification with a serializable linear model."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


FEATURES = ("tenure_months", "monthly_spend", "support_tickets", "usage_hours")


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"customer_id", "observed_at", "churned", *FEATURES}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns: {sorted(required - set(reader.fieldnames or []))}")
        rows = []
        seen = set()
        for line, row in enumerate(reader, start=2):
            if row["customer_id"] in seen:
                raise ValueError(f"Repeated customer_id at line {line}; avoid entity leakage")
            seen.add(row["customer_id"])
            try:
                values = [float(row[name]) for name in FEATURES]
                target = int(row["churned"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid numeric value at line {line}") from exc
            if target not in (0, 1) or not all(math.isfinite(v) for v in values):
                raise ValueError(f"Invalid target or non-finite feature at line {line}")
            rows.append({"customer_id": row["customer_id"], "observed_at": row["observed_at"],
                         "x": values, "y": target})
    if len(rows) < 10:
        raise ValueError("At least 10 unique observations are required")
    return sorted(rows, key=lambda item: item["observed_at"])


@dataclass
class ChurnModel:
    means: list[float]
    scales: list[float]
    weights: list[float]
    bias: float
    threshold: float = 0.5

    def predict_proba(self, values: list[float]) -> float:
        if len(values) != len(FEATURES) or not all(math.isfinite(v) for v in values):
            raise ValueError("Expected four finite feature values")
        score = self.bias + sum(w * (v - m) / s for w, v, m, s in
                                zip(self.weights, values, self.means, self.scales))
        return _sigmoid(score)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"feature_order": FEATURES, **self.__dict__}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ChurnModel":
        data = json.loads(path.read_text(encoding="utf-8"))
        if tuple(data.pop("feature_order")) != FEATURES:
            raise ValueError("Model feature schema does not match this release")
        return cls(**data)


def train(rows: list[dict], *, epochs: int = 700, learning_rate: float = 0.08,
          l2: float = 0.01) -> ChurnModel:
    if epochs < 1 or learning_rate <= 0 or l2 < 0:
        raise ValueError("Invalid training hyperparameters")
    columns = list(zip(*(row["x"] for row in rows)))
    means = [sum(col) / len(col) for col in columns]
    scales = [max(math.sqrt(sum((v - m) ** 2 for v in col) / len(col)), 1e-9)
              for col, m in zip(columns, means)]
    x = [[(v - m) / s for v, m, s in zip(row["x"], means, scales)] for row in rows]
    y = [row["y"] for row in rows]
    weights = [0.0] * len(FEATURES)
    bias = 0.0
    for _ in range(epochs):
        errors = [_sigmoid(bias + sum(w * v for w, v in zip(weights, features))) - label
                  for features, label in zip(x, y)]
        gradient = [sum(error * features[j] for error, features in zip(errors, x)) / len(x)
                    + l2 * weights[j] for j in range(len(FEATURES))]
        weights = [w - learning_rate * g for w, g in zip(weights, gradient)]
        bias -= learning_rate * sum(errors) / len(errors)
    return ChurnModel(means, scales, weights, bias)


def evaluate(model: ChurnModel, rows: list[dict]) -> dict[str, float | int]:
    tp = tn = fp = fn = 0
    brier = 0.0
    for row in rows:
        probability = model.predict_proba(row["x"])
        predicted = int(probability >= model.threshold)
        actual = row["y"]
        tp += predicted == 1 and actual == 1
        tn += predicted == 0 and actual == 0
        fp += predicted == 1 and actual == 0
        fn += predicted == 0 and actual == 1
        brier += (probability - actual) ** 2
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"samples": len(rows), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "brier": brier / len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/churn-model.json"))
    args = parser.parse_args()
    rows = load_rows(args.csv)
    split = max(1, min(len(rows) - 1, int(len(rows) * 0.8)))
    model = train(rows[:split])
    model.save(args.output)
    print(json.dumps({"split": "chronological", "train": len(rows[:split]),
                      "test": evaluate(model, rows[split:])}, indent=2))


if __name__ == "__main__":
    main()
