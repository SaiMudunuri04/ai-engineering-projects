"""One-step demand forecasting with lag features and rolling-origin backtests."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path


def load_series(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"date", "units", "price", "promotion"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns: {sorted(required - set(reader.fieldnames or []))}")
        rows = []
        for row in reader:
            day = dt.date.fromisoformat(row["date"])
            units, price, promotion = float(row["units"]), float(row["price"]), int(row["promotion"])
            if min(units, price) < 0 or promotion not in (0, 1) or not all(
                math.isfinite(v) for v in (units, price)
            ):
                raise ValueError("Invalid demand observation")
            rows.append({"date": day, "units": units, "price": price, "promotion": promotion})
    rows.sort(key=lambda item: item["date"])
    if len(rows) < 21:
        raise ValueError("At least 21 daily observations are required")
    if len({row["date"] for row in rows}) != len(rows):
        raise ValueError("Duplicate dates are not allowed")
    if any((right["date"] - left["date"]).days != 1 for left, right in zip(rows, rows[1:])):
        raise ValueError("Dates must be consecutive; impute missing dates explicitly")
    return rows


def features(rows: list[dict], index: int) -> list[float]:
    if index < 7:
        raise ValueError("A full week of history is required")
    row = rows[index]
    weekday = row["date"].weekday()
    return [1.0, rows[index - 1]["units"], rows[index - 7]["units"],
            sum(item["units"] for item in rows[index - 7:index]) / 7,
            row["price"], float(row["promotion"]),
            math.sin(2 * math.pi * weekday / 7), math.cos(2 * math.pi * weekday / 7)]


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda i: abs(augmented[i][col]))
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        if abs(augmented[col][col]) < 1e-12:
            raise ValueError("Singular design matrix")
        scale = augmented[col][col]
        augmented[col] = [value / scale for value in augmented[col]]
        for row in range(n):
            if row != col:
                factor = augmented[row][col]
                augmented[row] = [a - factor * b for a, b in zip(augmented[row], augmented[col])]
    return [row[-1] for row in augmented]


def fit(rows: list[dict], end: int, ridge: float = 10.0) -> list[float]:
    if end <= 7 or end > len(rows) or ridge <= 0:
        raise ValueError("Invalid training window or ridge penalty")
    x = [features(rows, i) for i in range(7, end)]
    y = [rows[i]["units"] for i in range(7, end)]
    width = len(x[0])
    matrix = [[sum(row[i] * row[j] for row in x) + (ridge if i == j and i else 0.0)
               for j in range(width)] for i in range(width)]
    vector = [sum(row[i] * target for row, target in zip(x, y)) for i in range(width)]
    return _solve(matrix, vector)


def predict(weights: list[float], rows: list[dict], index: int) -> float:
    return max(0.0, sum(w * value for w, value in zip(weights, features(rows, index))))


def backtest(rows: list[dict], initial_train: int | None = None, refit_every: int = 7) -> dict:
    initial_train = initial_train or max(14, int(len(rows) * 0.7))
    if initial_train < 14 or initial_train >= len(rows) or refit_every < 1:
        raise ValueError("Invalid backtest configuration")
    forecasts = []
    weights = []
    for index in range(initial_train, len(rows)):
        if index == initial_train or (index - initial_train) % refit_every == 0:
            weights = fit(rows, index)
        forecast = predict(weights, rows, index)
        forecasts.append({"date": rows[index]["date"].isoformat(),
                          "actual": rows[index]["units"], "forecast": round(forecast, 3)})
    absolute_errors = [abs(item["actual"] - item["forecast"]) for item in forecasts]
    total_actual = sum(item["actual"] for item in forecasts)
    return {"method": "expanding-window one-step forecast", "refit_every_days": refit_every,
            "observations": len(forecasts), "mae": sum(absolute_errors) / len(absolute_errors),
            "wape": sum(absolute_errors) / total_actual if total_actual else None,
            "forecasts": forecasts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/forecast-report.json"))
    args = parser.parse_args()
    report = backtest(load_series(args.csv))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "forecasts"}, indent=2))


if __name__ == "__main__":
    main()
