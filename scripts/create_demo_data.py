"""Generate clearly synthetic, deterministic fixtures for local smoke runs."""

import csv
import datetime as dt
import json
import random
from pathlib import Path


def main() -> None:
    rng = random.Random(42)
    root = Path(__file__).resolve().parents[1] / "data"
    root.mkdir(exist_ok=True)
    with (root / "churn_synthetic.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["customer_id", "observed_at", "churned", "tenure_months",
                         "monthly_spend", "support_tickets", "usage_hours"])
        for i in range(500):
            tenure = rng.randint(1, 48)
            spend = round(rng.uniform(15, 120), 2)
            tickets = rng.randint(0, 7)
            usage = round(rng.uniform(1, 40), 2)
            risk = -0.5 - 0.035 * tenure + 0.28 * tickets - 0.08 * usage + 0.006 * spend
            probability = 1 / (1 + __import__("math").exp(-risk))
            churn = int(rng.random() < probability)
            day = dt.date(2024, 1, 1) + dt.timedelta(days=i)
            writer.writerow([f"demo-{i:04d}", day.isoformat(), churn, tenure, spend, tickets, usage])
    with (root / "demand_synthetic.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "units", "price", "promotion"])
        for i in range(180):
            day = dt.date(2025, 1, 1) + dt.timedelta(days=i)
            promotion = int(i % 19 in (0, 1))
            price = 10.0 + (i // 30) * 0.2
            units = max(0, round(45 + i * 0.025 + (12 if day.weekday() >= 5 else 0)
                                 + 9 * promotion - 2 * (price - 10) + rng.gauss(0, 3)))
            writer.writerow([day.isoformat(), units, price, promotion])
    templates = {
        0: ["How can I update the billing address on an invoice?",
            "The payment receipt for my subscription is missing.",
            "Please explain the tax line on last month's bill.",
            "I need to change the card used for recurring charges."],
        1: ["The API returns a timeout when I submit a request.",
            "My deployment fails with a container startup error.",
            "The webhook signature check is rejecting events.",
            "I cannot connect to the database from the service."],
    }
    for filename, count, start in (("tickets_train_synthetic.jsonl", 40, 0),
                                   ("tickets_val_synthetic.jsonl", 12, 40)):
        with (root / filename).open("w", encoding="utf-8") as handle:
            for i in range(start, start + count):
                label = i % 2
                text = templates[label][i % 4] + f" Reference case {i:03d}."
                handle.write(json.dumps({"text": text, "label": label}) + "\n")
    print(f"Wrote synthetic fixtures to {root}")


if __name__ == "__main__":
    main()
