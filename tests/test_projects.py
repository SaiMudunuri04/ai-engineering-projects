import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path

from portfolio_ai import agents, core_ml, deep_learning, fine_tuning, forecasting, multimodal, rag


class ProjectTests(unittest.TestCase):
    def test_churn_training_and_artifact_round_trip(self):
        rows = []
        for i in range(100):
            label = int(i % 5 in (0, 1))
            rows.append({"customer_id": str(i), "observed_at": f"2026-01-{i + 1:03d}",
                         "x": [float(i % 20), 40.0 + i % 10, float(label * 5),
                               float(20 - label * 10)], "y": label})
        model = core_ml.train(rows[:80])
        self.assertGreater(core_ml.evaluate(model, rows[80:])["f1"], 0.8)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            model.save(path)
            self.assertAlmostEqual(model.predict_proba(rows[80]["x"]),
                                   core_ml.ChurnModel.load(path).predict_proba(rows[80]["x"]))

    def test_churn_rejects_repeated_customer(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.csv"
            header = "customer_id,observed_at,churned,tenure_months,monthly_spend,support_tickets,usage_hours\n"
            row = "same,2026-01-01,0,1,20,0,10\n"
            path.write_text(header + row * 10)
            with self.assertRaisesRegex(ValueError, "entity leakage"):
                core_ml.load_rows(path)

    def test_forecast_cannot_see_later_actuals(self):
        start = dt.date(2026, 1, 1)
        rows = [{"date": start + dt.timedelta(days=i), "units": float(40 + i % 7),
                 "price": 10.0, "promotion": 0} for i in range(42)]
        first = forecasting.backtest(rows, initial_train=28)["forecasts"][0]["forecast"]
        changed = [dict(row) for row in rows]
        changed[-1]["units"] = 10000.0
        second = forecasting.backtest(changed, initial_train=28)["forecasts"][0]["forecast"]
        self.assertEqual(first, second)

    def test_vision_split_detects_duplicate_image(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split in ("train", "val"):
                for label in ("good", "defect"):
                    path = root / split / label / "image.png"
                    path.parent.mkdir(parents=True)
                    path.write_bytes(label.encode())
            with self.assertRaisesRegex(ValueError, "same image"):
                deep_learning.validate_splits(root)

    def test_rag_retrieves_and_checks_citations(self):
        records = [{"source": "a.md", "offset": 0, "text": "Restart the worker after its queue drains."},
                   {"source": "b.md", "offset": 0, "text": "Rotate credentials during maintenance."}]
        class FakeGenerator:
            def complete(self, system, user):
                self.assertion = "[S1]" in user
                return "Drain the queue before restart. [S1]"
        fake = FakeGenerator()
        response = rag.answer("How do I restart the worker?", rag.BM25Index(records), fake)
        self.assertTrue(fake.assertion)
        self.assertEqual(response["citation_check"], "valid_ids")
        self.assertEqual(response["sources"][0]["source"], "a.md")

    def test_multimodal_ranks_images(self):
        class FakeEncoder:
            def text(self, text):
                return [1.0, 0.0] if "cat" in text else [0.0, 1.0]
            def image(self, path):
                return [1.0, 0.0] if path.stem == "cat" else [0.0, 1.0]
        records = [{"id": "cat", "caption": "", "image_path": "/tmp/cat.png"},
                   {"id": "dog", "caption": "", "image_path": "/tmp/dog.png"}]
        result = multimodal.search("cat", multimodal.build_index(records, FakeEncoder()), FakeEncoder())
        self.assertEqual(result[0]["id"], "cat")

    def test_agent_rejects_write_tool_and_respects_step_budget(self):
        class BadPlanner:
            def complete(self, system, user):
                return json.dumps({"action": "delete_cluster"})
        result = agents.run_incident("High latency", BadPlanner(), agents.ReadOnlyTools({}, {}), max_steps=2)
        self.assertEqual(result["status"], "step_limit")
        self.assertEqual(result["trace"][0]["observation"]["error"], "tool not allowed")

    def test_finetuning_rejects_train_validation_overlap(self):
        one = [{"fingerprint": "a"}]
        with self.assertRaisesRegex(ValueError, "overlap"):
            fine_tuning.validate_holdout(one, one)
        self.assertEqual(fine_tuning.binary_metrics([1, 0, 1], [1, 0, 0])["accuracy"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
