# Customer churn scoring · core ML 01

A command-line churn classifier for account-level observations. It checks schema, non-finite values, and repeated customer IDs, then sorts by observation date and reserves the newest 20% as a holdout. Feature means and scales are fitted only on training rows. The model is L2-regularized logistic regression implemented in Python, with a JSON artifact that records feature order.

Run `python3 scripts/create_demo_data.py`, then `python3 -m portfolio_ai.core_ml data/churn_synthetic.csv`. Output includes precision, recall, F1, and Brier score on the chronological holdout. The `artifacts/churn-model.json` file can be loaded with `ChurnModel.load` and used via `predict_proba`.

For a real study, supply one row per customer with `customer_id,observed_at,churned,tenure_months,monthly_spend,support_tickets,usage_hours`. Define the label using a future observation window that does not overlap the feature window; that business definition is outside this code. Review class balance, calibration, and subgroup error rates before any operational use. Synthetic fixtures are for smoke testing only.
