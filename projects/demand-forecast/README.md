# Demand forecasting · core ML 02

A one-step demand forecaster using prior-day units, prior-week units, trailing-seven-day mean, price, promotion, and weekday seasonality. It fits a ridge regression and backtests with an expanding training window. Refit occurs every seven days. The report includes individual forecasts, mean absolute error, and weighted absolute percentage error.

Run `python3 scripts/create_demo_data.py`, then `python3 -m portfolio_ai.forecasting data/demand_synthetic.csv`. Provide a real daily CSV with `date,units,price,promotion` to evaluate your own series. Dates must be consecutive; missing dates require an explicit business decision before training.

This is a one-step-ahead design: at prediction time it assumes yesterday's actual units and the current day's price/promotion plan are available. It is not a multi-week forecast. The test suite checks that changing a later actual cannot alter an earlier forecast.
