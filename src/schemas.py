"""
src/schemas.py — Pandera schemas for pipeline data contracts.
"""
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

prices_clean_schema = pa.DataFrameSchema(
    {
        "date": pa.Column("datetime64[ns]", nullable=False),
        "ticker": pa.Column(pd.StringDtype(), nullable=False),
        "open": pa.Column(float, [pa.Check.gt(0)], nullable=False),
        "high": pa.Column(float, [pa.Check.gt(0)], nullable=False),
        "low": pa.Column(float, [pa.Check.gt(0)], nullable=False),
        "close": pa.Column(float, [pa.Check.gt(0)], nullable=False),
        "volume": pa.Column(pd.Int64Dtype(), nullable=True),
    },
    checks=[
        pa.Check(
            lambda df: (df["high"] >= df["low"]).all(),
            error="high must be >= low for all rows",
        ),
    ],
    unique=["date", "ticker"],
    coerce=False,
)

returns_schema = pa.DataFrameSchema(
    {
        "date": pa.Column("datetime64[ns]", nullable=False),
        "ticker": pa.Column(pd.StringDtype(), nullable=False),
        "simple_return": pa.Column(float, nullable=False),
        "log_return": pa.Column(float, nullable=False),
    },
)

metrics_per_ticker_schema = pa.DataFrameSchema(
    {
        "ticker": pa.Column(pd.StringDtype(), nullable=False),
        "metric_name": pa.Column(pd.StringDtype(), nullable=False),
        "value": pa.Column(float, nullable=True),
        "category": pa.Column(
            pd.StringDtype(),
            nullable=False,
            checks=pa.Check.isin(["return", "risk", "risk_adjusted"]),
        ),
    }
)

rolling_metrics_schema = pa.DataFrameSchema(
    {
        "date": pa.Column("datetime64[ns]", nullable=False),
        "ticker": pa.Column(pd.StringDtype(), nullable=False),
        "metric_name": pa.Column(pd.StringDtype(), nullable=False),
        "value": pa.Column(float, nullable=True),
    }
)

drawdown_schema = pa.DataFrameSchema(
    {
        "date": pa.Column("datetime64[ns]", nullable=False),
        "ticker": pa.Column(pd.StringDtype(), nullable=False),
        "close": pa.Column(float, checks=pa.Check.gt(0), nullable=False),
        "running_peak": pa.Column(float, checks=pa.Check.gt(0), nullable=False),
        "drawdown_pct": pa.Column(float, checks=pa.Check.le(0), nullable=False),
    }
)

forecasts_schema = pa.DataFrameSchema(
    {
        "scenario_name": pa.Column(pd.StringDtype(), nullable=False),
        "ticker": pa.Column(pd.StringDtype(), nullable=False),
        "model_type": pa.Column(pd.StringDtype(), nullable=False),
        "target": pa.Column(pd.StringDtype(), nullable=False),
        "forecast_date": pa.Column("datetime64[ns]", nullable=False),
        "forecast": pa.Column(float, nullable=False),
        "lower_ci": pa.Column(float, nullable=False),
        "upper_ci": pa.Column(float, nullable=False),
        "confidence_level": pa.Column(float, nullable=False),
    }
)

forecast_metrics_schema = pa.DataFrameSchema(
    {
        "scenario_name": pa.Column(pd.StringDtype(), nullable=False),
        "ticker": pa.Column(pd.StringDtype(), nullable=False),
        "model_type": pa.Column(pd.StringDtype(), nullable=False),
        "metric_name": pa.Column(pd.StringDtype(), nullable=False),
        "value": pa.Column(float, nullable=True),
    }
)

stationarity_schema = pa.DataFrameSchema(
    {
        "ticker": pa.Column(pd.StringDtype(), nullable=False),
        "series_type": pa.Column(pd.StringDtype(), nullable=False),
        "adf_statistic": pa.Column(float, nullable=True),
        "p_value": pa.Column(float, nullable=True),
        "is_stationary": pa.Column(bool, nullable=False),
        "critical_1pct": pa.Column(float, nullable=True),
        "critical_5pct": pa.Column(float, nullable=True),
    }
)
