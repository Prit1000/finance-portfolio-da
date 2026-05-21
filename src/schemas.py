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
