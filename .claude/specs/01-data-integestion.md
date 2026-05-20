# SPEC: Data Ingestion Module (Multi-Asset Portfolio)

**Module:** `src/data_ingestion.py`
**Pipeline Position:** 1 of 7
**Owner:** Data Engineering Layer
**Status:** Not started

---

## 1. Problem Statement

Build the first module of a finance portfolio analysis pipeline. This module must fetch historical OHLCV price data and fundamental metadata for a **multi-asset portfolio** (5–10 tickers across sectors) from Yahoo Finance, and persist the results to local CSV/JSON files in `data/raw/`.

**Why this matters:**
- All downstream modules (cleaning, EDA, metrics, forecasting, Monte Carlo) depend on these files
- This is the **only** module allowed to make network calls — isolating I/O simplifies testing and reruns
- The pipeline must be reproducible: rerunning ingestion with the same config should produce identical outputs (subject to Yahoo data availability)

**Pain point being solved:** Without this module, every downstream step would hit the network independently, making the pipeline slow, brittle, and rate-limited.

---

## 2. Functional Requirements

| # | Requirement | Priority |
|---|---|---|
| FR1 | Fetch daily OHLCV data for a configurable list of tickers over a configurable date range | Must |
| FR2 | Fetch fundamental metadata (sector, industry, market cap, beta, currency, P/E, 52w high/low, short name) for each ticker | Must |
| FR3 | Use `auto_adjust=True` so all prices are split/dividend-adjusted | Must |
| FR4 | Retry failed requests up to `MAX_RETRIES` times with exponential backoff | Must |
| FR5 | Log each ticker's fetch status (success / retry / failure) to console and a log file | Must |
| FR6 | Persist prices as a long-format CSV (`prices_raw.csv`) | Must |
| FR7 | Persist metadata as JSON (`metadata.json`) | Must |
| FR8 | Skip tickers that fail after all retries but continue the run (do not crash the pipeline) | Must |
| FR9 | Print a summary at the end: total tickers requested, succeeded, failed, total rows fetched | Must |
| FR10 | Allow re-runs to overwrite existing raw files without prompting | Should |

---

## 3. API Contracts

### 3.1 Inputs (from `config.py`)

```python
TICKERS: list[str]        # e.g. ["AAPL", "MSFT", "GOOGL", "JPM", "XOM", "JNJ", "WMT"]
DATE_START: str           # "YYYY-MM-DD", e.g. "2020-01-01"
DATE_END: str             # "YYYY-MM-DD", e.g. "2024-12-31"
FETCH_INTERVAL: str       # "1d" (daily). Other values not supported in v1.
MAX_RETRIES: int          # 3
RAW_DATA_DIR: pathlib.Path  # Path("data/raw")
LOG_DIR: pathlib.Path     # Path("logs")
```

### 3.2 Public Functions

```python
def fetch_prices(tickers: list[str], start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV for all tickers in a single yf.download() call (batch is faster than looping).
    Returns long-format DataFrame.
    """
```

**Output schema (`pd.DataFrame`):**

| Column | dtype | Description |
|---|---|---|
| `date` | `datetime64[ns]` | Trading day |
| `ticker` | `string` | Uppercase symbol |
| `open` | `float64` | Adjusted open |
| `high` | `float64` | Adjusted high |
| `low` | `float64` | Adjusted low |
| `close` | `float64` | Adjusted close (primary for return calc) |
| `volume` | `int64` | Daily traded volume |

---

```python
def fetch_metadata(tickers: list[str]) -> dict[str, dict]:
    """
    Loop over tickers, call yf.Ticker(t).info. Extract a fixed set of keys.
    Missing keys → None (do not raise).
    """
```

**Output schema (`dict`):**

```json
{
  "AAPL": {
    "shortName": "Apple Inc.",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "marketCap": 3400000000000,
    "currency": "USD",
    "beta": 1.25,
    "trailingPE": 32.5,
    "fiftyTwoWeekHigh": 250.0,
    "fiftyTwoWeekLow": 160.0
  },
  "MSFT": { ... }
}
```

---

```python
def save_raw(prices: pd.DataFrame, metadata: dict, raw_dir: Path) -> None:
    """
    Persist prices → prices_raw.csv (no index column)
    Persist metadata → metadata.json (indent=2, ensure_ascii=False)
    Create raw_dir if it does not exist.
    """
```

---

```python
def run_ingestion() -> dict:
    """
    Orchestrator. Called by main.py.
    Returns a summary dict:
      {
        "tickers_requested": int,
        "tickers_succeeded": int,
        "tickers_failed": list[str],
        "rows_fetched": int,
        "duration_sec": float
      }
    """
```

---

## 4. Constraints

| Constraint | Reason |
|---|---|
| Use `yfinance` only — no paid APIs in v1 | Free, no API key needed |
| Single batch `yf.download()` for prices (not per-ticker loop) | ~10x faster, fewer rate-limit hits |
| `auto_adjust=True` mandatory | Raw close prices are wrong for return calcs (ignore splits/dividends) |
| Long-format output only | Wide-format breaks if a new ticker is added later |
| No data cleaning here | Separation of concerns — `data_cleaning.py` owns that |
| All paths via `pathlib.Path`, not string concat | Cross-platform safety |
| All config from `config.py` — no hardcoded values | Reproducibility, easier scenario testing |
| Use `loguru` for logging, not `print()` | Persistent log files, structured output |
| `tenacity` for retries on `fetch_metadata` (single-ticker calls) | yfinance .info is flaky, frequently times out |

---

## 5. Edge Cases & Error Handling

| Case | Expected Behavior |
|---|---|
| Ticker doesn't exist (e.g. typo `"AAPLE"`) | Log warning, exclude from outputs, add to `tickers_failed` list. Do not crash. |
| Ticker delisted mid-range | yfinance returns partial data — keep what's available, log the actual date range received |
| Network failure | Retry up to `MAX_RETRIES` with exponential backoff (1s, 2s, 4s). After exhaustion, log error and skip ticker. |
| `yf.Ticker(t).info` returns empty dict | Save ticker with all metadata fields = `None`, do not fail |
| `DATE_END` is in the future | Yahoo returns data up to last trading day — accept silently |
| `DATE_START` > `DATE_END` | Raise `ValueError` immediately before fetching |
| `data/raw/` doesn't exist | Create it (`mkdir(parents=True, exist_ok=True)`) |
| Existing `prices_raw.csv` from previous run | Overwrite without prompting (re-runs must be idempotent) |
| Empty `TICKERS` list | Raise `ValueError` — nothing to fetch |
| All tickers fail | Log critical error, return summary with `tickers_succeeded = 0`. Do not write empty CSV. |
| Mixed currencies in portfolio (USD + INR) | Save raw as-is. Currency normalization is a downstream concern. Log a warning if >1 unique currency detected. |

---

## 6. Acceptance Criteria

The module is considered complete when **all** the following are true:

- [ ] Running `python main.py` executes `run_ingestion()` as Step 1 of the pipeline
- [ ] `data/raw/prices_raw.csv` exists with the exact schema in §3.2
- [ ] `data/raw/metadata.json` exists with one entry per successful ticker
- [ ] Row count in `prices_raw.csv` ≈ `(num_trading_days × num_successful_tickers)` (±5% for holidays/halts)
- [ ] Re-running `main.py` produces files with identical row counts (idempotent)
- [ ] Introducing a fake ticker (e.g. `"FAKE123"`) into `config.TICKERS` does **not** crash the pipeline — it's logged and skipped
- [ ] Log file in `logs/ingestion_YYYY-MM-DD.log` shows per-ticker status
- [ ] Summary dict is printed to console at end of run
- [ ] No module other than `data_ingestion.py` imports `yfinance`
- [ ] All functions have type hints and docstrings
- [ ] No hardcoded tickers, dates, or paths anywhere in `src/data_ingestion.py`

---

## 7. File Deliverables

| File | Purpose |
|---|---|
| `src/data_ingestion.py` | Module implementation |
| `src/__init__.py` | Make `src` a package (empty file if not already present) |
| `config.py` | Add the variables listed in §3.1 |
| `main.py` | Integrate Step 1 — call `run_ingestion()` and print summary |
| `data/raw/.gitkeep` | Ensure folder is tracked |
| `logs/.gitkeep` | Ensure folder is tracked |

---

## 8. `main.py` Integration

After this module is built, `main.py` must look like this (testable end-to-end after Step 1 alone):

```python
"""
main.py — Finance Portfolio Analysis Pipeline
Run: python main.py
"""

from loguru import logger
from pathlib import Path

import config
from src import data_ingestion
# from src import data_cleaning       # TODO: Step 2
# from src import eda                  # TODO: Step 3
# from src import metrics              # TODO: Step 4
# from src import forecasting          # TODO: Step 5
# from src import monte_carlo          # TODO: Step 6
# from src import export               # TODO: Step 7


def main():
    logger.add(
        config.LOG_DIR / "pipeline_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        level="INFO",
    )
    logger.info("=" * 60)
    logger.info("PIPELINE START")
    logger.info("=" * 60)

    # ── Step 1: Data Ingestion ────────────────────────────────────────────
    logger.info("STEP 1/7 — Data Ingestion")
    summary = data_ingestion.run_ingestion()
    logger.info(f"Ingestion summary: {summary}")

    # ── Step 2: Data Cleaning ──────── (pending) ──────────────────────────
    # logger.info("STEP 2/7 — Data Cleaning")
    # data_cleaning.run_cleaning()

    # ── Step 3: EDA ──────────────── (pending) ────────────────────────────
    # ── Step 4: Metrics ──────────── (pending) ────────────────────────────
    # ── Step 5: Forecasting ──────── (pending) ────────────────────────────
    # ── Step 6: Monte Carlo ──────── (pending) ────────────────────────────
    # ── Step 7: Export ───────────── (pending) ────────────────────────────

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
```

---

## 9. Manual Test Plan (After Implementation)

Run these in order to verify Step 1 works standalone:

```bash
# 1. Fresh run
rm -rf data/raw/ logs/
python main.py

# 2. Verify outputs exist
ls data/raw/
# Expected: prices_raw.csv, metadata.json

# 3. Inspect prices
python -c "import pandas as pd; df = pd.read_csv('data/raw/prices_raw.csv'); print(df.shape); print(df.head()); print(df['ticker'].unique())"

# 4. Inspect metadata
python -c "import json; print(json.dumps(json.load(open('data/raw/metadata.json')), indent=2)[:500])"

# 5. Idempotency check — rerun and confirm row counts match
python main.py
# Row count should be identical

# 6. Resilience check — add "FAKE123" to TICKERS, rerun
# Pipeline should complete, FAKE123 should appear in tickers_failed
```

---

## 10. Out of Scope (Explicit Non-Goals)

- ❌ Real-time / streaming data
- ❌ Intraday bars (1-minute, 1-hour)
- ❌ Options chains, futures, crypto
- ❌ Multi-source ingestion (Alpha Vantage, IEX, etc.)
- ❌ Database persistence (Postgres, SQLite) — CSV/JSON is sufficient for v1
- ❌ Incremental updates (only fetch missing dates) — always full refetch in v1
- ❌ Data quality checks beyond "did it return rows" — that's `data_cleaning.py`'s job

---

## 11. Notes for Claude Code

- Implement functions in the order listed in §3.2 (`fetch_prices` → `fetch_metadata` → `save_raw` → `run_ingestion`)
- After each function, write a quick smoke test in a comment block — don't skip this
- Use `tenacity.retry` decorator for `fetch_metadata` per-ticker calls
- Use `loguru.logger` everywhere; do not use `print()` except in the final summary
- Do not add any feature not listed in this spec — if something seems missing, flag it instead of adding silently