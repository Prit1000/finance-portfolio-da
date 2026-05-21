# SPEC: Data Cleaning Module (Multi-Asset Portfolio)

**Module:** `src/data_cleaning.py`
**Pipeline Position:** 2 of 7
**Owner:** Data Engineering Layer
**Status:** Not started
**Depends on:** `src/data_ingestion.py` (Step 1) — consumes `data/raw/prices_raw.csv` and `data/raw/metadata.json`

---

## 1. Problem Statement

Build the second module of the pipeline. This module must read raw OHLCV data produced by `data_ingestion.py`, apply a deterministic set of cleaning rules, and persist analysis-ready datasets to `data/processed/`.

**Why this matters:**
- Raw Yahoo data has gaps, occasional duplicates, missing volume on halt days, and the odd zero/negative price from data feed errors
- Downstream modules (EDA, metrics, forecasting, Monte Carlo) assume clean, gap-free, schema-validated input — they should not contain any defensive cleaning code
- Returns must be computed once, here, and reused everywhere — recomputing in every module is a source of bugs and inconsistency

**Pain point being solved:** Without this module, every metric and model would have to handle missing values, outliers, and schema drift independently. Centralising cleaning means one place to audit, one place to fix bugs, one place to explain decisions to stakeholders.

**Philosophical stance:** **Flag, don't delete.** In finance, a 20% single-day move is often real (earnings, COVID, flash crash). Cleaning preserves data fidelity and produces a separate audit trail — it does not silently rewrite history.

---

## 2. Functional Requirements

| # | Requirement | Priority |
|---|---|---|
| FR1 | Load `prices_raw.csv` and `metadata.json` from `data/raw/` | Must |
| FR2 | Validate raw schema against expected dtypes; raise on mismatch | Must |
| FR3 | Remove exact duplicate `(date, ticker)` rows, keeping first occurrence | Must |
| FR4 | Reindex each ticker's series to the configured trading calendar (e.g. NYSE) | Must |
| FR5 | Forward-fill missing `close` prices up to `MAX_CONSECUTIVE_FILLS` days; drop rows beyond that | Must |
| FR6 | Drop rows with zero or negative `open`/`high`/`low`/`close` | Must |
| FR7 | Drop entire ticker if data coverage < `MIN_COVERAGE_PCT` of expected trading days | Must |
| FR8 | Flag (do not modify) daily returns exceeding `OUTLIER_RETURN_THRESHOLD` | Must |
| FR9 | Compute daily simple return and log return per ticker | Must |
| FR10 | Persist cleaned prices to `data/processed/prices_clean.parquet` | Must |
| FR11 | Persist returns to `data/processed/returns_daily.parquet` | Must |
| FR12 | Persist flagged outliers to `data/processed/flagged_observations.parquet` | Must |
| FR13 | Write a structured `cleaning_report.json` summarising every change made | Must |
| FR14 | Validate final cleaned DataFrame against `prices_clean_schema` (pandera) before writing | Must |
| FR15 | Log per-ticker cleaning actions to `logs/cleaning_YYYY-MM-DD.log` | Must |
| FR16 | Idempotent: re-running on already-clean data must produce identical outputs | Must |
| FR17 | Warn (do not fail) if metadata contains >1 unique currency among surviving tickers | Should |

---

## 3. API Contracts

### 3.1 Inputs (from `config.py`)

```python
# Existing from Step 1
RAW_DATA_DIR: pathlib.Path           # Path("data/raw")
LOG_DIR: pathlib.Path                # Path("logs")

# New for Step 2
PROCESSED_DATA_DIR: pathlib.Path     # Path("data/processed")
MIN_COVERAGE_PCT: float              # 0.80 — drop ticker if below this
OUTLIER_RETURN_THRESHOLD: float      # 0.25 — flag |daily return| > 25%
FILL_METHOD: str                     # "ffill" — only ffill supported in v1
MAX_CONSECUTIVE_FILLS: int           # 3 — max consecutive days to forward-fill
TRADING_CALENDAR: str                # "NYSE" — passed to pandas_market_calendars
```

### 3.2 Public Functions

```python
def load_raw(raw_dir: Path) -> tuple[pd.DataFrame, dict]:
    """
    Read prices_raw.csv and metadata.json. Parse `date` to datetime,
    uppercase tickers, set explicit dtypes. Raise FileNotFoundError if
    either file is missing.
    """
```

---

```python
def validate_raw_schema(df: pd.DataFrame) -> None:
    """
    Assert columns, dtypes, and required fields match Step 1's output contract.
    Raise ValueError with a descriptive message on mismatch.
    """
```

---

```python
def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Drop exact duplicate (date, ticker) rows. Keep first.
    Returns (deduped_df, num_duplicates_removed).
    """
```

---

```python
def reindex_to_calendar(df: pd.DataFrame, calendar_name: str) -> pd.DataFrame:
    """
    For each ticker, reindex to the full set of expected trading days within
    [df.date.min(), df.date.max()] using pandas_market_calendars.
    Missing trading days become NaN rows (to be handled by handle_missing).
    """
```

---

```python
def handle_missing(
    df: pd.DataFrame,
    method: str,
    max_consecutive: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Per ticker:
      - Forward-fill close, open, high, low up to max_consecutive days.
      - Beyond max_consecutive, drop the rows.
      - Volume: do NOT fill (zero volume is meaningful — halt day).
    Returns (filled_df, fills_per_ticker).
    """
```

---

```python
def drop_invalid_prices(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Drop rows where any of open/high/low/close <= 0.
    Returns (filtered_df, num_rows_dropped).
    """
```

---

```python
def enforce_coverage(
    df: pd.DataFrame,
    min_pct: float,
    calendar_name: str,
) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    """
    For each ticker, compute coverage = actual_days / expected_trading_days.
    Drop tickers below min_pct.
    Returns (filtered_df, dropped_tickers, coverage_per_ticker).
    """
```

---

```python
def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per ticker, sorted by date:
      simple_return = close.pct_change()
      log_return = np.log(close / close.shift(1))
    First row per ticker will have NaN returns — drop these.
    Returns long-format DataFrame: [date, ticker, simple_return, log_return].
    """
```

---

```python
def flag_outliers(
    returns: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify rows where |simple_return| > threshold.
    Returns (returns_unchanged, flagged_df).
    Flagged DataFrame schema: [date, ticker, simple_return, log_return, reason].
    Does NOT modify the returns DataFrame — flagging only.
    """
```

---

```python
def save_processed(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    flagged: pd.DataFrame,
    report: dict,
    out_dir: Path,
) -> None:
    """
    Write:
      - prices_clean.parquet
      - returns_daily.parquet
      - flagged_observations.parquet
      - cleaning_report.json (indent=2)
    Create out_dir if missing. Overwrite existing files without prompting.
    """
```

---

```python
def run_cleaning() -> dict:
    """
    Orchestrator. Called by main.py.
    Pipeline:
      load_raw → validate_raw_schema → remove_duplicates → reindex_to_calendar
      → handle_missing → drop_invalid_prices → enforce_coverage
      → compute_returns → flag_outliers → schema_validate → save_processed

    Returns the cleaning_report dict (also persisted to JSON).
    """
```

---

## 4. Output Contracts

### 4.1 `data/processed/prices_clean.parquet`

Long format, one row per (date, ticker). Schema-validated via pandera.

| Column | dtype | Constraint |
|---|---|---|
| `date` | `datetime64[ns]` | Not null; aligned to trading calendar |
| `ticker` | `string` | Uppercase; in surviving ticker list |
| `open` | `float64` | > 0 |
| `high` | `float64` | > 0; >= low |
| `low` | `float64` | > 0; <= high |
| `close` | `float64` | > 0 |
| `volume` | `Int64` | >= 0; nullable |

Uniqueness: `(date, ticker)` is a composite primary key.

### 4.2 `data/processed/returns_daily.parquet`

| Column | dtype | Notes |
|---|---|---|
| `date` | `datetime64[ns]` | |
| `ticker` | `string` | |
| `simple_return` | `float64` | `close.pct_change()` |
| `log_return` | `float64` | `ln(close_t / close_t-1)` |

First trading day per ticker excluded (no prior close).

### 4.3 `data/processed/flagged_observations.parquet`

| Column | dtype | Notes |
|---|---|---|
| `date` | `datetime64[ns]` | |
| `ticker` | `string` | |
| `simple_return` | `float64` | |
| `log_return` | `float64` | |
| `reason` | `string` | e.g. `"abs_return_exceeds_threshold"` |

May be empty (zero rows) if no outliers detected — file still written.

### 4.4 `data/processed/cleaning_report.json`

```json
{
  "run_timestamp": "2026-05-21T10:42:11Z",
  "input": {
    "rows": 9821,
    "tickers": 7,
    "date_range": ["2023-01-03", "2024-12-30"]
  },
  "output": {
    "rows": 9750,
    "tickers": 7,
    "date_range": ["2023-01-03", "2024-12-30"]
  },
  "actions": {
    "duplicates_removed": 0,
    "invalid_price_rows_dropped": 2,
    "missing_filled": {"AAPL": 2, "JPM": 1},
    "rows_dropped_after_fill_limit": 0,
    "tickers_dropped_low_coverage": [],
    "outliers_flagged": {"XOM": 3, "AAPL": 1}
  },
  "coverage_pct": {
    "AAPL": 0.998, "MSFT": 1.000, "GOOGL": 0.998,
    "JPM": 0.996, "XOM": 1.000, "JNJ": 1.000, "WMT": 1.000
  },
  "currency_warning": null,
  "duration_sec": 1.34
}
```

---

## 5. Constraints

| Constraint | Reason |
|---|---|
| Do NOT import `yfinance` or make network calls | Only `data_ingestion.py` may touch the network |
| Use **Parquet** for processed outputs, not CSV | Typed, compressed, 10× faster reads downstream |
| Use **pandera** for schema validation on final output | Catches contract violations before downstream modules consume bad data |
| Use `pandas_market_calendars` for trading day reindexing | Avoids hand-coded holiday lists; correct across exchanges |
| Long-format only | Wide format breaks when tickers are added/dropped |
| All thresholds via `config.py` — no magic numbers | Reproducibility, scenario testing |
| `loguru` for logging; `print()` only in final summary | Persistent logs, structured output |
| Flag outliers, do NOT modify them | Preserves data fidelity; real market events are not bugs |
| Volume is never forward-filled | Zero/missing volume is meaningful (halt, holiday) |
| Cleaning must be deterministic | Same input → byte-identical output |

---

## 6. Edge Cases & Error Handling

| Case | Expected Behavior |
|---|---|
| `prices_raw.csv` missing | Raise `FileNotFoundError` with hint: "Run Step 1 first" |
| `metadata.json` missing | Log warning, continue (cleaning does not depend on metadata) |
| Raw CSV has unexpected columns | Raise `ValueError` from `validate_raw_schema` |
| Ticker has fewer rows than 2 (cannot compute return) | Drop ticker, log warning |
| All tickers fail coverage check | Raise `RuntimeError` — pipeline cannot continue with empty data |
| `MAX_CONSECUTIVE_FILLS = 0` | Disables filling; drop any row with missing close |
| Outlier threshold = 0 | Every non-zero return flagged. Allowed (useful for debugging). |
| Trading calendar returns no days for given range | Raise `ValueError` — bad date range or calendar name |
| Existing files in `data/processed/` | Overwrite without prompting (idempotent reruns) |
| Volume is negative | Treat as invalid → set to NaN, log warning |
| `high < low` in a row | Drop row, log warning (data feed error) |
| Mixed currencies among surviving tickers | Write `currency_warning` field in report; do not fail |
| `pandas_market_calendars` not installed | Raise `ImportError` with install hint |

---

## 7. Acceptance Criteria

The module is considered complete when **all** the following are true:

- [ ] `python main.py` executes `run_cleaning()` as Step 2 after Step 1
- [ ] `data/processed/prices_clean.parquet` exists and passes `prices_clean_schema.validate()`
- [ ] `data/processed/returns_daily.parquet` exists with both `simple_return` and `log_return` columns
- [ ] `data/processed/flagged_observations.parquet` exists (may be empty)
- [ ] `data/processed/cleaning_report.json` exists with all fields from §4.4
- [ ] No row in `prices_clean.parquet` has zero or negative OHLC values
- [ ] No duplicate `(date, ticker)` pairs in `prices_clean.parquet`
- [ ] Re-running `main.py` produces byte-identical Parquet files (idempotent)
- [ ] Introducing an artificially-corrupted ticker (e.g. inject a negative price into raw) results in that row being dropped and logged — pipeline does not crash
- [ ] Setting `MIN_COVERAGE_PCT = 0.99` with a sparse ticker drops it cleanly
- [ ] `logs/cleaning_YYYY-MM-DD.log` shows per-ticker actions
- [ ] No module other than `data_cleaning.py` imports `pandas_market_calendars`
- [ ] All public functions have type hints and docstrings
- [ ] No hardcoded thresholds, tickers, dates, or paths in `src/data_cleaning.py`
- [ ] At least 6 unit tests in `tests/unit/test_data_cleaning.py` pass

---

## 8. File Deliverables

| File | Purpose |
|---|---|
| `src/data_cleaning.py` | Module implementation |
| `src/schemas.py` | Pandera schemas (`prices_clean_schema`, `returns_schema`) — new file |
| `config.py` | Add the new variables listed in §3.1 |
| `main.py` | Uncomment & integrate Step 2 |
| `tests/unit/test_data_cleaning.py` | Unit tests (see §10) |
| `data/processed/.gitkeep` | Ensure folder is tracked |
| `requirements.txt` | Add `pandera`, `pandas_market_calendars`, `pyarrow` |

---

## 9. `main.py` Integration

```python
# ── Step 2: Data Cleaning ─────────────────────────────────────────────
logger.info("STEP 2/7 — Data Cleaning")
cleaning_summary = data_cleaning.run_cleaning()
logger.info(f"Cleaning summary: {cleaning_summary}")
```

Uncomment the `from src import data_cleaning` import at the top.

---

## 10. Unit Test Plan

`tests/unit/test_data_cleaning.py` — minimum tests:

| # | Test | What it proves |
|---|---|---|
| 1 | `test_remove_duplicates_keeps_first` | Duplicate `(date, ticker)` rows are deduped |
| 2 | `test_drop_invalid_prices_removes_zero_and_negative` | Zero/negative OHLC rows are dropped |
| 3 | `test_handle_missing_respects_max_consecutive_fills` | Gaps > max fills are dropped, not filled forever |
| 4 | `test_handle_missing_does_not_fill_volume` | Volume NaN stays NaN |
| 5 | `test_enforce_coverage_drops_sparse_ticker` | Ticker with < min_pct days is removed |
| 6 | `test_compute_returns_first_row_dropped` | First trading day per ticker has no return row |
| 7 | `test_flag_outliers_does_not_modify_returns` | Returns DataFrame is unchanged after flagging |
| 8 | `test_schema_validation_catches_negative_price` | Pandera raises on invalid data |
| 9 | `test_idempotent_rerun` | Running cleaning twice on clean data produces identical output |

Use small synthetic DataFrames as fixtures in `conftest.py`. Do NOT load real data from `data/raw/` in unit tests.

---

## 11. Manual Test Plan (After Implementation)

```bash
# 1. Fresh run after Step 1
rm -rf data/processed/
python main.py

# 2. Verify outputs exist
ls data/processed/
# Expected: prices_clean.parquet, returns_daily.parquet,
#           flagged_observations.parquet, cleaning_report.json

# 3. Inspect cleaned prices
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/prices_clean.parquet')
print('Shape:', df.shape)
print('Dtypes:', df.dtypes.to_dict())
print('Tickers:', df['ticker'].unique())
print('NaN counts:', df.isna().sum().to_dict())
assert (df[['open','high','low','close']] > 0).all().all(), 'Found non-positive prices!'
print('All OHLC > 0 ✓')
"

# 4. Inspect returns
python -c "
import pandas as pd
r = pd.read_parquet('data/processed/returns_daily.parquet')
print('Shape:', r.shape)
print(r.groupby('ticker')['simple_return'].agg(['mean','std','min','max']))
"

# 5. View cleaning report
cat data/processed/cleaning_report.json | python -m json.tool

# 6. Idempotency — checksum before and after rerun should match
md5sum data/processed/prices_clean.parquet
python main.py
md5sum data/processed/prices_clean.parquet  # must match

# 7. Resilience — inject a bad row, rerun, confirm it's dropped
python -c "
import pandas as pd
df = pd.read_csv('data/raw/prices_raw.csv')
df.loc[0, 'close'] = -50.0
df.to_csv('data/raw/prices_raw.csv', index=False)
"
python main.py
# Check cleaning_report.json — invalid_price_rows_dropped should be >= 1
```

---

## 12. Out of Scope (Explicit Non-Goals)

- ❌ Currency normalization / FX conversion — only a warning if mixed currencies detected
- ❌ Survivorship bias correction
- ❌ Corporate actions beyond what `auto_adjust=True` from Step 1 handles
- ❌ Imputation methods beyond forward-fill (no interpolation, no model-based fills) in v1
- ❌ Outlier *correction* — flagging only
- ❌ Cross-ticker validation (e.g. comparing AAPL to a sector index)
- ❌ Intraday cleaning (only daily bars in v1)
- ❌ Streaming / incremental cleaning — always full reprocess

---

## 13. Notes for Claude Code

- Implement functions in the order listed in §3.2 — each function is independently testable
- After each function, write its unit test before moving to the next (TDD-lite)
- Use `pandas_market_calendars.get_calendar(TRADING_CALENDAR).schedule(start, end)` for trading days
- For Parquet writes use `df.to_parquet(path, engine="pyarrow", index=False, compression="snappy")`
- Use `loguru.logger` everywhere; reserve `print()` for the final summary line in `main.py`
- Keep `run_cleaning()` thin — it should read like a table of contents, not contain business logic
- If something seems missing from this spec, flag it before adding silently
- Do NOT add cleaning steps not listed here (e.g. winsorisation, z-score capping) — flag them as scope discussion items first
