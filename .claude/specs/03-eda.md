# SPEC: Exploratory Data Analysis Module (Multi-Asset Portfolio)

**Module:** `src/eda.py`
**Pipeline Position:** 3 of 7
**Owner:** Analytics Layer
**Status:** Not started
**Depends on:** `src/data_cleaning.py` (Step 2) — consumes `data/processed/prices_clean.parquet`, `data/processed/returns_daily.parquet`, and `data/raw/metadata.json`

---

## 1. Problem Statement

Build the third module of the pipeline. This module must read cleaned price and return data produced by `data_cleaning.py`, generate a deterministic set of exploratory charts and summary statistics, and persist them to `outputs/plots/` and `outputs/reports/`.

**Why this matters:**
- EDA is the bridge between clean data and quantitative modeling — assumptions made here (normality, stationarity, correlation structure) directly determine which models are valid in Steps 5 and 6
- Visual inspection catches data quality issues that schema validation cannot — flat-lined prices, suspicious gaps, sector concentration
- Stakeholder communication starts here. Charts produced in this step feed directly into the final report (Step 7)

**Pain point being solved:** Without a structured EDA module, analysts re-explore the same data inconsistently across notebooks. Centralising EDA produces a reproducible, version-controlled baseline that every downstream model and stakeholder can reference.

**Philosophical stance:** **Describe, don't decide.** EDA quantifies and visualises — it does not filter, transform, or modify the underlying data. All filtering decisions were made in Step 2. EDA is read-only on processed data.

---

## 2. Functional Requirements

| # | Requirement | Priority |
|---|---|---|
| FR1 | Load `prices_clean.parquet`, `returns_daily.parquet`, and `metadata.json` | Must |
| FR2 | Validate inputs against `prices_clean_schema` and `returns_schema` before analysis | Must |
| FR3 | Generate per-ticker price trend charts with 20-day and 50-day rolling means | Must |
| FR4 | Generate per-ticker volume charts | Must |
| FR5 | Generate per-ticker return distribution charts (histogram + Q-Q plot) | Must |
| FR6 | Generate side-by-side boxplot of returns across all tickers | Must |
| FR7 | Compute return summary statistics (mean, std, skew, kurtosis, min, max) per ticker | Must |
| FR8 | Generate rolling 30-day annualized volatility chart per ticker | Must |
| FR9 | Generate monthly volatility heatmap (month × ticker) | Must |
| FR10 | Generate correlation matrix heatmap (returns) | Must |
| FR11 | Identify top-N most correlated ticker pairs and generate pairwise scatter plots | Must |
| FR12 | Generate sector-grouped correlation analysis using `metadata.json` | Must |
| FR13 | Identify top-N largest single-day moves per ticker (table) | Must |
| FR14 | Cross-check EDA-detected anomalies against `cleaning_report.json` flagged outliers | Must |
| FR15 | Persist all charts as PNG at `EDA_PLOT_DPI` resolution | Must |
| FR16 | Persist all summary tables to `outputs/reports/eda_summary.json` | Must |
| FR17 | Log per-block actions to `logs/eda_YYYY-MM-DD.log` | Must |
| FR18 | Idempotent: re-running on identical input must produce identical outputs | Must |
| FR19 | Use `metadata.json` sector field to colour-code charts where applicable | Should |

---

## 3. API Contracts

### 3.1 Inputs (from `config.py`)

```python
# Existing
RAW_DATA_DIR: pathlib.Path           # Path("data/raw")
PROCESSED_DATA_DIR: pathlib.Path     # Path("data/processed")
LOG_DIR: pathlib.Path                # Path("logs")

# New for Step 3
PLOTS_DIR: pathlib.Path              # Path("outputs/plots")
REPORTS_DIR: pathlib.Path            # Path("outputs/reports")
EDA_PLOT_DPI: int                    # 300 — print-quality PNG
EDA_ROLLING_WINDOWS: list[int]       # [20, 50] — for price trend overlays
EDA_VOL_WINDOW: int                  # 30 — rolling volatility window in days
EDA_TRADING_DAYS_PER_YEAR: int       # 252 — annualization factor
EDA_TOP_N_CORRELATIONS: int          # 3 — number of top correlated pairs to scatter
EDA_TOP_N_MOVES: int                 # 10 — top single-day moves per ticker
EDA_PLOT_STYLE: str                  # "seaborn-v0_8-whitegrid" — matplotlib style
```

### 3.2 Public Functions

```python
def load_processed_data(
    processed_dir: Path,
    raw_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Read prices_clean.parquet, returns_daily.parquet, and metadata.json.
    Validate against pandera schemas before returning.
    Raise FileNotFoundError if any file is missing with hint to run prior steps.
    Returns (prices, returns, metadata).
    """
```

---

```python
def plot_price_trends(
    prices: pd.DataFrame,
    rolling_windows: list[int],
    out_dir: Path,
    dpi: int,
) -> list[Path]:
    """
    For each ticker, generate a line chart of adjusted close price with
    rolling mean overlays for each window in rolling_windows.
    Saves to out_dir/01_price_trends/{ticker}_price.png.
    Returns list of saved file paths.
    """
```

---

```python
def plot_volume_trends(
    prices: pd.DataFrame,
    out_dir: Path,
    dpi: int,
) -> list[Path]:
    """
    For each ticker, generate a bar chart of daily volume.
    Saves to out_dir/01_price_trends/{ticker}_volume.png.
    Returns list of saved file paths.
    """
```

---

```python
def plot_return_distributions(
    returns: pd.DataFrame,
    out_dir: Path,
    dpi: int,
) -> tuple[list[Path], pd.DataFrame]:
    """
    Per ticker, generate:
      - Histogram of simple_return with KDE overlay
      - Q-Q plot vs normal distribution
    Also generate one combined boxplot across all tickers.
    Saves to out_dir/02_return_distributions/.
    Returns (saved_paths, distribution_stats_df) where stats_df has columns:
      [ticker, mean, std, skew, kurtosis, min, max, jarque_bera_pvalue].
    """
```

---

```python
def plot_volatility(
    returns: pd.DataFrame,
    window: int,
    trading_days_per_year: int,
    out_dir: Path,
    dpi: int,
) -> tuple[list[Path], pd.DataFrame]:
    """
    Compute rolling volatility = returns.rolling(window).std() * sqrt(trading_days_per_year).
    Generate:
      - One line chart per ticker showing rolling annualized volatility
      - One heatmap of monthly volatility (month × ticker)
    Saves to out_dir/03_volatility/.
    Returns (saved_paths, monthly_vol_df) where monthly_vol_df is pivoted month × ticker.
    """
```

---

```python
def plot_correlations(
    returns: pd.DataFrame,
    metadata: dict,
    top_n_pairs: int,
    out_dir: Path,
    dpi: int,
) -> tuple[list[Path], pd.DataFrame, list[tuple[str, str, float]]]:
    """
    Generate:
      - Correlation matrix heatmap of simple_return across all tickers
      - Pairwise scatter plots for the top_n_pairs most correlated (absolute) pairs
      - Sector-grouped correlation heatmap using metadata sector field
    Saves to out_dir/04_correlations/.
    Returns (saved_paths, corr_matrix_df, top_pairs) where top_pairs is
    [(ticker_a, ticker_b, correlation), ...] sorted by |correlation| desc.
    """
```

---

```python
def detect_outliers(
    returns: pd.DataFrame,
    cleaning_report: dict,
    top_n: int,
) -> dict:
    """
    For each ticker, identify the top_n largest absolute single-day returns.
    Cross-reference against cleaning_report['actions']['outliers_flagged'].
    Returns:
      {
        "top_moves_per_ticker": {ticker: [{date, simple_return, log_return}, ...]},
        "eda_vs_cleaning_match": {ticker: {"eda_count": int, "cleaning_count": int, "overlap": int}},
        "zero_volume_days": [{date, ticker}, ...],
        "zero_price_change_days": [{date, ticker, consecutive_days}, ...]
      }
    """
```

---

```python
def save_eda_summary(
    distribution_stats: pd.DataFrame,
    monthly_vol: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    top_pairs: list[tuple[str, str, float]],
    outlier_report: dict,
    saved_plot_paths: list[Path],
    out_dir: Path,
    *,
    input_data: dict | None = None,
    sector_avg_correlation: dict | None = None,
    run_timestamp: str | None = None,
    duration_sec: float = 0.0,
) -> Path:
    """
    Write a structured summary to out_dir/eda_summary.json (indent=2).
    Create out_dir if missing. Overwrite without prompting.
    Keyword-only args enrich the JSON with pipeline context:
      input_data — row/ticker counts from load_processed_data
      sector_avg_correlation — per-sector mean correlation from plot_correlations
      run_timestamp — ISO-8601 UTC string (defaults to now)
      duration_sec — wall-clock seconds elapsed before save
    Returns path to saved JSON.
    """
```

---

```python
def run_eda() -> dict:
    """
    Orchestrator. Called by main.py.
    Pipeline:
      load_processed_data → validate schemas
      → plot_price_trends → plot_volume_trends → plot_return_distributions
      → plot_volatility → plot_correlations → detect_outliers
      → save_eda_summary

    Returns a summary dict:
      {
        "plots_generated": int,
        "tickers_analyzed": int,
        "anomalies_flagged": int,
        "summary_path": str,
        "duration_sec": float
      }
    """
```

---

## 4. Output Contracts

### 4.1 `outputs/plots/` Directory Structure

```
outputs/plots/
├── 01_price_trends/
│   ├── AAPL_price.png       # one per ticker
│   ├── AAPL_volume.png
│   └── ...
├── 02_return_distributions/
│   ├── AAPL_histogram.png   # one per ticker
│   ├── AAPL_qqplot.png
│   └── all_tickers_boxplot.png
├── 03_volatility/
│   ├── AAPL_rolling_vol.png # one per ticker
│   └── monthly_vol_heatmap.png
├── 04_correlations/
│   ├── correlation_matrix.png
│   ├── sector_correlation.png
│   └── top_pair_{A}_{B}_scatter.png  # one per top correlated pair
└── 05_outliers/
    └── (no plots — summary in eda_summary.json)
```

All PNGs saved at `EDA_PLOT_DPI` resolution with transparent or white background.

### 4.2 `outputs/reports/eda_summary.json`

```json
{
  "run_timestamp": "2026-05-24T10:42:11Z",
  "input": {
    "prices_rows": 9750,
    "returns_rows": 9743,
    "tickers": 7,
    "date_range": ["2023-01-03", "2024-12-30"]
  },
  "distribution_stats": {
    "AAPL": {
      "mean": 0.00102, "std": 0.01823, "skew": -0.12, "kurtosis": 4.31,
      "min": -0.0987, "max": 0.0823, "jarque_bera_pvalue": 0.0012
    }
  },
  "monthly_volatility": {
    "2023-01": {"AAPL": 0.21, "MSFT": 0.18, "...": "..."},
    "2023-02": {"...": "..."}
  },
  "correlations": {
    "matrix": {
      "AAPL": {"AAPL": 1.0, "MSFT": 0.72, "...": "..."}
    },
    "top_pairs": [
      {"ticker_a": "AAPL", "ticker_b": "MSFT", "correlation": 0.72},
      {"ticker_a": "JPM", "ticker_b": "XOM", "correlation": 0.54}
    ],
    "sector_avg_correlation": {
      "Technology": 0.68,
      "Financials": 0.0,
      "Energy": 0.0
    }
  },
  "outliers": {
    "top_moves_per_ticker": {
      "AAPL": [
        {"date": "2024-08-05", "simple_return": -0.078, "log_return": -0.081}
      ]
    },
    "eda_vs_cleaning_match": {
      "AAPL": {"eda_count": 10, "cleaning_count": 1, "overlap": 1}
    },
    "zero_volume_days": [],
    "zero_price_change_days": []
  },
  "plots_generated": {
    "total": 47,
    "by_block": {
      "01_price_trends": 14,
      "02_return_distributions": 15,
      "03_volatility": 8,
      "04_correlations": 5,
      "05_outliers": 0
    }
  },
  "duration_sec": 8.21
}
```

---

## 5. Constraints

| Constraint | Reason |
|---|---|
| Do NOT import `yfinance` or make network calls | Only `data_ingestion.py` may touch the network |
| Do NOT recompute returns — read from `returns_daily.parquet` | Single source of truth; recomputing risks inconsistency |
| Do NOT modify or write to `data/processed/` | EDA is read-only on processed data |
| Use `matplotlib` + `seaborn` for static charts (not `plotly` in v1) | Reproducible, embeddable in PDF reports later |
| Long-format DataFrames only | Wide format breaks when tickers are added/dropped |
| All thresholds, window sizes, DPI via `config.py` | Reproducibility, scenario testing |
| `loguru` for logging; no `print()` in `src/eda.py` | Persistent logs, structured output |
| All plots saved at consistent DPI from config | Print-quality consistency across reports |
| Idempotent: same inputs → identical PNGs and JSON | Reproducibility |
| Validate inputs via pandera schemas at module entry | Catch contract violations early |
| Close matplotlib figures explicitly after saving (`plt.close()`) | Prevent memory leaks in long runs |
| File naming convention: `{block_num}_{plot_type}_{ticker}.png` | Sortable, predictable |

---

## 6. Edge Cases & Error Handling

| Case | Expected Behavior |
|---|---|
| `prices_clean.parquet` missing | Raise `FileNotFoundError` with hint: "Run Step 2 first" |
| `returns_daily.parquet` missing | Raise `FileNotFoundError` with hint: "Run Step 2 first" |
| `metadata.json` missing | Log warning, skip sector-grouped analysis, continue |
| `cleaning_report.json` missing | Log warning, skip EDA-vs-cleaning cross-check, continue |
| Single ticker in dataset | Skip correlation analysis, log warning |
| Ticker has fewer rows than `EDA_VOL_WINDOW` | Skip volatility chart for that ticker, log warning |
| Ticker has fewer rows than 30 days | Skip Q-Q plot for that ticker (insufficient data) |
| All tickers in same sector | Sector correlation degenerates to single block; log info |
| Existing files in `outputs/plots/` | Overwrite without prompting (idempotent reruns) |
| Missing `sector` in metadata for a ticker | Group under `"Unknown"` sector |
| `EDA_TOP_N_CORRELATIONS > num_pairs` | Use all available pairs, log warning |
| Returns all zero for a ticker (flat-lined) | Generate plots but flag in summary as `"flat_lined": true` |
| matplotlib backend issue (no display) | Use `Agg` backend explicitly — works headless |
| `outputs/plots/` or `outputs/reports/` doesn't exist | Create them (`mkdir(parents=True, exist_ok=True)`) |

---

## 7. Acceptance Criteria

The module is considered complete when **all** the following are true:

- [ ] `python main.py` executes `run_eda()` as Step 3 after Step 2
- [ ] `outputs/plots/` contains all five block subdirectories with expected PNGs
- [ ] `outputs/reports/eda_summary.json` exists with all fields from §4.2
- [ ] Number of price trend PNGs = `2 × num_tickers` (price + volume per ticker)
- [ ] `correlation_matrix.png` exists and shows all surviving tickers
- [ ] `monthly_vol_heatmap.png` exists with rows = months, columns = tickers
- [ ] Re-running `main.py` produces byte-identical JSON output (idempotent)
- [ ] Removing `metadata.json` does NOT crash the pipeline — sector analysis is skipped with a warning
- [ ] Setting `EDA_TOP_N_CORRELATIONS = 1` produces exactly one scatter plot
- [ ] `logs/eda_YYYY-MM-DD.log` shows per-block actions
- [ ] No module other than `eda.py` writes to `outputs/plots/`
- [ ] All public functions have type hints and docstrings
- [ ] No hardcoded thresholds, tickers, window sizes, or paths in `src/eda.py`
- [ ] At least 7 unit tests in `tests/unit/test_eda.py` pass
- [ ] No `print()` calls in `src/eda.py`
- [ ] All matplotlib figures explicitly closed (no memory leak warnings)

---

## 8. File Deliverables

| File | Purpose |
|---|---|
| `src/eda.py` | Module implementation |
| `config.py` | Add the new variables listed in §3.1 |
| `main.py` | Uncomment & integrate Step 3 |
| `tests/unit/test_eda.py` | Unit tests (see §10) |
| `outputs/plots/.gitkeep` | Ensure folder is tracked |
| `outputs/reports/.gitkeep` | Ensure folder is tracked |
| `requirements.txt` | Confirm `matplotlib`, `seaborn`, `scipy` are present |

---

## 9. `main.py` Integration

```python
# ── Step 3: EDA ───────────────────────────────────────────────────────
logger.info("STEP 3/7 — Exploratory Data Analysis")
eda_summary = eda.run_eda()
logger.info(f"EDA summary: {eda_summary}")
```

Uncomment the `from src import eda` import at the top.

---

## 10. Unit Test Plan

`tests/unit/test_eda.py` — minimum tests:

| # | Test | What it proves |
|---|---|---|
| 1 | `test_load_processed_data_raises_on_missing_file` | FileNotFoundError raised with helpful hint when parquet missing |
| 2 | `test_plot_price_trends_creates_expected_files` | One price PNG per ticker saved at correct path |
| 3 | `test_plot_return_distributions_returns_stats_df` | Stats DataFrame has all expected columns and correct ticker count |
| 4 | `test_plot_volatility_skips_tickers_below_window` | Tickers with insufficient data are skipped, not crashed |
| 5 | `test_plot_correlations_top_pairs_sorted_by_abs_value` | Top pairs returned in descending absolute correlation order |
| 6 | `test_detect_outliers_returns_top_n_per_ticker` | Exactly top_n moves returned per ticker |
| 7 | `test_save_eda_summary_writes_valid_json` | JSON file written, parseable, contains required keys |
| 8 | `test_idempotent_rerun` | Running EDA twice produces identical JSON output |
| 9 | `test_handles_missing_metadata_gracefully` | Missing metadata.json → warning logged, pipeline continues |
| 10 | `test_no_modifications_to_input_dataframes` | Input prices/returns DataFrames unchanged after run |

Use small synthetic DataFrames as fixtures in `conftest.py`. Use `tmp_path` for output directories. Do NOT load real data from `data/processed/` in unit tests.

---

## 11. Manual Test Plan (After Implementation)

```bash
# 1. Fresh run after Step 2
rm -rf outputs/plots/ outputs/reports/
python main.py

# 2. Verify outputs exist
ls outputs/plots/
# Expected: 01_price_trends, 02_return_distributions, 03_volatility,
#           04_correlations, 05_outliers

ls outputs/reports/
# Expected: eda_summary.json

# 3. Count plots per block
find outputs/plots -name "*.png" | wc -l
# Expected: roughly 2*N + 2*N + (N+1) + (3+top_n) where N = num tickers

# 4. Inspect summary
cat outputs/reports/eda_summary.json | python -m json.tool | head -50

# 5. Verify distribution stats
python -c "
import json
s = json.load(open('outputs/reports/eda_summary.json'))
for t, stats in s['distribution_stats'].items():
    print(f\"{t}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, kurt={stats['kurtosis']:.2f}\")
"

# 6. Check correlation matrix is symmetric and diagonal == 1
python -c "
import json
import numpy as np
s = json.load(open('outputs/reports/eda_summary.json'))
m = s['correlations']['matrix']
tickers = list(m.keys())
mat = np.array([[m[a][b] for b in tickers] for a in tickers])
assert np.allclose(np.diag(mat), 1.0), 'Diagonal must be 1.0'
assert np.allclose(mat, mat.T), 'Matrix must be symmetric'
print('Correlation matrix OK ✓')
"

# 7. Idempotency — checksum before and after rerun should match
md5sum outputs/reports/eda_summary.json
python main.py
md5sum outputs/reports/eda_summary.json  # must match

# 8. Resilience — remove metadata, rerun, confirm pipeline does not crash
mv data/raw/metadata.json data/raw/metadata.json.bak
python main.py
# Should complete with warning about sector analysis skipped
mv data/raw/metadata.json.bak data/raw/metadata.json

# 9. Visually inspect a sample of plots
# Open outputs/plots/01_price_trends/AAPL_price.png
# Open outputs/plots/04_correlations/correlation_matrix.png
```

---

## 12. Out of Scope (Explicit Non-Goals)

- ❌ Interactive plots (plotly, bokeh) — static PNG only in v1; plotly belongs in Step 7 export
- ❌ Statistical hypothesis testing beyond Jarque-Bera normality — full inference belongs in Step 4 metrics
- ❌ Time-series decomposition (trend/seasonal/residual) — belongs in Step 5 forecasting
- ❌ Risk metrics (VaR, CVaR, Sharpe, drawdown) — belongs in Step 4 metrics
- ❌ Comparison to benchmark index (S&P 500) — not in v1 scope
- ❌ Factor analysis (Fama-French, PCA) — not in v1 scope
- ❌ Modifying or filtering data — EDA is strictly read-only on processed data
- ❌ Generating PDF/HTML reports — belongs in Step 7 export
- ❌ Dashboards (Streamlit, Dash) — not in v1 scope

---

## 13. Notes for Claude Code

- Implement functions in the order listed in §3.2 — each plot function is independently testable
- After each function, write its unit test before moving to the next (TDD-lite)
- Use `matplotlib.use("Agg")` at module top to force headless backend
- Always call `plt.close(fig)` after `fig.savefig()` to prevent memory leaks
- For seaborn plots, use `sns.set_style()` based on `EDA_PLOT_STYLE` config
- For PNG writes use `fig.savefig(path, dpi=EDA_PLOT_DPI, bbox_inches="tight")`
- Use `loguru.logger` everywhere; reserve `print()` for the final summary line in `main.py`
- Keep `run_eda()` thin — it should read like a table of contents, not contain plotting logic
- For correlation matrix, use `returns.pivot(index="date", columns="ticker", values="simple_return").corr()` — long-to-wide pivot only inside this function, never persist wide format
- For sector analysis, join returns with metadata sector field inside `plot_correlations` — do not pre-join globally
- If something seems missing from this spec, flag it before adding silently
- Do NOT add EDA steps not listed here (e.g. autocorrelation plots, candlestick charts, drawdown plots) — flag them as scope discussion items first
- The `detect_outliers` function does NOT need plots — it produces a structured dict only. The `05_outliers/` directory exists for future expansion