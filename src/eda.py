"""
src/eda.py — Step 3 of the Finance Portfolio Analysis Pipeline.

Reads cleaned price and return data produced by data_cleaning.py, generates
a deterministic set of exploratory charts and summary statistics, and persists
them to outputs/plots/ and outputs/reports/.

Constraints:
- No yfinance imports or network calls.
- No writes to data/processed/ (read-only on processed data).
- Long-format DataFrames only; wide format only as transient local vars.
- All thresholds, window sizes, and paths come from config.py.
- loguru for logging; no print() statements.
- matplotlib Agg backend for headless operation.
- plt.close(fig) after every fig.savefig() to prevent memory leaks.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy import stats

import config
from src.schemas import prices_clean_schema, returns_schema


# ── Public Functions ───────────────────────────────────────────────────────────


def load_processed_data(
    processed_dir: Path,
    raw_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Read prices_clean.parquet, returns_daily.parquet, and metadata.json.
    Validate against pandera schemas before returning.
    Raise FileNotFoundError if any parquet file is missing with hint to run Step 2.
    Returns (prices, returns, metadata).
    """
    prices_path = processed_dir / "prices_clean.parquet"
    returns_path = processed_dir / "returns_daily.parquet"
    metadata_path = raw_dir / "metadata.json"

    if not prices_path.exists():
        raise FileNotFoundError(
            f"{prices_path} not found. Run Step 2 first (data_cleaning.run_cleaning())."
        )
    if not returns_path.exists():
        raise FileNotFoundError(
            f"{returns_path} not found. Run Step 2 first (data_cleaning.run_cleaning())."
        )

    prices = pd.read_parquet(prices_path)
    returns = pd.read_parquet(returns_path)

    try:
        prices_clean_schema.validate(prices)
    except Exception as exc:
        raise RuntimeError(f"prices_clean.parquet failed schema validation: {exc}") from exc

    try:
        returns_schema.validate(returns)
    except Exception as exc:
        raise RuntimeError(f"returns_daily.parquet failed schema validation: {exc}") from exc

    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as fh:
            metadata: dict = json.load(fh)
        logger.info(f"Loaded metadata for {len(metadata)} tickers")
    else:
        logger.warning(f"{metadata_path} not found — sector-grouped analysis will be skipped")
        metadata = {}

    logger.info(
        f"Loaded prices: {len(prices)} rows, {prices['ticker'].nunique()} tickers; "
        f"returns: {len(returns)} rows"
    )
    return prices, returns, metadata


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
    block_dir = out_dir / "01_price_trends"
    block_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for ticker, grp in prices.groupby("ticker"):
        grp = grp.sort_values("date").copy()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(grp["date"], grp["close"], label="Close", linewidth=1.2, color="steelblue")
        for window in rolling_windows:
            rolled = grp["close"].rolling(window).mean()
            ax.plot(grp["date"], rolled, label=f"{window}-day MA", linewidth=1.0, linestyle="--")
        ax.set_title(f"{ticker} — Price Trend")
        ax.set_xlabel("Date")
        ax.set_ylabel("Adjusted Close Price")
        ax.legend()
        fig.tight_layout()
        path = block_dir / f"{ticker}_price.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        logger.info(f"Saved price trend chart: {path.name}")

    return saved


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
    block_dir = out_dir / "01_price_trends"
    block_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for ticker, grp in prices.groupby("ticker"):
        grp = grp.sort_values("date").copy()
        vol = grp["volume"].astype(float)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(grp["date"], vol, width=1.0, color="steelblue", alpha=0.7)
        ax.set_title(f"{ticker} — Daily Volume")
        ax.set_xlabel("Date")
        ax.set_ylabel("Volume")
        fig.tight_layout()
        path = block_dir / f"{ticker}_volume.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        logger.info(f"Saved volume chart: {path.name}")

    return saved


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
    block_dir = out_dir / "02_return_distributions"
    block_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    stats_rows: list[dict] = []

    for ticker, grp in returns.groupby("ticker"):
        grp = grp.sort_values("date")
        sr = grp["simple_return"].dropna()

        # Histogram + KDE
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(sr, bins=50, density=True, alpha=0.5, color="steelblue", label="Returns")
        try:
            kde_x = np.linspace(sr.min(), sr.max(), 300)
            kde = stats.gaussian_kde(sr)
            ax.plot(kde_x, kde(kde_x), color="darkorange", linewidth=1.5, label="KDE")
        except Exception as exc:
            logger.debug(f"KDE skipped for {ticker}: {exc}")
        ax.set_title(f"{ticker} — Return Distribution")
        ax.set_xlabel("Simple Return")
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()
        hist_path = block_dir / f"{ticker}_histogram.png"
        fig.savefig(hist_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(hist_path)
        logger.info(f"Saved histogram: {hist_path.name}")

        # Q-Q plot (skip if insufficient data)
        if len(sr) >= config.EDA_MIN_QQ_ROWS:
            fig, ax = plt.subplots(figsize=(6, 6))
            (osm, osr), (slope, intercept, _) = stats.probplot(sr, dist="norm", fit=True)
            ax.scatter(osm, osr, s=10, alpha=0.5, color="steelblue")
            ax.plot(
                osm,
                slope * np.array(osm) + intercept,
                color="red",
                linewidth=1.2,
            )
            ax.set_title(f"{ticker} — Q-Q Plot")
            ax.set_xlabel("Theoretical Quantiles")
            ax.set_ylabel("Sample Quantiles")
            fig.tight_layout()
            qq_path = block_dir / f"{ticker}_qqplot.png"
            fig.savefig(qq_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved.append(qq_path)
            logger.info(f"Saved Q-Q plot: {qq_path.name}")
        else:
            logger.warning(
                f"Skipping Q-Q plot for {ticker}: {len(sr)} rows < {config.EDA_MIN_QQ_ROWS} minimum"
            )

        # Per-ticker statistics
        jb_pvalue: float = float("nan")
        if len(sr) >= 8:
            try:
                _, jb_pvalue = stats.jarque_bera(sr)
            except Exception as exc:
                logger.debug(f"Jarque-Bera skipped for {ticker}: {exc}")

        flat_lined = bool((sr == 0.0).all())
        if flat_lined:
            logger.warning(f"Ticker {ticker} has all-zero returns — flagged as flat-lined")

        stats_rows.append({
            "ticker": str(ticker),
            "mean": float(sr.mean()),
            "std": float(sr.std()),
            "skew": float(sr.skew()),
            "kurtosis": float(sr.kurtosis()),
            "min": float(sr.min()),
            "max": float(sr.max()),
            "jarque_bera_pvalue": float(jb_pvalue),
            "flat_lined": flat_lined,
        })

    # Combined boxplot across all tickers
    tickers = sorted(returns["ticker"].unique().tolist())
    if tickers:
        wide = returns.pivot_table(
            index="date", columns="ticker", values="simple_return", aggfunc="mean"
        )
        wide = wide[sorted(wide.columns)]
        fig, ax = plt.subplots(figsize=(max(8, len(tickers) * 1.5), 6))
        wide.boxplot(ax=ax, column=sorted(wide.columns))
        ax.set_title("Return Distribution — All Tickers")
        ax.set_xlabel("Ticker")
        ax.set_ylabel("Simple Return")
        fig.tight_layout()
        box_path = block_dir / "all_tickers_boxplot.png"
        fig.savefig(box_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(box_path)
        logger.info(f"Saved combined boxplot: {box_path.name}")

    stats_df = pd.DataFrame(stats_rows)
    return saved, stats_df


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
      - One heatmap of monthly volatility (month x ticker)
    Saves to out_dir/03_volatility/.
    Returns (saved_paths, monthly_vol_df) where monthly_vol_df is pivoted month x ticker.
    """
    block_dir = out_dir / "03_volatility"
    block_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    annualization = math.sqrt(trading_days_per_year)

    monthly_vol_pieces: list[pd.DataFrame] = []

    for ticker, grp in returns.groupby("ticker"):
        grp = grp.sort_values("date").copy()
        sr = grp["simple_return"]

        if len(sr) < window:
            logger.warning(
                f"Skipping rolling volatility for {ticker}: "
                f"{len(sr)} rows < window size {window}"
            )
            continue

        rolling_vol = sr.rolling(window).std() * annualization

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(grp["date"], rolling_vol, color="darkorange", linewidth=1.2)
        ax.set_title(f"{ticker} — Rolling {window}-Day Annualized Volatility")
        ax.set_xlabel("Date")
        ax.set_ylabel("Annualized Volatility")
        fig.tight_layout()
        path = block_dir / f"{ticker}_rolling_vol.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        logger.info(f"Saved rolling vol chart: {path.name}")

        # Monthly vol for heatmap
        grp["year_month"] = grp["date"].dt.to_period("M").astype(str)
        monthly = (
            grp.groupby("year_month")["simple_return"]
            .std()
            .mul(annualization)
            .reset_index()
        )
        monthly.columns = ["year_month", "volatility"]
        monthly["ticker"] = str(ticker)
        monthly_vol_pieces.append(monthly)

    if not monthly_vol_pieces:
        return saved, pd.DataFrame()

    all_monthly = pd.concat(monthly_vol_pieces, ignore_index=True)
    monthly_vol_pivot = all_monthly.pivot(
        index="year_month", columns="ticker", values="volatility"
    )
    monthly_vol_pivot = monthly_vol_pivot.sort_index()
    monthly_vol_pivot.columns = sorted([str(c) for c in monthly_vol_pivot.columns])

    n_months = len(monthly_vol_pivot)
    n_tickers = len(monthly_vol_pivot.columns)
    fig, ax = plt.subplots(figsize=(max(8, n_tickers * 1.4), max(6, n_months * 0.35)))
    sns.heatmap(
        monthly_vol_pivot,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.3,
        cbar_kws={"label": "Annualized Volatility"},
    )
    ax.set_title("Monthly Volatility Heatmap")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Month")
    fig.tight_layout()
    heatmap_path = block_dir / "monthly_vol_heatmap.png"
    fig.savefig(heatmap_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(heatmap_path)
    logger.info(f"Saved monthly volatility heatmap: {heatmap_path.name}")

    return saved, monthly_vol_pivot


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
    block_dir = out_dir / "04_correlations"
    block_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    tickers = sorted(returns["ticker"].unique().tolist())
    if len(tickers) < 2:
        logger.warning("Fewer than 2 tickers available — skipping correlation analysis")
        return [], pd.DataFrame(), []

    # Long-to-wide pivot is transient: only inside this function, never persisted
    wide = returns.pivot_table(
        index="date", columns="ticker", values="simple_return", aggfunc="mean"
    )
    wide = wide[tickers]  # deterministic column order
    corr_matrix = wide.corr()

    # Correlation matrix heatmap
    n = len(tickers)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n)))
    sns.heatmap(
        corr_matrix,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        annot_kws={"size": max(6, 10 - n)},
    )
    ax.set_title("Return Correlation Matrix")
    fig.tight_layout()
    corr_path = block_dir / "correlation_matrix.png"
    fig.savefig(corr_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(corr_path)
    logger.info(f"Saved correlation matrix: {corr_path.name}")

    # All unique pairs sorted by |correlation| descending
    all_pairs: list[tuple[str, str, float]] = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            a, b = tickers[i], tickers[j]
            c = corr_matrix.loc[a, b]
            if not math.isnan(c):
                all_pairs.append((a, b, float(c)))
    all_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    if top_n_pairs > len(all_pairs):
        logger.warning(
            f"EDA_TOP_N_CORRELATIONS={top_n_pairs} > available pairs={len(all_pairs)}; "
            f"using all {len(all_pairs)} pairs"
        )
    top_pairs = all_pairs[:top_n_pairs]

    for ticker_a, ticker_b, corr_val in top_pairs:
        aligned = pd.DataFrame({"x": wide[ticker_a], "y": wide[ticker_b]}).dropna()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(aligned["x"], aligned["y"], alpha=0.4, s=10, color="steelblue")
        ax.set_title(f"{ticker_a} vs {ticker_b}  (r = {corr_val:.2f})")
        ax.set_xlabel(ticker_a)
        ax.set_ylabel(ticker_b)
        fig.tight_layout()
        scatter_path = block_dir / f"top_pair_{ticker_a}_{ticker_b}_scatter.png"
        fig.savefig(scatter_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(scatter_path)
        logger.info(f"Saved pairwise scatter: {scatter_path.name}")

    # Sector-grouped correlation heatmap
    if metadata:
        sector_map: dict[str, str] = {
            t: ((metadata.get(t) or {}).get("sector") or "Unknown")
            for t in tickers
        }
        unique_sectors = sorted(set(sector_map.values()))
        if len(unique_sectors) == 1:
            logger.info(
                f"All tickers in sector '{unique_sectors[0]}' — "
                f"sector heatmap collapses to a single block"
            )

        sector_corr_data: dict[str, dict[str, float]] = {}
        for s1 in unique_sectors:
            sector_corr_data[s1] = {}
            for s2 in unique_sectors:
                t1_list = [t for t in tickers if sector_map[t] == s1]
                t2_list = [t for t in tickers if sector_map[t] == s2]
                vals = [
                    float(corr_matrix.loc[t1, t2])
                    for t1 in t1_list
                    for t2 in t2_list
                    if t1 != t2 and not math.isnan(corr_matrix.loc[t1, t2])
                ]
                sector_corr_data[s1][s2] = float(np.mean(vals)) if vals else float("nan")

        sector_pivot = pd.DataFrame(sector_corr_data).T
        sector_pivot = sector_pivot[sorted(sector_pivot.columns)]

        ns = len(unique_sectors)
        fig, ax = plt.subplots(figsize=(max(5, ns * 1.5), max(4, ns * 1.2)))
        sns.heatmap(
            sector_pivot,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            annot_kws={"size": 9},
        )
        ax.set_title("Sector Average Correlation")
        fig.tight_layout()
        sector_path = block_dir / "sector_correlation.png"
        fig.savefig(sector_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(sector_path)
        logger.info(f"Saved sector correlation heatmap: {sector_path.name}")
    else:
        logger.warning("No metadata available — skipping sector-grouped correlation analysis")

    return saved, corr_matrix, top_pairs


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
        "zero_volume_days": [],           # populated by run_eda (needs prices)
        "zero_price_change_days": [],     # populated by run_eda (needs prices)
      }
    """
    cleaning_outliers: dict[str, int] = {}
    if cleaning_report:
        cleaning_outliers = cleaning_report.get("actions", {}).get("outliers_flagged", {})

    top_moves_per_ticker: dict[str, list[dict]] = {}
    eda_vs_cleaning_match: dict[str, dict] = {}

    for ticker, grp in returns.groupby("ticker"):
        grp = grp.sort_values("date")
        n = min(top_n, len(grp))
        top_idx = grp["simple_return"].abs().nlargest(n).index
        top_rows = grp.loc[top_idx].sort_values("simple_return", ascending=False, key=abs)

        moves = (
            top_rows.assign(date=top_rows["date"].dt.strftime("%Y-%m-%d"))
            [["date", "simple_return", "log_return"]]
            .astype({"simple_return": float, "log_return": float})
            .to_dict("records")
        )
        top_moves_per_ticker[str(ticker)] = moves

        eda_count = len(moves)
        cleaning_count = int(cleaning_outliers.get(str(ticker), 0))
        overlap = sum(
            1 for m in moves
            if abs(m["simple_return"]) > config.OUTLIER_RETURN_THRESHOLD
        )
        eda_vs_cleaning_match[str(ticker)] = {
            "eda_count": eda_count,
            "cleaning_count": cleaning_count,
            "overlap": overlap,
        }

    return {
        "top_moves_per_ticker": top_moves_per_ticker,
        "eda_vs_cleaning_match": eda_vs_cleaning_match,
        "zero_volume_days": [],
        "zero_price_change_days": [],
    }


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
    Returns path to saved JSON.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    dist_stats_dict: dict = {}
    if not distribution_stats.empty:
        for row in distribution_stats.to_dict("records"):
            t = str(row["ticker"])
            jb = row.get("jarque_bera_pvalue", float("nan"))
            entry: dict = {
                "mean": round(float(row["mean"]), 6),
                "std": round(float(row["std"]), 6),
                "skew": round(float(row["skew"]), 6),
                "kurtosis": round(float(row["kurtosis"]), 6),
                "min": round(float(row["min"]), 6),
                "max": round(float(row["max"]), 6),
                "jarque_bera_pvalue": round(float(jb), 6)
                if not math.isnan(float(jb))
                else None,
            }
            if row.get("flat_lined"):
                entry["flat_lined"] = True
            dist_stats_dict[t] = entry

    monthly_vol_dict: dict = {}
    if not monthly_vol.empty:
        for month_idx in monthly_vol.index:
            row = monthly_vol.loc[month_idx]
            monthly_vol_dict[str(month_idx)] = {
                str(col): (round(float(v), 4) if not math.isnan(float(v)) else None)
                for col, v in row.items()
            }

    corr_matrix_dict: dict = {}
    if not corr_matrix.empty:
        for ticker_a in sorted(corr_matrix.index):
            corr_matrix_dict[str(ticker_a)] = {
                str(ticker_b): round(float(corr_matrix.loc[ticker_a, ticker_b]), 6)
                for ticker_b in sorted(corr_matrix.columns)
            }

    top_pairs_list = [
        {"ticker_a": a, "ticker_b": b, "correlation": round(c, 6)}
        for a, b, c in top_pairs
    ]

    by_block: dict[str, int] = {
        "01_price_trends": 0,
        "02_return_distributions": 0,
        "03_volatility": 0,
        "04_correlations": 0,
        "05_outliers": 0,
    }
    for p in saved_plot_paths:
        for block in by_block:
            if block in p.parts:
                by_block[block] += 1
                break

    correlations_section: dict = {
        "matrix": corr_matrix_dict,
        "top_pairs": top_pairs_list,
    }
    if sector_avg_correlation is not None:
        correlations_section["sector_avg_correlation"] = sector_avg_correlation

    summary: dict = {
        "run_timestamp": run_timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "distribution_stats": dist_stats_dict,
        "monthly_volatility": monthly_vol_dict,
        "correlations": correlations_section,
        "outliers": outlier_report,
        "plots_generated": {
            "total": len(saved_plot_paths),
            "by_block": by_block,
        },
        "duration_sec": round(duration_sec, 2),
    }

    if input_data is not None:
        summary["input"] = input_data

    out_path = out_dir / "eda_summary.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    logger.info(f"Saved EDA summary: {out_path}")
    return out_path


def run_eda() -> dict:
    """
    Orchestrator. Called by main.py.
    Pipeline:
      load_processed_data -> validate schemas
      -> plot_price_trends -> plot_volume_trends -> plot_return_distributions
      -> plot_volatility -> plot_correlations -> detect_outliers
      -> save_eda_summary

    Returns a summary dict:
      {
        "plots_generated": int,
        "tickers_analyzed": int,
        "anomalies_flagged": int,
        "summary_path": str,
        "duration_sec": float
      }
    """
    t0 = datetime.now(timezone.utc)

    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = config.LOG_DIR / f"eda_{t0.strftime('%Y-%m-%d')}.log"
    log_id = logger.add(log_path, rotation="1 day", level="INFO", enqueue=True)

    try:
        try:
            plt.style.use(config.EDA_PLOT_STYLE)
        except OSError:
            logger.warning(
                f"Matplotlib style '{config.EDA_PLOT_STYLE}' not found — using default"
            )

        # ── Load ──────────────────────────────────────────────────────────────
        logger.info("EDA Block 0: Loading and validating processed data")
        prices, returns, metadata = load_processed_data(
            config.PROCESSED_DATA_DIR,
            config.RAW_DATA_DIR,
        )

        input_data = {
            "prices_rows": len(prices),
            "returns_rows": len(returns),
            "tickers": int(returns["ticker"].nunique()),
            "date_range": [
                returns["date"].min().strftime("%Y-%m-%d"),
                returns["date"].max().strftime("%Y-%m-%d"),
            ],
        }

        all_paths: list[Path] = []

        # ── Block 1: Price & volume trends ────────────────────────────────────
        logger.info("EDA Block 1: Price trends")
        all_paths.extend(
            plot_price_trends(
                prices, config.EDA_ROLLING_WINDOWS, config.PLOTS_DIR, config.EDA_PLOT_DPI
            )
        )
        logger.info("EDA Block 1: Volume trends")
        all_paths.extend(
            plot_volume_trends(prices, config.PLOTS_DIR, config.EDA_PLOT_DPI)
        )

        # ── Block 2: Return distributions ─────────────────────────────────────
        logger.info("EDA Block 2: Return distributions")
        dist_paths, dist_stats = plot_return_distributions(
            returns, config.PLOTS_DIR, config.EDA_PLOT_DPI
        )
        all_paths.extend(dist_paths)

        # ── Block 3: Volatility ───────────────────────────────────────────────
        logger.info("EDA Block 3: Volatility")
        vol_paths, monthly_vol = plot_volatility(
            returns,
            config.EDA_VOL_WINDOW,
            config.EDA_TRADING_DAYS_PER_YEAR,
            config.PLOTS_DIR,
            config.EDA_PLOT_DPI,
        )
        all_paths.extend(vol_paths)

        # ── Block 4: Correlations ─────────────────────────────────────────────
        logger.info("EDA Block 4: Correlations")
        corr_paths, corr_matrix, top_pairs = plot_correlations(
            returns,
            metadata,
            config.EDA_TOP_N_CORRELATIONS,
            config.PLOTS_DIR,
            config.EDA_PLOT_DPI,
        )
        all_paths.extend(corr_paths)

        # Sector-average correlation for the summary JSON
        sector_avg_correlation: dict[str, float] | None = None
        if metadata and not corr_matrix.empty:
            tickers_present = sorted(corr_matrix.columns.tolist())
            sector_map = {
                t: ((metadata.get(t) or {}).get("sector") or "Unknown")
                for t in tickers_present
            }
            sector_avg_correlation = {}
            for sector in sorted(set(sector_map.values())):
                sector_tickers = [t for t in tickers_present if sector_map[t] == sector]
                vals = [
                    float(corr_matrix.loc[t1, t2])
                    for i, t1 in enumerate(sector_tickers)
                    for t2 in sector_tickers[i + 1:]
                    if not math.isnan(corr_matrix.loc[t1, t2])
                ]
                sector_avg_correlation[sector] = round(float(np.mean(vals)), 4) if vals else 0.0

        # ── Block 5: Outlier detection ────────────────────────────────────────
        logger.info("EDA Block 5: Outlier detection")
        cleaning_report: dict = {}
        cleaning_report_path = config.PROCESSED_DATA_DIR / "cleaning_report.json"
        if cleaning_report_path.exists():
            with open(cleaning_report_path, "r", encoding="utf-8") as fh:
                cleaning_report = json.load(fh)
        else:
            logger.warning(
                "cleaning_report.json not found — EDA-vs-cleaning cross-check skipped"
            )

        outlier_report = detect_outliers(returns, cleaning_report, config.EDA_TOP_N_MOVES)

        # Enrich with zero-volume and zero-price-change from prices
        zero_volume_days: list[dict] = []
        zero_price_change_days: list[dict] = []
        for ticker, grp in prices.groupby("ticker"):
            grp = grp.sort_values("date").copy()
            zero_vol_dates = grp.loc[grp["volume"].fillna(0) == 0, "date"]
            zero_volume_days.extend(
                {"date": d.strftime("%Y-%m-%d"), "ticker": str(ticker)}
                for d in zero_vol_dates
            )
            grp["_is_flat"] = grp["close"].diff().abs().fillna(1.0) == 0
            grp["_run_id"] = (~grp["_is_flat"]).cumsum()
            flat_grp = grp[grp["_is_flat"]].copy()
            if not flat_grp.empty:
                flat_grp["consecutive_days"] = (
                    flat_grp.groupby("_run_id")["_is_flat"].transform("sum").astype(int)
                )
                zero_price_change_days.extend(
                    flat_grp.assign(
                        date=flat_grp["date"].dt.strftime("%Y-%m-%d"),
                        ticker=str(ticker),
                    )[["date", "ticker", "consecutive_days"]].to_dict("records")
                )

        outlier_report["zero_volume_days"] = zero_volume_days
        outlier_report["zero_price_change_days"] = zero_price_change_days

        # ── Block 6: Save summary ─────────────────────────────────────────────
        logger.info("EDA Block 6: Saving summary")
        duration_pre_save = (datetime.now(timezone.utc) - t0).total_seconds()
        summary_path = save_eda_summary(
            dist_stats,
            monthly_vol,
            corr_matrix,
            top_pairs,
            outlier_report,
            all_paths,
            config.REPORTS_DIR,
            input_data=input_data,
            sector_avg_correlation=sector_avg_correlation,
            run_timestamp=t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
            duration_sec=duration_pre_save,
        )

        total_anomalies = sum(
            len(v) for v in outlier_report["top_moves_per_ticker"].values()
        )
        result = {
            "plots_generated": len(all_paths),
            "tickers_analyzed": int(returns["ticker"].nunique()),
            "anomalies_flagged": total_anomalies,
            "summary_path": str(summary_path),
            "duration_sec": round((datetime.now(timezone.utc) - t0).total_seconds(), 2),
        }

        logger.info(f"EDA complete: {result}")
        return result

    finally:
        logger.remove(log_id)
