# System Architecture — SPY Expiry Signal Pipeline

**Project**: DAMO-699-3 MDA Capstone — University of Niagara Falls Canada
**Date**: 2026-04-24
**Submission readiness update**: 2026-05-04
**Authors**: Xin, Osmond (Yi) et al.

> **Current status note (2026-05-04):** Sections that describe missing implementation
> work reflect the repository state at the original 2026-04-24 architecture review.
> The submission package has since been completed: portfolio simulation,
> enrichment joins, composite scoring, Monte Carlo validation, walk-forward
> validation, H₂ testing, ML validation, final report, and editable presentation
> deck are now present. The current submission checklist is summarized in
> Section 10.1.

---

## 1. Overview

The project is structured as a five-layer analytical pipeline. Raw market data enters at Layer 1 and flows through feature engineering, strategy modelling, evaluation, and finally surfaces in the notebooks (Layer 5) as interactive analysis and visualisation.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 1 · DATA          modules/data/                              │
│  CacheManager · DataLoader · DataValidator · ExpiryCalculator       │
└─────────────────────────────┬───────────────────────────────────────┘
                              │  Dict[ticker → OHLCV DataFrame]
┌─────────────────────────────▼───────────────────────────────────────┐
│  Layer 2 · FEATURES       modules/features/                         │
│  BaseFeature · TechnicalIndicators · FeaturePipeline                │
└─────────────────────────────┬───────────────────────────────────────┘
                              │  DataFrame + RSI, Consecutive_Count,
                              │  SMA, BB, MACD, Volume_Ratio columns
┌─────────────────────────────▼───────────────────────────────────────┐
│  Layer 3 · MODELS         modules/models/                           │
│  BaseStrategy · RSIReversalStrategy · ConsecutiveCandleStrategy     │
│  MADistanceStrategy · VolumeMACD_ComboStrategy · ModelRegistry      │
└─────────────────────────────┬───────────────────────────────────────┘
                              │  predict(df, expiry_date) → 1 | 0
┌─────────────────────────────▼───────────────────────────────────────┐
│  Layer 4 · EVALUATION     modules/evaluation/                       │
│  ExpiryScanner · PerformanceCalculator · PatternAnalyzer · metrics  │
└─────────────────────────────┬───────────────────────────────────────┘
                              │  watchlist DataFrame,
                              │  Sharpe, Win Rate, Profit Factor, MDD
                              │  Composite Score, Monte Carlo, Walk-forward
┌─────────────────────────────▼───────────────────────────────────────┐
│  Layer 5 · APPLICATION    notebooks/                                │
│  spy_expiry_analysis_v2 · pattern_distribution_scanner             │
│  active_list_scanner · take_profit_research                        │
└─────────────────────────────────────────────────────────────────────┘
```

Supporting concerns that span all layers:

| Concern | Location |
|---------|----------|
| Parameter presets | `modules/config/` |
| Optional pre-filters (ADX, volatility) | `modules/analysis/` |
| Offline data tools | `tools/` |
| Unit tests | `tests/unit/` |
| Reference literature | `papers/` |

---

## 2. Layer 1 — Data

**Directory**: [modules/data/](modules/data/)

### 2.1 CacheManager (`cache_manager.py`)

Responsible for acquiring and maintaining the local OHLCV cache.

- **Full download mode**: Fetches all ~500 S&P 500 tickers from Stooq.com (free, no key). Polite rate-limiting at 1 s per request. Resumable — skips already-downloaded tickers.
- **Incremental mode**: Uses the Alpaca Markets REST API (free tier) to append the most recent daily bars to each existing CSV. Batch size 50, delay 0.5 s per request to stay within the 200 req/min limit.
- **Auto-detection**: `update_cache()` inspects the cache directory; if CSVs are missing it triggers a full download, otherwise it runs incremental.

Cache layout:

```
cache/
├── constituent_data/<TICKER>.csv   # One CSV per stock, columns: Date, Symbol, Open, High, Low, Close, Volume
├── sp500_list.csv                  # Current ~500 S&P 500 tickers
├── vix.csv                         # FRED VIXCLS 1990–2026 (9,468 rows)
├── fomc_dates.csv                  # Federal Reserve FOMC meeting dates (97 rows, 2015–2026)
├── earnings_dates.csv              # Nasdaq earnings calendar (21,326 events)
└── expiry_dates.csv                # 3rd-Friday monthly dates (derived by ExpiryCalculator)
```

### 2.2 DataLoader (`data_loader.py`)

Reads the cached CSVs into a `Dict[str, pd.DataFrame]` keyed by ticker. Handles date parsing and index alignment.

### 2.3 DataValidator (`data_validator.py`)

Checks each ticker DataFrame for:
- Gaps in the date sequence (missing trading days)
- Duplicate dates
- Non-monotonic price sequences

Flags problems to the caller; does not silently drop rows.

### 2.4 ExpiryCalculator (`expiry_calculator.py`)

Computes the 3rd Friday of each month for a given year range via `generate_expiry_dates(start, end) → List[pd.Timestamp]`. Returns dates in memory only — it does **not** write `cache/expiry_dates.csv`. That file exists in the cache as a pre-computed artefact but is not produced by this module; notebooks call `generate_expiry_dates()` directly rather than reading the CSV.

---

## 3. Layer 2 — Features

**Directory**: [modules/features/](modules/features/)

### 3.1 BaseFeature (`base.py`)

Abstract interface that every feature calculator must implement:

```python
calculate(df: pd.DataFrame) -> pd.DataFrame   # add columns, return extended df
get_feature_names() -> List[str]              # declare which columns it adds
```

### 3.2 TechnicalIndicators (`technical_indicators.py`)

The core feature calculator used by all capstone notebooks and strategies. Adds the following columns to any OHLCV DataFrame:

| Column | Definition |
|--------|-----------|
| `RSI` | 14-period Wilder RSI |
| `MACD`, `MACD_Signal`, `MACD_Hist` | 12/26/9 EMA-based MACD |
| `SMA_9`, `SMA_20`, `SMA_50` | Simple moving averages |
| `MA_Distance_9/20/50` | `(Close - SMA) / SMA × 100` — % deviation |
| `BB_Upper`, `BB_Middle`, `BB_Lower` | Bollinger Bands (20-period, 2σ) |
| `BB_Position` | `(Close - BB_Lower) / (BB_Upper - BB_Lower)` — normalised position within bands |
| `Volume_SMA`, `Volume_Ratio` | 20-period volume MA; `Volume / Volume_SMA` |
| `Consecutive_Count` | Length of current close-to-close streak (up or down), capped at 5 |
| `Consecutive_Direction` | Direction of streak: `+1` (green) or `−1` (red) |

> **Implementation note on `Consecutive_Count`**: direction is determined by close-to-close comparison (`today < yesterday → red`) rather than open-to-close. Free data sources (Stooq) sometimes have adjusted open prices that diverge from the actual traded open; close-to-close is more reliable. The streak counter uses a single vectorised numpy pass, approximately 500× faster than the earlier row-loop implementation.

### 3.3 FeaturePipeline (`feature_pipeline.py`)

Chains any sequence of `BaseFeature` instances:

```python
pipeline = FeaturePipeline([TechnicalIndicators(), HighPointIndicators()])
df_enriched = pipeline.transform(df)
```

Each calculator is called in order; if one raises an exception a warning is printed and the pipeline continues with the next calculator. This allows optional calculators (e.g., `HighPointIndicators`) to be included safely.

### 3.4 Enrichment Features

These columns are required by the capstone hypotheses and are implemented in
`modules/features/enrichment_features.py`:

| Column | Source | Status |
|--------|--------|--------|
| `VIX_Level` | `cache/vix.csv` | Complete |
| `VIX_Regime` | Derived (`Low < 15`, `Medium 15–25`, `High > 25`) | Complete |
| `FOMC_Proximity` | `cache/fomc_dates.csv` | Complete |
| `Earnings_Proximity` | `cache/earnings_dates.csv` | Complete |

---

## 4. Layer 3 — Models

**Directory**: [modules/models/](modules/models/)

### 4.1 BaseStrategy (`base.py`)

Abstract interface for all rule-based strategies:

```python
predict(df: pd.DataFrame, expiry_date: pd.Timestamp) -> int
# Returns:  1 = long,  0 = neutral,  −1 = short (unused in capstone)
```

Includes helpers `_get_value_at_date()` and `validate_data()` shared by all concrete strategies.

### 4.2 Concrete Strategies (`pattern_models.py`)

| Class | Signal logic | Primary use |
|-------|-------------|-------------|
| `RSIReversalStrategy` | `RSI < threshold AND Consecutive_Count ≥ min AND direction == −1` | **Primary capstone strategy** (H₁) |
| `ConsecutiveCandleStrategy` | `Consecutive_Count ≥ 4`, with optional MA filter | Alternative rule for comparison |
| `MADistanceStrategy` | Price more than X% below SMA, optionally filtered by RSI | Mean-reversion baseline |
| `VolumeMACD_ComboStrategy` | Volume ratio > 2× average AND MACD histogram sign | Trend-following alternative |

`RSIReversalStrategy` is the pre-specified rule tested against the null hypothesis. Its **class default** is `rsi_threshold = 30`, `min_consecutive = 3`, `lookback_days = 1`. The **v4.1 optimised values** are `rsi_threshold = 22`, `min_consecutive = 3`; `hold_days = 6` is a **portfolio simulation parameter** (not part of the strategy config) and belongs in the backtest loop. All three values are currently set only at the notebook call site, not persisted as a named preset in `modules/config/` (see §8 gap note).

### 4.3 ModelRegistry (`model_registry.py`)

Maps snake_case name strings to strategy classes via a factory pattern. Notebooks instantiate strategies by name using `ModelRegistry.create()`, not `.get()`:

```python
strategy = ModelRegistry.create('rsi_reversal', config={'rsi_threshold': 22})
```

Registered names: `'rsi_reversal'`, `'consecutive_candle'`, `'ma_distance'`, `'volume_macd_combo'`, `'ema_signal'`, plus six weekly-strategy variants (`'weekly_rsi_reversal'`, etc.).

---

## 5. Layer 4 — Evaluation

**Directory**: [modules/evaluation/](modules/evaluation/)

### 5.1 ExpiryScanner (`expiry_scanner.py`)

Iterates over a `Dict[ticker → DataFrame]` and applies a `PredictivePatterns` matcher to each stock's latest data. Returns a ranked watchlist DataFrame. Acts as the forward-looking signal generator for each upcoming expiry date.

### 5.2 PerformanceCalculator (`performance_calculator.py`)

Computes per-stock returns for a given expiry date (or a list of dates). Supports two return modes:

- **intraday**: `(Close − Open) / Open` on the expiry day
- **close_to_close**: `(Close[t] − Close[t−1]) / Close[t−1]`

Handles timezone-naive vs. UTC index mismatches via a normalised `searchsorted` lookup, with a ±5-day fallback for non-trading-day expiry dates.

### 5.3 PatternAnalyzer (`pattern_analyzer.py`)

Extracts technical indicator snapshots (RSI, `Consecutive_Count`, MA distances, MACD, BB_Position, Volume_Ratio) for the trading day **before** each expiry date, scoped to a given set of performers (top or bottom). Produces a flat DataFrame used for descriptive comparison — e.g., "did top performers tend to have lower RSI the day before expiry?"

> **Note on scope**: `PatternAnalyzer` is a feature-extraction helper, not a signal evaluator. It does not compute "hit rates" or "win rates" — those are produced by `metrics.summarize_patterns()` once a signal column and a return column are both present in the same DataFrame.

### 5.4 Metrics (`metrics.py`)

Standalone statistical functions used throughout notebooks:

| Function | Purpose |
|----------|---------|
| `calculate_correlation_matrix()` | Pearson/Spearman/Kendall correlation matrix of features |
| `statistical_significance_test()` | t-test, Mann-Whitney U, or KS test between two return groups |
| `summarize_patterns()` | Aggregated win rate and return stats per pattern |
| `calculate_sharpe_ratio()` | Annualised Sharpe (daily returns, 252 periods) |
| `calculate_max_drawdown()` | Peak-to-trough drawdown with start/end indices |
| `calculate_composite_score()` | Pre-registered weighted capstone score |
| `calculate_information_ratio()` | Active return vs. benchmark |

Monte Carlo, walk-forward, and parameter-sensitivity validation are implemented
in `modules/evaluation/robust_validation.py`.

---

## 6. Layer 5 — Notebooks

See Section 7 for detailed per-notebook analysis.

---

## 7. Notebook Deep-Dive

### 7.1 `spy_expiry_analysis_v2.ipynb` — Main Research Notebook

**Role in the project**: This is the primary capstone deliverable notebook. It implements the end-to-end analytical workflow described in the project proposal (Sections VII and VIII) and is where the three hypotheses (H₁, H₂, H₃) will be evaluated and visualised.

**Current structure (39 cells, executed with no saved notebook errors):**

| Section | Content | Status |
|---------|---------|--------|
| §1 Configuration | Target expiry date and parameter setup | Complete |
| §2 Imports | Module and library imports | Complete |
| §3 Data Layer | `CacheManager`, `DataLoader`, expiry calendar | Complete |
| §4 Feature Engineering | `FeaturePipeline` → `TechnicalIndicators` | Complete |
| §5 Expiry Day Performance and Portfolio Simulation | Descriptive expiry returns plus `run_simulation()` IS/OOS backtest | Complete |
| §6 Pattern Analysis | `PatternAnalyzer.analyze_all_performers()` — extracts indicator snapshots for top/bottom performers | Complete as descriptive analysis; does not apply `RSIReversalStrategy.predict()` |
| §7 Multi-Expiry Analysis | Historical expiry descriptive analysis plus simulation outputs | Complete |
| §8 Statistical Analysis | Monte Carlo, walk-forward, H₂ z-test, sensitivity analysis | Complete |
| §9 Predictive Pattern Discovery | Historical pattern analysis | Complete as supporting analysis |
| §10 Visualisations | Six required capstone charts | Complete |
| §11 Next Expiry Prediction | Forward-looking watchlist for next expiry | Complete |
| §12 Summary | Print summary stats | Complete |

The previous key missing pieces have been implemented: T+1 portfolio simulation,
VIX/FOMC/Earnings enrichment, Monte Carlo validation, walk-forward heatmap,
equity curve, RSI/consecutive distributions, and VIX-regime win-rate chart.

**Relationship to project**: This notebook is the equivalent of the capstone final analysis report in executable form. Every visualisation listed in Section VIII of the proposal (`project_purpose.md`) must appear here.

---

### 7.2 `pattern_distribution_scanner.ipynb` — Signal Distribution Scanner

**Role in the project**: Exploratory data notebook. Its purpose is to characterise how often the core signal conditions (RSI oversold, consecutive red candles) occur across the S&P 500 universe at any given scan date, and to show the distribution of each pattern independently.

**Current structure (16 cells)**:

| Section | Content |
|---------|---------|
| §1 Configuration | Scan date |
| §2–3 Imports + Data | Load cached OHLCV, run `TechnicalIndicators` |
| §4 Feature Calculation | Apply `FeaturePipeline` |
| §5 Pattern Detection | Custom `detect_patterns()` function checks RSI/consecutive thresholds |
| §6 Visualisation | Bar charts of how many stocks satisfy each pattern condition |
| §7 Detailed Results | Per-stock tables for each matched pattern |

**Relationship to project**: Useful for verifying that signal conditions are neither trivially common nor pathologically rare at any given point in time. However, this notebook operates on a **single scan date** (one `SCAN_DATE` variable), so it cannot produce the two proposal-required visualisations on its own:

- *RSI distribution at signal dates* — must come from the portfolio simulation `trades_df`, showing RSI values at all ~2,000–3,300 historical trigger points across all 132 expiry dates.
- *Consecutive candle count distribution* — same source; requires the full backtest output.

The pattern_distribution_scanner is a development and debugging aid, not the primary source of these charts. The actual distributions will be derived from `trades_df` once the portfolio simulation is built.

---

### 7.3 `active_list_scanner.ipynb` — Multi-Condition Stock Screener

**Role in the project**: Utility notebook for generating a shortlist of stocks that satisfy multiple pattern conditions simultaneously. It is the practical, forward-looking screening tool used to identify candidates before each monthly expiry.

**Current structure (16 cells)**:

| Section | Content |
|---------|---------|
| §1–2 Setup | Imports, configuration |
| §3–4 Data Load | `DataLoader`, feature pipeline |
| §5 Run Scan | `ActiveListScanner.scan()` — builds historical active list and identifies dual/triple-pattern candidates |
| §6 Active List Details | Frequency distribution of how often each ticker appears across historical scans |
| §7 List 1 — Dual-Pattern | Stocks matching BOTH `Oversold_Reversal` AND `Deep_Pullback` simultaneously |
| §8 List 2 — Triple Candidates | Dual-pattern stocks that also appear in the high-frequency historical list |
| Summary | Count tables and final ranked output |

**Relationship to project**: Provides a frequency-based pre-screening layer — stocks that repeatedly appear in historical top/bottom performer lists are candidates worth monitoring. `ActiveListScanner` sorts by `total_appearances`, not by RSI. The proposal's "top-3 per expiry ranked by lowest RSI" selection rule belongs in the portfolio simulation loop, not here. This notebook is an operational aid, not a substitute for the RSI-ranked signal selection.

**Note on capstone scope**: This notebook is primarily an operational tool (generating the watchlist for each upcoming expiry), not a backtesting or hypothesis-testing tool. Its results are inputs to `spy_expiry_analysis_v2.ipynb` rather than standalone evidence. For the capstone, the signal-selection logic here should be described in the Methods section and its output used to define the portfolio selection step.

---

### 7.4 `take_profit_research.ipynb` — Exit Strategy Research

**Role in the project**: Research notebook investigating whether different exit mechanisms (take-profit levels, trailing stops, RSI-based exits) improve risk-adjusted returns beyond a fixed hold period. This addresses the exit side of the portfolio simulation described in Section 7.4 of the proposal.

**Current structure (23 cells)**:

| Section | Content |
|---------|---------|
| §1 Load Data + Features | `TechnicalIndicators` + `HighPointIndicators` |
| §2 Generate Trades | `EMAScanner` to find historical buy signals; build trade DataFrame |
| §3 Strategy Comparison | 6 exit strategies benchmarked: fixed hold, fixed %, trailing stop, RSI exit, partial take-profit, resistance-based exit |
| §4 Visualisation | Box plots of return distribution per strategy |
| §5 Parameter Sensitivity | Grid search over take-profit %, trailing stop %, RSI exit thresholds |
| §6 HEIGHT_SCORE Analysis | Does the `HEIGHT_SCORE` indicator (proximity to resistance) predict max gain? |
| Summary | Best strategy and parameter summary |

**Relationship to project**: Directly informs the stop-loss / take-profit parameter choice in the portfolio simulation. Section 7.4 of the proposal states: "Positions are exited after a fixed hold period or on stop-loss / take-profit triggers determined via parameter search." This notebook is that parameter search.

**Resolved consistency note**: The notebook has been re-run with
`RSIReversalStrategy` entries for the capstone calibration. The remaining
`HEIGHT_SCORE` / resistance-based exit material is an extension beyond the core
proposal scope.

---

## 8. Supporting Modules

### `modules/config/`
Frozen parameter presets for various strategy variants. Existing files:
`capstone_v4_params.py`, `ema_params.py`, `ema5_short_params.py`,
`weekly_params.py`, `meta_params.py`, `breakout_params.py`,
`divergence_params.py`, `high_point_params.py`, `mtf_params.py`, and
`active_list_params.py`.

`capstone_v4_params.py` is the capstone-specific single source of truth for
RSI threshold, minimum consecutive candles, hold period, commission, position
caps, composite weights, random seed, and walk-forward window definitions.

### `modules/analysis/`
Optional pre-filters applied before strategy scanning:
- `adx_prescreen.py`: Filters out trending stocks (high ADX) where mean reversion is less reliable.
- `volatility_filter.py`, `weekly_volatility_filter.py`: Remove stocks with abnormally high volatility that may distort RSI signals.

These filters are labelled "optional" in the proposal (Section 7.6) and are not part of the core backtested strategy; they are sensitivity-analysis tools.

### `modules/exploration/`
Four standalone exploratory scripts used during hypothesis development — **not referenced by any notebook** and not part of the production pipeline:
- `explore_deep_oversold.py`: Tests a composite signal combining `Deep_Pullback` + `Oversold_Reversal` + `Volume_Surge_Reversal`.
- `explore_composite_patterns.py`: Generalised composite pattern framework (`BasePattern` + `CompositePattern` dataclasses).
- `explore_extreme_overbought.py`: Mirror analysis for overbought conditions.
- `explore_momentum_reversal.py`: Momentum-reversal composite signal study.

These scripts are research artifacts from earlier iterations; their findings informed the final strategy design but they do not need to appear in the capstone submission.

### Undocumented `modules/evaluation/` sub-modules
Several evaluation modules are used by notebooks but omitted from §5:

| Module | Used by | Purpose |
|--------|---------|---------|
| `take_profit_backtester.py` | `take_profit_research.ipynb` | Runs 6 exit strategies (fixed hold, fixed %, trailing stop, RSI exit, partial TP, resistance) against a set of trades; computes Sharpe, Profit Factor per strategy |
| `ema_scanner.py` | `take_profit_research.ipynb` (§2) | Scans stock data for EMA bounce/crossover entry signals; generates the trade list fed into `take_profit_backtester` |
| `active_list_scanner.py` | `active_list_scanner.ipynb` | Tracks which tickers appear repeatedly in historical scans (frequency-based active list); exposes `ActiveListScanner` class |
| `ema_backtester.py` | Not used by any notebook or tool | Backtests `EMASignalStrategy` over historical dates — legacy/standalone script |
| `meta_scanner.py` | Not used by any notebook or tool | Aggregates signals from 5 individual scanners into a weighted composite watchlist — legacy/standalone script |
| `weekly_scanner.py` | Not used by any notebook or tool | Scans weekly-timeframe data for `WeeklyCompositeStrategy` — no weekly notebooks exist |
| `weekly_predictive_patterns.py` | Not used by any notebook or tool | Weekly-timeframe pattern definitions — no weekly notebooks exist |

### `tools/`

| Script | Purpose |
|--------|---------|
| `download_enrichment_data.py` | Fetches VIX (FRED), FOMC calendar, and earnings dates into `cache/` |
| `update_sp500.py` | Refreshes `cache/sp500_list.csv` from current S&P 500 membership |
| `updatesp500fromStooq.py` | Alternative updater pulling from Stooq |
| `return_calculator.py` | Standalone T+N return calculator for ad-hoc analysis |
| `return_calculator_stoploss.py` | Return calculator with stop-loss logic |
| `generate_watchlist.py` | CLI tool: runs `ExpiryScanner.scan()` then applies `VolatilityFilter`; outputs only stocks matching ≥ 2 patterns |

### `tests/unit/`
Three unit test modules:
- `test_data_validator.py` — gap detection, duplicate detection
- `test_expiry_calculator.py` — 3rd-Friday date arithmetic across years
- `test_technical_indicators.py` — RSI, MACD, Bollinger, consecutive candle correctness

---

## 9. Data Flow Summary

```
Stooq.com / Alpaca API
        │
        ▼ CacheManager.update_cache()
cache/constituent_data/*.csv
        │
        ▼ DataLoader.load_sp500_batch() / load_ticker()
Dict[ticker → OHLCV DataFrame]
        │
        ▼ FeaturePipeline.transform()     (+ VIX/FOMC/Earnings join — PENDING)
Dict[ticker → Feature DataFrame]
   (RSI, Consecutive_Count, SMA, BB, MACD, Volume_Ratio, ...)
        │
        ├── PATH A: Forward-looking watchlist (§11, EXISTS)
        │       ExpiryScanner.scan() → ranked watchlist for next expiry
        │
        ├── PATH B: Descriptive analysis (§5/§7, EXISTS)
        │       PerformanceCalculator.calculate_returns()
        │       → intraday returns on expiry day (Open→Close, NOT proposal backtest)
        │
        └── PATH C: Portfolio simulation (⚠ NOT IMPLEMENTED)
                RSIReversalStrategy.predict() → signals per (ticker, expiry_date)
                → top-3 selection by RSI → T+1 open entry → N-day hold
                → 0.2% commission deduction → trades_df
                        │
                        ▼
                metrics.py: Sharpe, Win Rate, Profit Factor, Max Drawdown
                ⚠ calculate_composite_score() — NOT IMPLEMENTED
                        │
                        ▼
                IS/OOS split       ⚠ NOT IMPLEMENTED
                Walk-forward (8w)  ⚠ NOT IMPLEMENTED
                Monte Carlo (500)  ⚠ NOT IMPLEMENTED
                        │
                        ▼
                Notebooks: charts, hypothesis test outputs
```

---

## 10. Remaining Work — Todo List

The following tasks are derived from the project milestone table (Section X of `project_purpose.md`), the gap analysis (`doc/202604/16/gap_analysis.md`), and the current state of the codebase as of 2026-04-24.

### 10.1 Submission Readiness Snapshot — 2026-05-04

The 2026-04-24 gap list has been closed for the capstone submission package.

| Area | Current artefact | Status |
|---|---|---|
| Frozen v4.1 parameters | `modules/config/capstone_v4_params.py` | Complete |
| Enrichment joins | `modules/features/enrichment_features.py` | Complete |
| Portfolio simulation | `modules/evaluation/portfolio_simulator.py` and `spy_expiry_analysis_v2.ipynb` | Complete |
| Composite score | `modules/evaluation/metrics.calculate_composite_score()` | Complete |
| IS/OOS split | `spy_expiry_analysis_v2.ipynb`, summarized in `final_report.md` | Complete |
| Monte Carlo and walk-forward validation | `modules/evaluation/robust_validation.py` | Complete |
| H₂ VIX z-test | `spy_expiry_analysis_v2.ipynb`, summarized in `final_report.md` | Complete |
| Parameter sensitivity | `modules/evaluation/robust_validation.py`, summarized in `final_report.md` | Complete |
| ML validation and H₃ | `notebooks/ml_validation.ipynb` | Complete |
| Survivorship-bias proxy universe | `cache/sp500_constituents_2015.csv` | Complete |
| Final report | `doc/capstone/final_report.md` | Complete |
| Editable presentation deck | `doc/capstone/spy_expiry_capstone_presentation.pptx` | Complete |
| Test suite | `python -m pytest tests/ -q` | Passing: 69 tests, 4 subtests |

### Completed P0/P1 Submission Work

- [x] Portfolio simulation in `modules/evaluation/portfolio_simulator.py`
- [x] Composite score in `modules/evaluation/metrics.py`
- [x] ML dependencies pinned in `requirements.txt`
- [x] VIX/FOMC/Earnings enrichment in `modules/features/enrichment_features.py`
- [x] IS/OOS holdout split in `spy_expiry_analysis_v2.ipynb`
- [x] Monte Carlo permutation test in `modules/evaluation/robust_validation.py`
- [x] Walk-forward 8-window analysis in `modules/evaluation/robust_validation.py`
- [x] Six core visualisations in `spy_expiry_analysis_v2.ipynb`
- [x] Final report in `doc/capstone/final_report.md`
- [x] Formal H₂ two-proportion z-test in `spy_expiry_analysis_v2.ipynb`
- [x] ML feature matrix and H₃ models in `notebooks/ml_validation.ipynb`
- [x] ML walk-forward validation and benchmark comparison table
- [x] Method justification and literature review in the final report
- [x] Frozen capstone parameter file, `modules/config/capstone_v4_params.py`
- [x] RSIReversalStrategy take-profit recalibration in `take_profit_research.ipynb`
- [x] Parameter sensitivity analysis
- [x] Survivorship-bias sensitivity check using `cache/sp500_constituents_2015.csv`
- [x] Editable presentation deck in `doc/capstone/spy_expiry_capstone_presentation.pptx`

---

### P2 — Extensions (bonus differentiation)

- [ ] **Market regime segmentation (optional robustness check)**
  Classify each expiry date as Bull / Sideways / Bear by SPY 200-day SMA. Report Profit Factor per regime. Criterion: positive Profit Factor in ≥ 2 of 3 regimes. Listed as Optional in Section 7.6.

- [ ] **Earnings filter impact analysis (optional robustness check)**
  Compare win rate and Profit Factor with and without the `Earnings_Proximity` filter applied. Quantifies how much fundamental noise contaminates the technical signal.

- [ ] **Power BI dashboard** (optional, mentioned in rubric)
  Interactive replica of the core visualisations (equity curve, regime breakdown, parameter sensitivity). High effort; only pursue if the written deliverables are complete.

- [ ] **SQLite data layer** (optional, mentioned in rubric)
  Migrate `cache/*.csv` files into a SQLite database managed via `modules/data/`. Adds a SQL / relational database component to the project's demonstrated skill set.

---

### Milestone Status Summary

| # | Milestone | Target | Status |
|---|-----------|--------|--------|
| 1 | Data Infrastructure | Mar 2026 | ✅ Complete |
| 2 | Feature Engineering | Apr 2026 | ✅ Complete |
| 3 | Rule-Based Model & Backtesting | May 2026 | ✅ Complete |
| 4 | Statistical Validation (Monte Carlo, walk-forward) | Jun 2026 | ✅ Complete |
| 5 | VIX Regime Analysis (H₂) | Jul 2026 | ✅ Complete |
| 6 | ML Validation (H₃) | Jul–Aug 2026 | ✅ Complete |
| 7 | Final Report & Defence | Aug–Sep 2026 | ✅ Complete for submission package |
