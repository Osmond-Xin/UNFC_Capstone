# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

Master of Data Analytics capstone project. Investigates whether RSI oversold + consecutive red candle patterns can predict S&P 500 stock returns on SPY monthly options expiry days (3rd Friday). Research pipeline: data collection → feature engineering → strategy backtesting → statistical validation.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Data — full download (~10 min, no API key) or incremental via Alpaca
python -c "from modules.data import CacheManager; CacheManager().update_cache()"
python tools/update_sp500.py
python tools/download_enrichment_data.py   # VIX, FOMC, earnings calendars

# Tests
python -m pytest tests/
python -m pytest tests/unit/test_technical_indicators.py -v
```

## Architecture

5-layer pipeline: **Data → Features → Models → Evaluation → Application (notebooks)**

### Layer responsibilities

- **`modules/data/`** — `CacheManager` fetches from Stooq (full) or Alpaca (incremental). `DataLoader` reads cached CSVs into DataFrames. `DataValidator` checks for gaps, duplicate dates, non-monotonic prices. `ExpiryCalculator` computes the 3rd-Friday monthly expiry calendar.

- **`modules/features/`** — `BaseFeature` defines the `calculate(df) → df` interface. `TechnicalIndicators` implements RSI, SMA, Bollinger Bands, MACD, Volume Ratio, and `Consecutive_Count` (days where Close < Open). `FeaturePipeline` chains multiple `BaseFeature` instances: `pipeline.transform(df)` runs them in order, left-joining new columns.

- **`modules/models/`** — `BaseStrategy` defines `predict(df, expiry_date) → int` (1=long, 0=neutral). Concrete strategies in `pattern_models.py`: `RSIReversalStrategy`, `ConsecutiveCandleStrategy`, `MADistanceStrategy`, `VolumeMACD_ComboStrategy`. `model_registry.py` maps name strings to classes for notebook use.

- **`modules/evaluation/`** — `ExpiryScanner` runs a strategy across all expiry dates. `PerformanceCalculator` computes returns for (ticker, date) pairs with T+N hold logic. `metrics.py` has standalone metric functions. `PatternAnalyzer` does pattern distribution analysis.

- **`modules/config/`** — Frozen parameter presets per strategy variant (e.g., `ema_params.py`, `weekly_params.py`). Reference presets for notebook use.

- **`modules/analysis/`** — ADX pre-screen and volatility filters applied before scanning. Optional pre-filters, not part of the core backtested strategy.

### Data contracts
- Raw cache: `cache/constituent_data/<TICKER>.csv` — columns `Date, Symbol, Open, High, Low, Close, Volume`
- Enrichment cache: `cache/vix.csv`, `cache/fomc_dates.csv`, `cache/earnings_dates.csv`
- Feature columns use uppercase: `RSI`, `SMA_20`, `BB_Upper`, `Consecutive_Count`
- Strategy `predict()` returns `1` (long) or `0` (neutral); `-1` (short) is defined in base but unused

## Mandatory Constraints

### Documentation Storage Rules
All documentation MUST be stored in the `doc/` folder with date-based paths:
- Path format: `doc/YYYYMM/DD/` (year-month as folder, day as subfolder)
- Example: For 2026-04-16, use `doc/202604/16/`
- Design documents go in `doc/design/`
- Capstone requirements live in `doc/capstone/`

### Current Work Tracking
The `CURRENT_WORK.md` file in the project root tracks work in progress:
- **MUST update** at the start of any significant task
- **MUST clear** after task completion
- Purpose: enable other sessions or context restarts to resume quickly
