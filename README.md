# SPY Options Expiry Day Pattern Analysis

**Master of Data Analytics — Capstone Project**

A modular Python framework for investigating whether technical patterns in S&P 500 stocks can predict individual stock performance on SPY monthly options expiry days (3rd Friday of each month). The project applies the full analytics lifecycle — data collection, feature engineering, strategy backtesting, statistical validation, and communication of results.

---

## Research Question

> Can technical patterns (RSI oversold conditions + consecutive bearish candles) observed in the days before SPY monthly options expiry dates reliably predict individual stock returns on expiry day?

---

## Analytical Approach

| Stage | Method |
|-------|--------|
| Data Collection | S&P 500 daily OHLCV via Stooq (free, full history) + Alpaca (incremental updates) |
| Feature Engineering | RSI, SMA, consecutive candle count, MA-distance (vectorized pandas) |
| Strategy Modelling | Rule-based signal: RSI < threshold AND ≥ N consecutive red candles |
| Backtesting | Realistic portfolio simulation (T+1 entry, 3 positions/day max, equal-weight sizing, 0.2% commission) |
| Model Evaluation | In-Sample / Out-of-Sample split, composite score (Sharpe + Win Rate + Profit Factor + MaxDD) |
| Robustness Testing | Walk-forward analysis, Monte Carlo permutation test, parameter sensitivity analysis |

---

## Key Results (v4.1, validated)

| Metric | Value |
|--------|-------|
| Strategy | RSI ≤ 20, Hold 9d, SMA-18, SL 4%, TP 12%, Max-Consec-Red 8 |
| OOS Composite Score | **0.518** |
| Sharpe Ratio | **0.970** |
| Profit Factor | **1.367** |
| Max Drawdown | **−3.9%** |
| Monte Carlo p-value | **0.004** (STRONG signal, p < 0.01) |
| Walk-Forward (8 windows, 2015–2026) | OOS mean 0.532, 3/8 windows stable |
| OOS Gap | **−0.012** (negative gap = OOS outperforms IS, zero overfitting) |

---

## Project Status

### Completed
- [x] **Data infrastructure** — S&P 500 OHLCV (~500 tickers, 2015–2026), VIX, FOMC calendar, and earnings dates acquired, validated, and cached
- [x] **Feature engineering** — Full feature matrix built for every (stock, expiry_date) pair: RSI, Consecutive_Count, MA_Distance, BB_Position, MACD_Hist, Volume_Ratio, VIX_Level/Regime, FOMC_Proximity, Earnings_Proximity
- [x] **Rule-based strategy** — `RSIReversalStrategy` implemented; 130+ parameter experiments across 5 phases (v1 → v4.1); optimal thresholds identified
- [x] **Portfolio backtesting** — Realistic simulation (T+1, 0.2% commission, max 3/day, max 15 concurrent); IS/OOS composite 0.518, OOS gap −0.012
- [x] **Monte Carlo permutation test** — 500 shuffles; p = 0.004 (H₀ rejected at α = 0.01)
- [x] **Walk-forward validation** — 8 rolling windows (2015–2026); OOS mean composite 0.532
- [x] **Parameter sensitivity analysis** — ±10% / ±20% perturbation of each parameter; RSI threshold identified as highest-sensitivity parameter
- [x] **Market regime baseline** — Bull / Sideways / Bear segmentation; positive Profit Factor confirmed in all three regimes

### Pending
- [ ] **VIX regime formal test (H₂)** — Two-proportion z-test comparing win rate in High VIX (> 25) vs. Low VIX (< 15) subsamples
- [ ] **Machine learning validation (H₃)** — Logistic Regression, Random Forest, XGBoost trained on full feature matrix with walk-forward time splits
- [ ] **SHAP analysis** — Per-feature importance (XGBoost); confirm RSI and Consecutive_Count rank in top-3
- [ ] **ML vs. rule-based comparison** — Portfolio Sharpe, Win Rate, Profit Factor: ML-derived signals vs. v4.1 rule baseline
- [ ] **Survivorship bias sensitivity** — Rerun backtest on frozen 2015 S&P 500 universe subset
- [ ] **Final visualization set** — Equity curve vs. SPY buy-and-hold, Monte Carlo histogram, walk-forward heatmap, VIX regime win-rate bar chart, SHAP summary plot
- [ ] **Written capstone report** — Full write-up with findings, limitations, and conclusions
- [ ] **Oral defence preparation**

---

## Project Structure

```
Capstone/
├── modules/                     # Core library (5-layer pipeline)
│   ├── data/                    # CacheManager, DataLoader, DataValidator, ExpiryCalculator
│   ├── features/                # TechnicalIndicators, FeaturePipeline
│   ├── models/                  # RSIReversalStrategy and variants
│   ├── evaluation/              # PerformanceCalculator, ExpiryScanner, metrics
│   ├── config/                  # Frozen strategy parameter presets
│   └── analysis/                # ADX pre-screen, volatility filters
├── notebooks/                   # Jupyter analysis notebooks
│   ├── spy_expiry_analysis_v2.ipynb      # Main monthly expiry analysis
│   └── spy_weekly_expiry_analysis.ipynb  # Weekly expiry extension
├── tools/                       # Standalone utility scripts
│   ├── update_sp500.py          # Incremental cache update
│   └── generate_watchlist.py    # Generate actionable watchlists
├── tests/                       # Unit tests
├── cache/
│   ├── sp500_list.csv           # S&P 500 universe (~500 tickers)
│   ├── expiry_dates.csv         # Pre-computed monthly expiry dates
│   └── constituent_data/        # Per-ticker OHLCV CSVs (gitignored)
├── doc/capstone/purpose.md      # Capstone project requirements
└── requirements.txt
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Add Alpaca API credentials to .env (only needed for incremental updates)
```

---

## Data

```bash
# Full download from Stooq.com — no API key required (~500 tickers, ~10 minutes)
python -c "from modules.data import CacheManager; CacheManager().update_cache()"

# Incremental update via Alpaca API (requires .env)
python tools/update_sp500.py
```

Cache files are stored in `cache/constituent_data/<TICKER>.csv` with columns `Date, Symbol, Open, High, Low, Close, Volume`.

---

## Running the Analysis

```bash
# Interactive notebook analysis
jupyter notebook notebooks/spy_expiry_analysis_v2.ipynb
```

---

## Strategy Logic

The core strategy identifies **oversold mean-reversion** opportunities before each SPY expiry date:

1. **Entry signal** (both conditions required — AND logic):
   - `RSI(14) < 20` — deeply oversold
   - `3 ≤ consecutive red candles ≤ 8` — selling pressure but not free-falling

2. **Portfolio constraints**: T+1 open execution, max 3 new entries per day ranked by lowest RSI, max 15 concurrent positions, equal-weight sizing, no leverage.

3. **Exit rules**: hold 9 trading days, stop loss at −4%, take profit at +12%.

---

## Validation Methodology

| Test | Description | Result |
|------|-------------|--------|
| IS/OOS Split | Train 2015–2025, test 2025 H2–2026 | OOS gap −0.012 |
| Walk-Forward | 8 rolling 3yr-IS / 1yr-OOS windows | OOS mean 0.532 |
| Monte Carlo | 500 random signal permutations | p = 0.004 |
| Sensitivity | ±10/20% perturbation of each parameter | RSI most sensitive (20%) |
| Market Regime | Bull / Sideways / Bear segmentation | Positive PF in all regimes |

---

## Unit Tests

```bash
python -m pytest tests/
python -m pytest tests/unit/test_technical_indicators.py
```
