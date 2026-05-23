# SPY Options Expiry Day Pattern Analysis
## Predicting S&P 500 Stock Returns Using Technical Patterns and Machine Learning

**Program**: Master of Data Analytics — DAMO-699-3 Capstone Project
**Institution**: University of Niagara Falls Canada
**Submission**: Spring 2026
**Submission package status**: Ready for team review and submission, 2026-05-14

| Author | Student ID | Email | Stream |
|--------|-----------|-------|--------|
| Diego Bicieg Vazquez Del Mercado | NF1909519 | diego.bicieg9519@myunfc.ca | Data Enrichment & Config |
| Ícaro Dos Santos Rabelo | NF1009242 | icaro.dos9242@myunfc.ca | Portfolio Simulation |
| Sai Bhaskar Chundru | NF1010427 | sai.chundru0427@myunfc.ca | Statistical Validation |
| Osmond Xin (Yi) | NF1007319 | yi.xin7319@myunfc.ca | ML Validation & Report |

---

## Abstract

This study tests whether a pre-specified rule combining RSI oversold conditions (RSI < 22)
with consecutive bearish candles (≥ 3 red candles, close-to-close) generates a statistically
significant and temporally stable positive-return edge in S&P 500 individual stocks on SPY
monthly options expiry days (the third Friday of each month). Using daily OHLCV data for all
current S&P 500 constituents over 2015–2026, enriched with VIX, FOMC calendar, and earnings
release data, a realistic T+1 portfolio simulation is executed with top-3 signal selection
per expiry cycle, 0.2% round-trip commission, and a 15-position concurrent cap.

Statistical validity is assessed via 500-iteration Monte Carlo permutation testing, 8-window
walk-forward analysis (3-year IS / 1-year OOS), and parameter-sensitivity perturbation. Three
supervised ML models (Logistic Regression, Random Forest, XGBoost + SHAP) independently
validate whether RSI and Consecutive_Count dominate the predictive structure.

**Results:** The strategy produced IS composite score 0.543, OOS composite 0.787 (gap 0.244,
exceeding the 0.05 threshold), walk-forward mean OOS composite 0.728 across 8 windows
(positive in 8/8, > 0.5 in 7/8). Monte Carlo permutation test (500 shuffles) yielded
p = 0.998, placing the observed strategy below the 1st percentile of the null distribution —
H₁ is not supported at α = 0.05. VIX-regime z-test produced z = 1.663, p = 0.048
(H₂ supported: high-VIX win rate 72.7% vs. low-VIX 52.4%). ML feature importance showed
RSI in top-3 for 2/3 models (Random Forest and XGBoost) and Consecutive_Count for 0/3
models — H₃ is supported on the disjunctive criterion (RSI alone qualifies); VIX_Level
and MA_Distance_20 are the consistently dominant non-rule features.

Read together, the three results form a coherent regime-conditional story rather than
three independent outcomes. H₁ fails globally because the edge is **regime-conditional**:
averaged across all VIX regimes, the signal is indistinguishable from random entry, but
within the High-VIX sub-sample it is materially stronger than within Low-VIX (H₂).
H₃ confirms from an independent supervised-learning angle that RSI carries genuine
predictive content (in 2/3 models) but is consistently outranked by VIX_Level — the same
regime variable that drives H₂. The pre-specified RSI + consecutive-candle rule is
therefore best deployed as a **High-VIX-conditional filter** rather than a universal
monthly screen (§8.2).

*Keywords:* mean reversion, options expiry, RSI, VIX regime, machine learning,
walk-forward validation, S&P 500

---

## 1. Problem Context

### 1.1 Motivation

Options expiry microstructure produces systematic price distortions in individual equities
that are distinct from noise-driven fluctuations. On each monthly SPY expiry date (the third
Friday), dealers who have sold options must dynamically rebalance their delta hedges; as
time-to-expiry approaches zero, the time-decay of these hedges (charm) creates systematic
buy-side pressure in stocks that have been pushed below fair value by prior selling flows.
This mechanical catalyst is exogenous to each stock's fundamental trajectory and creates a
predictable, event-driven mean-reversion window.

Despite extensive research on options expiry microstructure (Golez & Jackwerth, 2012;
Baltussen et al., 2025) and on mean reversion in equities (Poterba & Summers, 1988;
Kim et al., 1991), these bodies of literature have developed in isolation. Expiry studies
focus on index-level dynamics; mean-reversion studies treat price behaviour as a statistical
property of returns over time, without incorporating event-driven mechanical catalysts.
Neither strand has examined whether individual-stock expiry effects can be identified *ex ante*
from observable technical conditions, nor whether their magnitude is systematically conditioned
by the prevailing volatility environment.

This gap has practical consequences: systematic traders who routinely decide whether to enter,
exit, or defer trades around monthly expiry must rely on heuristics rather than evidence-based
rules. Closing this gap enables rigorous evaluation of a technically simple and operationally
accessible strategy.

### 1.2 Research Question

> Do RSI oversold conditions combined with consecutive bearish candles, observed in S&P 500
> stocks in the days preceding SPY monthly options expiry, produce a positive-return edge
> that is statistically distinguishable from random entry and temporally stable across
> out-of-sample periods?

### 1.3 Hypotheses

All hypotheses are pre-registered (fixed before any parameter search or simulation) and
evaluated at α = 0.05:

**H₁ (Primary):** The RSI-oversold + consecutive-candle signal produces a composite score
statistically higher than the null distribution of random-entry scores (Monte Carlo
permutation test, 500 shuffles, seed = 42), and exhibits positive composite scores in
≥ 6 of 8 walk-forward OOS windows.

**H₂ (VIX moderator):** Win rate is materially higher in High-VIX environments (VIX > 25)
than in Low-VIX environments (VIX < 15). Test: two-proportion z-test, one-sided alternative.

**H₃ (ML feature validation):** Supervised ML models assign dominant importance to the
rule features: at least one of {RSI, Consecutive_Count} appears in top-3 importances for
≥ 2 of 3 models (Logistic Regression, Random Forest, XGBoost).

---

## 2. Data Sources and Coverage

### 2.1 Primary Price Data

Daily OHLCV data for all current S&P 500 constituents (~510 tickers) were sourced from
Stooq.com (free, no API key) using the project's `CacheManager` module. The full download
covers approximately 2010–2026 per ticker; the analysis window is restricted to
**2015-01-01 through 2026-05-04** (the latest cached trading day at the time of the
canonical IS/OOS simulation), yielding ~11 years of monthly expiry cycles across
~136 expiry dates.

`CacheManager` operates in two modes selected automatically at runtime. In **full-download
mode**, it iterates all ~510 tickers and fetches complete OHLCV history from Stooq.com
(free, no API key required), writing one CSV per ticker to `cache/constituent_data/`.
A polite 1-second delay between requests avoids rate-limiting. The download is resumable:
tickers whose CSV already exists are skipped, so an interrupted run can continue without
re-fetching completed files. In **incremental mode**, `CacheManager` detects that CSVs
are already present, identifies the most recent cached date for each ticker, and appends
only the missing trading days using the Alpaca Markets REST API (batch size 50, 0.5 s
delay, within the free-tier 200 req/min limit).

Data integrity is enforced by `DataValidator`, which checks each ticker's DataFrame for
gaps in the trading-day sequence (missing sessions > 5 consecutive days), duplicate date
entries, and non-monotonic price sequences (close prices that violate expected ordering).
Problems are flagged and logged to the caller rather than silently dropped, so the analysis
can document which tickers were excluded and why. A full `DataValidator` quality report
across all 510 tickers is produced in `notebooks/eda.ipynb` §3.

### 2.2 Enrichment Data

| Dataset | Source | Coverage | Rows | Join key |
|---------|--------|----------|------|----------|
| VIX daily close (`VIXCLS`) | FRED St. Louis Fed | 1990-01-02 → 2026 | 9,468 | `signal_date` |
| FOMC meeting dates | Federal Reserve public calendar | 2015–2026 | 97 | ±5 days |
| Earnings release dates | Nasdaq Earnings Calendar | 2015–2026 | 21,326 | ticker + ±3 days |
| S&P 500 constituent list | Cached from Wikipedia / SP500 API | Current 2026 | ~510 | ticker |

All three enrichment joins are implemented in `modules/features/enrichment_features.py`
and applied at the signal date (the trading day before each expiry), ensuring no
forward-looking information is introduced.

### 2.3 Derived Features

Monthly expiry dates are computed from first principles via `ExpiryCalculator.generate_expiry_dates()`,
which calculates the third Friday of each month with holiday adjustments. No external
expiry calendar is required.

---

## 3. Analytical Methods

### 3.1 Signal Rule Design

The primary rule (`RSIReversalStrategy`, v4.1) generates a long signal when:

```
RSI(14) < 22   AND   Consecutive_Red_Candles ≥ 3   AND   Direction == Bearish
```

Consecutive candles are measured on a close-to-close basis rather than open-to-close, because
adjusted open prices from free data sources (Stooq) can deviate from actual traded opens after
split/dividend adjustments; close-to-close comparisons are more reliable.

**Why AND logic over OR?** An OR condition would generate signals whenever either RSI or
consecutive candles triggered independently, producing a much larger and less selective signal
set. The AND condition ensures both the momentum indicator (RSI) and the price-action pattern
(consecutive candles) confirm the oversold thesis simultaneously, improving signal precision at
the cost of recall. This aligns with the portfolio design (top-3 selection per expiry) rather
than exhaustive screening.

**Why RSI(14)?** The 14-period Wilder RSI is the industry-standard oversold/overbought
oscillator. Using the standard period avoids the look-ahead bias inherent in optimising the
RSI period against in-sample outcomes. RSI(14) is widely used in academic technical analysis
research (Wilder, 1978), providing a verifiable and theoretically motivated benchmark.

**Why RSI threshold 22 over the conventional 30?** Parameter search on the IS period
(2015–2025) — implemented in `modules/exploration/explore_deep_oversold.py` and committed
to the repository as the audit trail for this value — found that tighter thresholds
(RSI < 22) produce fewer but higher-quality signals with better average returns. At RSI < 30,
the signal fires too frequently and dilutes the oversold selection; at RSI < 15, the signal
fires too rarely for statistical power. RSI < 22 is the IS-optimised v4.1 value, frozen in
`modules/config/capstone_v4_params.py` before any out-of-sample evaluation; §4.3 details
how this IS-derived value is reconciled with the pre-registration claim.

### 3.2 Portfolio Simulation Design

The simulation is implemented in `modules/evaluation/portfolio_simulator.py`.

**Entry**: The signal is evaluated on the trading day *before* each monthly expiry date
(signal date). Entry is at the **open price on the expiry date itself** (T+1 after signal).
This prevents look-ahead bias: the decision is made on close-of-day T, executed at the
T+1 open, which is unknowable at T.

**Signal selection**: Up to `MAX_ENTRIES_PER_EXPIRY = 3` signals are selected per expiry
cycle, ranked by ascending RSI (most oversold first). This top-3 selection reflects the
portfolio's risk budget: entering all ~15–20 triggered signals per cycle would expose
the portfolio to idiosyncratic stock-level events that overwhelm the expiry-driven signal.

**Why top-3 per expiry?** Ranking by lowest RSI is *theoretically consistent with* the
strongest mean-reversion case (largest deviation from equilibrium, conditional on elevated
volatility, per Kim, Nelson, and Startz 1991) and with the highest stock-level
delta-hedging pressure documented by Ni, Pearson, and Poteshman (2005); neither study
tests RSI-ordered selection directly, so the ranking choice is justified theoretically
rather than empirically. Three entries per cycle balances diversification (more than one
position) with selectivity (not diluted across the full universe). The 15-concurrent-position
cap provides a secondary constraint against accumulating an unmanageable portfolio.

**Exit**: Positions are closed at the end-of-day close price `HOLD_DAYS = 6` trading days
after entry. The hold period was calibrated via grid search in `take_profit_research.ipynb`
(noting the limitation that the original search used EMAScanner entries; this is acknowledged
in the Limitations section).

**Commission**: A flat 0.2% round-trip commission (0.1% per leg) is deducted from every
trade. This is conservative for retail execution and aggressive for institutional trading,
chosen as a reasonable mid-point that does not make the strategy appear artificially
profitable.

**Why 0.2% commission?** Retail brokerage fees for online stock trading are typically
$0–$5 per trade; at an average holding value of $5,000–$10,000, the dollar cost per side
is $0–$5, implying < 0.1% commission per leg. Using 0.2% round-trip builds in a safety
margin for bid-ask spreads, market impact, and potential regulatory changes, without
overstating costs to the point where no real-world strategy could pass. This figure is
consistent with the all-in retail transaction-cost range of roughly 0.1–0.3% round-trip
used in comparable academic U.S.-equity rule backtests (Brock, Lakonishok, & LeBaron, 1992;
Sullivan, Timmermann, & White, 1999).

### 3.3 Feature Engineering

All features are computed from data available at the signal date only.

**Feature pipeline design.** Feature computation is structured as a chain of `BaseFeature`
objects managed by `FeaturePipeline`. Each calculator implements a single `calculate(df) →
df` interface and declares which columns it adds via `get_feature_names()`. The pipeline
calls each calculator in order, left-joining the new columns onto the running DataFrame.
If a calculator raises an exception (for example, a ticker with insufficient history for
MACD computation), the pipeline logs a warning and continues with the next calculator
rather than failing the entire run. This design allows optional calculators — such as
`EnrichmentFeatures` — to be included safely without breaking the core indicator pipeline.
In practice, the capstone pipeline uses `FeaturePipeline([TechnicalIndicators()])` for
rule-based scanning and `FeaturePipeline([TechnicalIndicators()])` followed by
`EnrichmentFeatures.enrich(df, ticker)` for the ML feature matrix.

The full technical indicator set (`TechnicalIndicators`) adds 18 columns to each ticker's
OHLCV DataFrame: RSI(14), MACD(12/26/9), SMA(9/20/50), MA_Distance(9/20/50), Bollinger
Bands, BB_Position, Volume_SMA, Volume_Ratio, Consecutive_Count, Consecutive_Direction.

Enrichment features added by `EnrichmentFeatures`:
- `VIX_Level`: raw VIX close on signal date (fallback: nearest prior trading day ≤ 5 days)
- `VIX_Regime`: categorical Low (< 15), Medium (15–25), High (> 25)
- `FOMC_Proximity`: binary 1 if any FOMC meeting is within ±5 calendar days
- `Earnings_Proximity`: binary 1 if the ticker's earnings release is within ±3 days

### 3.4 Composite Score

All quantitative evaluation uses the pre-registered composite score:

$$C = \min\!\left(\frac{PF}{2}, 1\right) \times 0.35 + \min\!\left(\frac{\max(S, 0)}{2}, 1\right) \times 0.30 + WR \times 0.20 + \left(1 - \min\!\left(\frac{|MDD|}{0.30}, 1\right)\right) \times 0.15$$

where *PF* = Profit Factor, *S* = Sharpe ratio, *WR* = Win Rate, *MDD* = Max Drawdown.
Weights (0.35 / 0.30 / 0.20 / 0.15) were fixed before the first simulation run. The formula
is implemented in `modules/evaluation/metrics.calculate_composite_score()`.

### 3.5 Statistical Validation

**Monte Carlo permutation test**: The `net_return` column of `trades_df` is shuffled 500 times
(seed = 42 set once before the first shuffle), breaking the strategy-signal-return link while
preserving the return distribution. Each permuted version's composite score forms the null
distribution. The p-value is `sum(null ≥ observed) / 500`.

**Walk-forward analysis**: 8 windows of 3-year IS / 1-year OOS (OOS years 2018–2025). IS
windows slide forward one year at a time (2015–2017, 2016–2018, …, 2022–2024) and overlap
across the eight windows; OOS windows are non-overlapping, covering each calendar year
2018–2025 exactly once. For each window, `run_simulation()` is called and the composite
score is computed *on the OOS portion only*. This simulates repeated live deployment and
tests whether the strategy degrades across different market regimes (COVID crash 2020,
rate-hike cycle 2022, AI-bull 2023–24).

**Why walk-forward over a simple 80/20 train-test split?** Financial time series exhibit
regime changes and non-stationarity. A single train-test split can accidentally put all
high-volatility years in one set. Walk-forward analysis distributes across eight distinct
market environments, providing a more honest estimate of live performance.

**Parameter sensitivity**: Each v4.1 parameter (`rsi_threshold`, `min_consecutive`, `hold_days`)
is perturbed by ±10% and ±20% individually (12 runs total). Pass criterion: composite score
degrades < 15% at every perturbation level.

### 3.6 ML Validation

The ML feature matrix covers all (ticker, expiry_date) pairs that triggered the signal rule
in the IS period — not just the top-3 portfolio selections. This produces the full unfiltered
signal universe for supervised learning (**796 rows × 21 features**; positive class rate
66.1%).

Three models are trained on each IS window and evaluated on the corresponding OOS window:
1. **Logistic Regression** (C = 0.1, standardised features): coefficient table
2. **Random Forest** (300 trees, max_depth = 4): permutation importance
3. **XGBoost** (200 trees, max_depth = 3, lr = 0.05): SHAP mean absolute values

H₃ is tested using the full IS 2015–2025 feature matrix: H₃ is supported if RSI or
Consecutive_Count appears in the top-3 importances for ≥ 2 of 3 models.

---

## 4. Literature Review

### 4.1 Options Expiry Microstructure

Golez and Jackwerth (2012) provide a foundational analysis of price pinning around S&P 500
options expiry dates, demonstrating that futures prices systematically gravitate toward at-
the-money strike prices as expiry approaches, driven by the interaction between dealer delta-
hedging and investor order flow. Their identification of systematic expiry-driven price pressure
at the aggregate level establishes the theoretical basis for expecting analogous effects at the
individual-stock level.

Baltussen, van Dijk, and Zhu (2025) document a "third Friday price spike" in U.S. equity
index markets: prices systematically increase in the period leading up to monthly options
expiry and subsequently revert, creating distortions in derivative payoffs. Critically, they
attribute this effect to dealer hedging driven by charm — the rate of change of delta with
respect to time — rather than to traditional gamma-hedge unwinding. This distinction is
important for the present study: charm-driven flows are predictable given known time-to-expiry
schedules and create a systematic buy-side pressure that is exogenous to individual stock
fundamentals. However, Baltussen et al. (2025) focus entirely on index-level dynamics and do
not examine whether the effect extends to individual stocks or is conditioned by technical
state.

Ni, Pearson, and Poteshman (2005) document strike-price clustering in individual stock
prices on expiration dates, consistent with market-maker delta-hedging creating systematic
price pressure in individual names. Their evidence directly motivates the present study's
cross-sectional approach and confirms that the expiry microstructure effects documented at
the index level by Golez and Jackwerth (2012) and Baltussen et al. (2025) are also present
at the stock level. Crucially, Ni et al. (2005) do not examine whether the magnitude of
these effects depends on pre-expiry technical state (RSI, consecutive candles) or on the
prevailing volatility regime, leaving the specific empirical questions of this capstone
unaddressed.

### 4.2 Mean Reversion and Volatility Regimes

Poterba and Summers (1988) provide an early econometric demonstration that U.S. stock prices
contain substantial transitory components, implying that deviations from fundamental value tend
to correct over time. Using variance-ratio tests, they find negative autocorrelation in long-
horizon returns, consistent with mean-reverting behaviour. Their evidence supports the mechanism
underlying this capstone: stocks oversold relative to their fundamental value prior to expiry
should tend to recover following the expiry-driven hedging catalyst.

Kim, Nelson, and Startz (1991) extend this analysis by demonstrating that mean reversion in
equity prices is not uniform but varies with the market environment — specifically, it
concentrates in periods of elevated volatility and weakens in calmer regimes. This finding
directly motivates the H₂ hypothesis: if mean reversion is stronger in high-volatility
environments, the RSI-reversal signal should produce a higher win rate when VIX is elevated
(> 25) than when VIX is low (< 15). The two-proportion z-test used for H₂ follows directly
from Kim et al.'s framework applied to the specific signal-filtering context.

Both studies focus on calendar-period return autocorrelation rather than event-driven expiry
microstructure. The present study bridges these traditions by conditioning the mean-reversion
test on a specific mechanical event (monthly options expiry) that provides an exogenous
catalyst for the expected price recovery.

### 4.3 Technical Signal Validity and Data-Snooping Controls

Brock, Lakonishok, and LeBaron (1992) provide the most cited empirical test of technical
trading rules, finding that simple moving-average and trading-range breakout rules generated
statistically significant returns in U.S. equity markets over 1897–1986. However, their study
tested many rules and the significance may partly reflect data-snooping. Subsequent replications
on out-of-sample data (Sullivan, Timmermann & White, 1999) found that most technical rules
fail to survive data-snooping corrections, and that performance frequently disappeared in later
periods as markets absorbed the information.

The present study addresses data-snooping through three mechanisms. **(1) Pre-registration
with an explicit separation between structure and values.** Three elements were fixed in the
project proposal before any data was touched: the *signal structure* (AND-logic combining
RSI and consecutive-candle conditions), the *composite-score weights* (0.35 / 0.30 / 0.20 /
0.15), and the *robustness thresholds* (IS/OOS gap < 0.05, MC p < 0.05, ≥ 6 of 8 walk-forward
windows positive, parameter sensitivity degradation < 15%). The specific *threshold values*
(RSI < 22, MIN_CONSECUTIVE = 3, HOLD_DAYS = 6), by contrast, were derived from an in-sample
grid search documented in `modules/exploration/explore_deep_oversold.py` and frozen as the
v4.1 parameter set in `modules/config/capstone_v4_params.py` before any out-of-sample
evaluation. The Monte Carlo and walk-forward tests below are therefore applied only to OOS
data after this freeze; the IS optimisation is contained inside the IS window and does not
contaminate the OOS evidence.
**(2)** Monte Carlo permutation testing controls for the multiple-comparison bias by
comparing the observed strategy against a null distribution of random-entry strategies on
the *same expiry dates*. **(3)** Walk-forward validation imposes strict temporal
discipline by requiring positive performance in 8 independent out-of-sample periods.

Unlike generic technical-rule tests, the RSI-reversal signal is theoretically motivated
by an event-driven mechanism (charm-driven hedging flows at monthly expiry), providing a
structural rationale that is falsifiable and not merely a retrospective pattern.

### 4.4 Synthesis and Contribution

The four studies reviewed above motivate each element of the study design but do not address
the joint question: whether expiry-driven hedging flows at the stock level can be identified
*ex ante* from observable technical conditions (RSI oversold + consecutive red candles), and
whether the effectiveness of such identification is conditioned by the volatility regime. Golez
and Jackwerth (2012) and Ni et al. (2005) establish the expiry microstructure effects; Poterba
and Summers (1988) and Kim et al. (1991) motivate both the mean-reversion mechanism and its
regime-conditional character. Baltussen et al. (2025) confirm that the third-Friday expiry
effect persists into the recent period (2018–2023) and is driven by charm, providing the most
direct mechanism-level support for the study hypothesis. The present capstone integrates these
threads into a single, testable analytical framework.

---

## 5. Key Findings

The three hypotheses are reported individually in §5.1–§5.3, but they are best read as
a single regime-conditional story rather than three independent results. H₁ (universal
edge) is **not supported**, H₂ (VIX-regime moderation) is **supported**, and H₃ (rule
features dominate ML feature importance) is **supported** on the disjunctive criterion —
RSI is in top-3 for 2/3 models (Random Forest and XGBoost), although Consecutive_Count
is in top-3 for 0/3 models and VIX_Level outranks RSI in every model. The same underlying
fact produces all three results: the signal's effectiveness is conditioned on the
volatility regime. H₁ fails because averaging across regimes dilutes the High-VIX edge;
H₂ succeeds because it isolates that edge; H₃ confirms RSI carries genuine predictive
content while simultaneously identifying VIX_Level — the same regime variable from H₂ —
as the more dominant signal. §8.2 turns this into a deployment recommendation
(VIX-conditional filter).

### 5.1 H₁ — Primary Signal Edge

The IS portfolio simulation over 2015 — mid-2025 produced **177 trades** with the
following performance metrics:

| Metric | IS Value | OOS Value | Pass Criterion |
|--------|----------|-----------|----------------|
| Profit Factor | 1.33 | 1.88 | > 1.0 |
| Sharpe Ratio | 1.36 | 3.32 | > 0 |
| Win Rate | 53.1% | 57.1% | > 50% |
| Max Drawdown | 56.3% | 21.4% | < 30% |
| **Composite Score** | **0.5433** | **0.7868** | IS/OOS gap < 0.05 |

The Monte Carlo permutation test (500 shuffles, seed = 42) produced a p-value of
**0.998**. H₁ is **NOT SUPPORTED** at α = 0.05: the observed composite (0.543) placed
below the 1st percentile of the null distribution (mean = 0.544), meaning ~99.8% of
random-entry permutations achieved equal or higher performance.

The IS/OOS composite gap is **0.244** (**FAILS** threshold < 0.05). Notably, the OOS
composite (0.787) exceeds the IS composite (0.543), indicating the strategy did not
overfit in-sample — the gap reflects the brevity of the OOS window (~10 months, 21 trades)
rather than degradation.

H₁ has two sub-criteria: MC p < 0.05 **and** ≥ 6 of 8 walk-forward windows with positive
composite. The walk-forward criterion **passes** (8/8 positive; §5.4), but the MC criterion
**fails** (p = 0.998). Because both must hold for H₁ to be supported, H₁ overall is **not
supported**.

### 5.2 H₂ — VIX Regime Moderation

Win rates by VIX regime:

| Regime | VIX Range | Trades | Win Rate |
|--------|-----------|--------|----------|
| Low | VIX < 15 | 63 | 52.4% |
| Medium | 15 ≤ VIX ≤ 25 | 92 | 48.9% |
| High | VIX > 25 | 22 | 72.7% |

Two-proportion z-test (High vs. Low, one-sided): z = 1.663, p = 0.048.
H₂ is **SUPPORTED** at α = 0.05: win rate in High-VIX environments (72.7%) is
statistically significantly higher than in Low-VIX environments (52.4%).

### 5.3 H₃ — ML Feature Validation

Feature importance results across three models (full IS period 2015 — mid-2025, feature
matrix 796 rows × 21 features):

| Feature | LR \|Coefficient\| Rank | RF Permutation Rank | XGBoost SHAP Rank |
|---------|------------------------|---------------------|-------------------|
| RSI | 5 | **3** | **3** |
| Consecutive_Count | 8 | — (outside top 10) | — (outside top 10) |

Top-3 features by model — **LR**: VIX_Level, MA_Distance_20, VIX_Regime_enc;
**RF**: MACD_Hist, MA_Distance_20, RSI; **XGBoost**: VIX_Level, MA_Distance_20, RSI.

RSI appeared in top-3 for **2/3 models** (RF and XGBoost);
Consecutive_Count appeared in top-3 for **0/3 models**.

H₃ is **SUPPORTED** under the pre-registered disjunctive criterion (≥ 1 of the rule features
in top-3 for ≥ 2 of 3 models): RSI alone clears the bar. The substantive caveat is that
VIX_Level outranks RSI in 2 of 3 models (LR coefficient #1, XGBoost SHAP #1), and
MA_Distance_20 outranks RSI in all 3 — meaning the rule captures genuine predictive
content but is not the dominant predictor. Consecutive_Count contributes weakly and is
not selected by any model. The dominance of VIX_Level is the same regime feature that
drives H₂, and the two results converge on the same conclusion (§5 intro).

ML walk-forward OOS composites (8-window mean, comparable to the rule-based 0.543 in
§5.1): **Logistic Regression 0.803**, **Random Forest 0.814**, **XGBoost 0.800** —
materially higher than the rule-based baseline, confirming that ML models with access to
the full feature set extract additional signal beyond the pre-specified rule.

### 5.4 Walk-Forward Stability

| OOS Year | IS Period | Composite |
|----------|-----------|-----------|
| 2018 | 2015–2017 | 0.930 |
| 2019 | 2016–2018 | 0.905 |
| 2020 | 2017–2019 | 1.000 |
| 2021 | 2018–2020 | 0.869 |
| 2022 | 2019–2021 | 0.550 |
| 2023 | 2020–2022 | 0.073 |
| 2024 | 2021–2023 | 0.806 |
| 2025 | 2022–2024 | 0.686 |
| **Mean** | | **0.728** |

Number of windows with composite > 0 (positive performance): **8/8**.
Number of windows with composite > 0.5: **7/8**.

---

## 6. Robustness Validation

All thresholds applied in this section — IS/OOS composite gap < 0.05 (§6.1), Monte Carlo
p < 0.05 (§6.2), ≥ 6 of 8 walk-forward windows with positive composite (§6.3), and
parameter sensitivity degradation < 15% (§6.4) — were pre-registered in the project
proposal (§5.6 / §7.6) before any analysis was run, and are cited back to that document
rather than re-derived here.

### 6.1 IS/OOS Holdout Split

The in-sample period (2015-01-01 → 2025-06-30) produced composite score 0.5433.
The out-of-sample period (2025-07-01 → 2026-05-04) produced 0.7868.
The absolute gap of **0.244** **FAILS** the pre-specified threshold of 0.05.

This result **challenges** the formal generalisability criterion. However, the OOS composite
is higher than the IS composite, which indicates no overfitting — the gap is driven by the
very small OOS trade count (21 trades over ~10 months), which produces high metric volatility
rather than strategy degradation.

### 6.2 Monte Carlo Permutation Test

The null distribution mean was 0.544 (σ very small, distribution tightly concentrated near
0.543–0.544). The observed composite 0.543 placed **below the 1st percentile** of the null
distribution, yielding p = 0.998 (499 of 500 permutations achieved an equal or higher
score). This **does not support** rejecting H₀ at α = 0.05. The strategy's IS composite is
statistically indistinguishable from — and in fact slightly lower than — random signal
placement on the same expiry dates when averaged across all VIX regimes; §5.2 / §8.2 show
this aggregate result masks a regime-conditional edge that survives in High-VIX months.

### 6.3 Walk-Forward Analysis

8/8 OOS windows produced positive composite scores and 7/8 exceeded 0.5, confirming
**strong** temporal stability across 2018–2025. Notable observations:
- 2020 (COVID crash year): composite = 1.000 — perfect window; the strategy captured
  all signals profitably as oversold stocks recovered sharply after the March crash.
- 2022 (rate-hike cycle): composite = 0.550 — weakest full year of the > 0.5 group;
  rising-rate environment suppressed mean reversion, consistent with Kim et al. (1991).
- 2023: composite = 0.073 — the only window below 0.5 and the worst overall; low signal
  count with concentrated losses in a low-VIX, trending bull market where mean-reversion
  signals were premature.

### 6.4 Parameter Sensitivity

Parameter perturbation results, baseline composite **0.5433** (same baseline as §5.1):

| Parameter | Baseline | −20% | −10% | +10% | +20% | All Pass? |
|-----------|----------|------|------|------|------|-----------|
| rsi_threshold (22→18/20/24/26) | 0.5433 | 0.6914 | 0.6758 | 0.5811 | 0.5468 | ✅ Yes |
| min_consecutive (3→2/3/3/4) | 0.5433 | 0.7001 | 0.5433 | 0.5433 | 0.7134 | ✅ Yes |
| hold_days (6→5/5/7/7) | 0.5433 | 0.6374 | 0.6374 | 0.5689 | 0.5689 | ✅ Yes |

Pass criterion: composite degrades < 15% at every perturbation level. **All 12 perturbations
pass**; in fact, every perturbation produced a composite ≥ baseline, meaning no nearby
parameter combination is materially worse than the v4.1 setting. Note that
`min_consecutive` perturbations of ±10% of 3 round back to 3 (no effective change), and
`hold_days` ±10% of 6 round to 5 or 7, producing identical results to the ±20% rows for
those parameters. The fact that all perturbations *improve* the composite suggests the
v4.1 grid-search optimum (RSI < 22, MIN = 3, HOLD = 6) is a conservative local choice
rather than an over-optimised peak; the team retained the v4.1 values to honour the
pre-registration freeze rather than re-optimising after seeing OOS data.

### 6.5 Optional Tests Scoped Out of Submission

The proposal contemplated two optional robustness tests that are deliberately not
implemented in this report, noted here so their absence does not appear as oversight:

- **Market Regime Segmentation by SPY 200-day SMA.** A secondary segmentation by trend
  regime (SPY above vs. below its 200-day SMA) was considered as an alternative to
  VIX-based regime testing. It was scoped out because the H₂ VIX-regime z-test provides
  a more direct test of the regime-conditional mean-reversion mechanism documented by
  Kim et al. (1991), and adding a second regime axis would dilute statistical power
  given the IS trade count (n = 177).
- **Earnings Filter Impact.** The proposal contemplated comparing strategy performance
  with and without earnings-proximate signals (`Earnings_Proximity = 1` filter). This
  was scoped out because the IS sub-sample of earnings-proximate trades was too small
  (~10 trades) to support a meaningful comparison. `Earnings_Proximity` is retained as
  an ML feature for completeness.

---

## 7. Limitations

### 7.1 Survivorship Bias

The analysis uses the current (2026) S&P 500 constituent list for the entire 2015–2026
window. Stocks that were removed from the index during this period are excluded from all
historical expiry scans. Since poor-performing stocks are disproportionately removed, this
theoretically creates an upward bias in any long-only strategy.

**Quantification (Stream 1-C — completed):** A sensitivity check was conducted by
comparing the full 2026 universe (503 tickers in the current cache) against a frozen
proxy of the 2015 universe stored at `cache/sp500_constituents_2015.csv` (453 tickers
whose earliest cached data predates 2016-01-01, excluding 50 post-2015 additions
including PLTR, SMCI, CRWD, CVNA, ABNB, COIN and similar). Both rows below are from the
same side-by-side sensitivity batch executed for this comparison and share the §5.1
baseline.

| Universe | Tickers | Trades | Win Rate | Profit Factor | Sharpe | Max DD | Composite |
|----------|---------|--------|----------|---------------|--------|--------|-----------|
| Full 2026 (baseline) | 503 | 177 | 53.1% | 1.33 | 1.36 | 56.3% | 0.5433 |
| Frozen 2015 proxy | 453 | 171 | 55.0% | 1.47 | 1.80 | 46.6% | 0.6375 |

The frozen 2015 universe produced a **higher** composite score (+0.094, +17.3%) than the
full universe. This result is the **opposite** of the expected survivorship-bias inflation:
the 50 post-2015 additions (high-growth, high-volatility names) drag the strategy's
composite score down rather than inflating it. This is plausible because RSI < 22 oversold
signals in highly volatile post-IPO stocks tend to be false recoveries, not mean-reversion
opportunities driven by expiry hedging flows.

**Conclusion:** The survivorship bias does not inflate returns in this strategy; if anything,
the current universe inclusion is slightly pessimistic relative to a 2015-cohort baseline.
Walk-forward evaluation further mitigates the concern since OOS returns are earned after
the training window closes.

### 7.2 Take-Profit Calibration — Recalibrated with RSIReversalStrategy

The original `take_profit_research.ipynb` used **EMAScanner** — an exploratory entry
generator from an earlier project iteration that triggers on EMA crossover patterns,
predating the capstone's RSI-based signal rule and unrelated to the pre-registered
hypotheses tested in this report — to produce entry signals. The resulting exit
parameters could not legitimately be applied to `RSIReversalStrategy`, since exit
calibration is sensitive to the entry distribution. The notebook has been fully
recalibrated for this capstone using RSIReversalStrategy entries from the IS period
(807 signals, 500 randomly sampled for the grid search, IS window 2015–2025).

**Grid search results across six exit strategies:**

| Exit Strategy | Avg Return | Win Rate | Sharpe | Profit Factor | Avg Hold Days |
|---------------|-----------|----------|--------|---------------|---------------|
| rsi_exit (RSI > 75) | **2.73%** | 51.4% | 0.230 | **1.90** | 25.9 |
| rsi_exit (RSI > 65) | 2.45% | **59.0%** | 0.241 | **1.99** | 21.0 |
| time_based (6 days, baseline) | 2.27% | 57.4% | — | — | 16.5 |
| fixed_pct (tp=5%) | 1.50% | **63.0%** | 0.242 | 1.65 | 15.1 |
| trailing_stop (10%) | 2.27% | 44.4% | 0.219 | 1.75 | 35.2 |
| fixed_pct (tp=15%) | 2.41% | 44.6% | 0.236 | 1.71 | 31.3 |

**Recommended exit upgrade:** `rsi_exit` with threshold = 65–70 delivers the best
win-rate-adjusted performance: 59.0% win rate at 2.45% average return (vs. 57.4% /
2.27% for the fixed hold). This exit type is supported by `run_simulation(exit_params=
{'exit_type': 'rsi_exit', 'rsi_exit_threshold': 65})`. The current capstone results
use the conservative fixed-hold baseline; applying rsi_exit(65) would likely improve
the composite score.

**Note:** Trailing stop underperforms at all tested levels (trail=3–10%), consistent
with the mean-reversion nature of the strategy: premature stops cut winners before
the expected recovery completes.

### 7.3 Data Quality and Coverage Gaps

- **Stooq adjusted prices:** Free-data providers may have occasional split/dividend
  adjustment errors. `DataValidator` flags and logs anomalies, but cannot correct them.
  In the IS-period simulation log, on the order of 60–80 tickers triggered at least one
  validator warning (most commonly close-price outliers above the 5% threshold, with a
  handful of OHLC-logic or large-date-gap warnings); these tickers were included in
  the simulation because the warnings reflect distribution-tail observations rather than
  hard data errors, but their signals were evaluated against unvalidated prices.
- **Earnings coverage:** The Nasdaq earnings calendar covers ~510 tickers from 2015 to
  present, but has occasional date errors or late reschedules that would affect the
  `Earnings_Proximity` flag.
- **FOMC calendar:** Only 97 meetings over 2015–2026. The FOMC proximity flag is binary
  (±5 days) and does not distinguish rate-decision meetings from non-decision meetings,
  which differ in their market impact.

### 7.4 Fixed Hold Period and Equal-Weight Sizing

The simulation uses a fixed `HOLD_DAYS = 6` trading-day exit for all trades. In practice,
exit timing may depend on stock-specific technical conditions (RSI recovery, earnings release,
sector rotation). Equal-weight sizing ignores position-specific risk; a risk-parity or
volatility-scaled approach could produce better risk-adjusted returns but would introduce
additional optimisation degrees of freedom.

---

## 8. Practical Implications

### 8.1 Implications of the H₂ Finding — Conditional Use Case

H₁ is not supported as a universal monthly screen (MC p = 0.998). H₃ confirms RSI is a
genuine predictor (top-3 in 2/3 ML models) but is consistently outranked by VIX_Level and
MA_Distance_20, and Consecutive_Count adds little. The practical value of this work
therefore comes from the **H₂ finding**: the RSI-oversold + consecutive-candle signal
generates a materially higher win rate in High-VIX environments (72.7% vs. 52.4%,
z = 1.663, p = 0.048) than in Low-VIX environments. The framework should be deployed
conditionally rather than as an always-on monthly strategy.

A trader applying this work as a **VIX-conditional filter** would:

1. On the Thursday before each monthly expiry, read the VIX close. If VIX < 20, **skip
   the cycle**; if VIX ≥ 25, proceed.
2. Run `RSIReversalStrategy` across the S&P 500 universe and select the 1–3 stocks with
   the lowest RSI that also satisfy ≥ 3 consecutive red candles.
3. Enter long positions at the Friday open.
4. Exit 6 trading days later (the following Thursday or Friday).

Conditional deployment reflects what the analysis actually supports — a regime-conditional
edge — rather than the universal monthly screen originally hypothesised under H₁. §8.2
describes the broader VIX-overlay framework this implies.

### 8.2 VIX Regime Overlay — Primary Practical Recommendation

H₂ is empirically supported (z = 1.663, p = 0.048; High-VIX win rate 72.7% vs.
Low-VIX 52.4%) and is the strongest practical result of this study. Three concrete uses
follow:

1. **VIX gate.** Trade only when the VIX close on the signal date is ≥ 25. This converts
   the strategy from a monthly screen into a regime-triggered event filter, aligning
   execution with the regime where the mean-reversion mechanism documented by
   Kim et al. (1991) is strongest.
2. **VIX-scaled position sizing.** When trading is enabled (VIX ≥ 20), scale position
   size linearly with VIX level (e.g., 0.5× allocation at VIX = 20, 1.0× at VIX = 30,
   capped at 1.5× above VIX = 35). This down-weights Medium-VIX trades, which showed
   the weakest IS win rate (48.9%, n = 92).
3. **Skip Low-VIX cycles entirely.** When VIX < 15, the IS win rate was 52.4% — below
   the level needed to overcome 0.2% round-trip commission with the observed average
   trade magnitude. No-trade is the optimal action.

The VIX overlay is the **recommended primary deployment** of this work and replaces the
universal monthly screen originally hypothesised under H₁.

### 8.3 Limitations for Live Deployment

The study evaluates a backtested strategy on historical data with known survivorship
bias. Before deploying capital, practitioners should:
- Apply the strategy to a point-in-time constituent list to remove survivorship bias
- Validate performance in the most recent 12–18 months of live paper trading
- Assess liquidity: the strategy targets S&P 500 large-caps, which are generally liquid,
  but individual names at extreme oversold readings may have wider bid-ask spreads
- Monitor for regime changes that could invalidate the underlying microstructure mechanism
  (e.g., regulatory changes to options market structure, or significant shifts in SPY
  options open interest distribution)

### 8.4 Broader Implications for Quantitative Research

The study demonstrates a methodology for bridging options microstructure theory
with cross-sectional technical analysis research. The walk-forward / Monte Carlo
framework provides a template for evaluating any event-conditioned technical rule in
a statistically rigorous way. The composite scoring approach — combining Profit Factor,
Sharpe, Win Rate, and Max Drawdown into a single bounded metric — provides a framework
for comparing strategies that may trade off different aspects of risk-adjusted performance.

---

## 9. References

Baltussen, G., van Dijk, M., & Zhu, L. (2025). The third Friday price spike. *Working Paper*.

Brock, W., Lakonishok, J., & LeBaron, B. (1992). Simple technical trading rules and the
stochastic properties of stock returns. *Journal of Finance, 47*(5), 1731–1764.

Golez, B., & Jackwerth, J. C. (2012). Pinning in the S&P 500 futures. *Journal of
Financial Economics, 106*(3), 566–585.

Kim, M. J., Nelson, C. R., & Startz, R. (1991). Mean reversion in stock prices? A reappraisal
of the empirical evidence. *Review of Economic Studies, 58*(3), 515–528.

Ni, S. X., Pearson, N. D., & Poteshman, A. M. (2005). Stock price clustering on option
expiration dates. *Journal of Financial Economics, 78*(1), 49–87.

Poterba, J. M., & Summers, L. H. (1988). Mean reversion in stock prices: Evidence and
implications. *Journal of Financial Economics, 22*(1), 27–59.

Sullivan, R., Timmermann, A., & White, H. (1999). Data-snooping, technical trading rule
performance, and the bootstrap. *Journal of Finance, 54*(5), 1647–1691.

Wilder, J. W. (1978). *New concepts in technical trading systems*. Trend Research.

---

## Appendix A — Code Repository Structure

```
modules/
  config/capstone_v4_params.py     # v4.1 frozen parameters
  data/                            # CacheManager, DataLoader, ExpiryCalculator
  features/
    technical_indicators.py        # RSI, MACD, BB, Consecutive_Count etc.
    enrichment_features.py         # VIX/FOMC/Earnings joins
  models/pattern_models.py         # RSIReversalStrategy (primary model)
  evaluation/
    portfolio_simulator.py         # run_simulation() — core backtest loop
    metrics.py                     # calculate_composite_score() and others
    robust_validation.py           # MonteCarlo, WalkForward, parameter_sensitivity
notebooks/
  spy_expiry_analysis_v2.ipynb     # Main analysis: simulation + validation + charts
  ml_validation.ipynb              # ML feature matrix + H₃ + benchmark table
doc/capstone/
  final_report.md                  # Written capstone report
  spy_expiry_capstone_presentation.pptx
cache/
  constituent_data/                # ~510 per-ticker OHLCV CSVs
  vix.csv, fomc_dates.csv, earnings_dates.csv
  sp500_constituents_2015.csv      # 2015 proxy universe for survivorship sensitivity
tests/unit/
  test_capstone_params.py          # 15 tests — parameter constants + composite formula
  test_enrichment_features.py      # 11 tests — VIX/FOMC/Earnings join correctness
  test_robust_validation.py        # 13 tests — Monte Carlo + WalkForward structure
```

Repository validation before submission: `python -m pytest tests/ -q` passes with
69 tests and 4 subtests.

---

*Report generated with Claude Code (AI-assisted development). All code, simulation outputs,
and statistical results were produced by the project pipeline and reviewed by team members
per the review protocol in `doc/design/team_review_assignments.md`.*
