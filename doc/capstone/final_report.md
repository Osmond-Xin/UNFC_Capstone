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

The options-expiry microstructure literature documents a real phenomenon: on monthly
expiry dates, dealer charm/delta hedging exerts systematic, non-fundamental price pressure
on the underlying (Baltussen et al., 2025; Golez & Jackwerth, 2012; Ni et al., 2005),
which — combined with the mean-reverting component of equity prices (Poterba & Summers,
1988; Kim et al., 1991) — could produce a predictable reversal window. This study tests
whether that **index- and option-level** mechanism can be captured at the **individual-stock
level**, *ex ante*, by an operationally simple and retail-executable technical rule:
RSI oversold (RSI < 22) combined with consecutive bearish candles (≥ 3 red, close-to-close),
entered on SPY monthly expiry days. Using daily OHLCV data for all current S&P 500
constituents over 2015–2026, enriched with VIX, FOMC, and earnings data, a realistic T+1
portfolio simulation is run with top-3 selection per expiry, 0.2% round-trip commission,
and a 15-position cap.

Three directional hypotheses were pre-registered: the signal produces an edge distinguishable
from a permutation null (H₁); that edge is moderated by the VIX volatility regime (H₂); and
supervised ML confirms the rule features dominate the predictive structure (H₃). The signal
*structure* (AND-logic of RSI and consecutive candles) and the composite-score weights were
fixed in advance; the numeric thresholds (RSI < 22, ≥ 3 candles, 6-day hold) were selected by
in-sample grid search and then frozen before any out-of-sample evaluation. Validity is
assessed via 500-iteration Monte Carlo permutation testing, 8-window walk-forward analysis,
parameter-sensitivity perturbation, three supervised ML models (Logistic Regression, Random
Forest, XGBoost + SHAP), and a large-sample re-test of H₂ on the full triggered-signal
universe rather than the small top-3 portfolio subset.

**All three hypotheses are not supported.** H₁: the pre-registered permutation gate fails
(Monte Carlo p = 1.000), but that gate is a degenerate return-sequencing null (§6.2, §7.5) and
is not the load-bearing test. The valid test is a **portfolio-level random-entry Monte Carlo**
(1,000 random portfolios on the rule's own entry dates): the rule's composite (0.557) sits at
the **62nd percentile**, **p = 0.378** — statistically indistinguishable from random
stock-picking. A direct **SPY benchmark** confirms this: matched trade-for-trade to SPY over
identical holding windows, the rule's picks earn a non-significant +0.21%/trade excess
(p = 0.627, ≈ zero after costs) and compound to **+101% vs. +166%** for simply buying SPY on
the same dates, with a 54.1% drawdown against SPY's 34.2% — less return for more risk than
doing nothing. H₂: on
the **full 718-signal in-sample universe** (not the 22-trade portfolio subset that produced the
original 72.7% vs. 52.4% gap), the High-VIX win rate (67.6%, n = 207) is essentially equal to
the Low-VIX win rate (67.3%, n = 110); z = 0.065, p = 0.474, with only a faint continuous
association remaining (Spearman ρ = 0.076, p = 0.042). The apparent regime edge was a
small-sample artifact of top-3 selection. H₃: RSI appears in the top-3 importances of only
1 of 3 models and Consecutive_Count in 0 of 3; the dominant features are VIX_Level,
MA_Distance, and MACD_Hist — the rule features are not the predictive drivers.

The three concordant null results, read alongside the high-arbitrage nature of large-cap U.S.
equities (McLean & Pontiff, 2016; Sullivan et al., 1999), converge on a single conclusion:
**although the expiry hedging-pressure mechanism is real at the index/option level, it cannot
be profitably captured, net of costs, at the individual-stock level by this simple ex-ante
technical rule in the modern, liquid S&P 500 universe.** The contribution of this study is
therefore not a deployable strategy but (i) a reusable, rigorous framework for falsifying
event-conditioned technical rules, and (ii) a correctly interpreted null result that separates
market beta from stock-selection alpha, and mechanism existence from tradeability.

*Keywords:* options expiry, dealer hedging, mean reversion, market efficiency, falsification,
Monte Carlo permutation, walk-forward validation, S&P 500

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
> that shows evidence of stock-selection alpha beyond broad-market beta, after costs and
> across out-of-sample periods?

### 1.3 Hypotheses

The signal structure, composite-score weights, and robustness gates were pre-registered before
simulation. Numeric threshold values were selected inside the in-sample window and then frozen
before out-of-sample evaluation. All tests are evaluated at α = 0.05:

**H₁ (Primary):** The RSI-oversold + consecutive-candle signal produces a composite score
that passes the pre-registered Monte Carlo permutation criterion (500 return-order shuffles,
seed = 42) and exhibits positive composite scores in ≥ 6 of 8 walk-forward OOS windows. This
permutation gate is a path-ordering diagnostic, not a random-entry/ticker-selection test; §7.5
explains the limitation.

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
**2015-01-01 through 2026-05-22** (the latest cached trading day used by the canonical
current-cache IS/OOS simulation in `spy_expiry_analysis_v2.ipynb`; the in-sample window
ends 2025-06-30 and the out-of-sample window runs 2025-07-01 → 2026-05-22), yielding ~11
years of monthly expiry cycles across ~136 expiry dates.

`CacheManager` operates in two modes selected automatically at runtime. In **full-download
mode**, it iterates all ~510 tickers and fetches complete OHLCV history from Stooq.com
(free, no API key required), writing one CSV per ticker to `cache/constituent_data/`.
A polite 1-second delay between requests avoids rate-limiting. The download is resumable:
tickers whose CSV already exists are skipped, so an interrupted run can continue without
re-fetching completed files. In **incremental mode**, `CacheManager` detects that CSVs
are already present, identifies the most recent cached date for each ticker, and appends
only the missing trading days using the Alpaca Markets REST API (batch size 50, 0.5 s
delay, within the free-tier 200 req/min limit).

Data integrity is enforced by `DataValidator` (`modules/data/data_validator.py`), which checks
each ticker's DataFrame for: presence of the required OHLCV columns; NaN values in critical
columns; large date gaps (> 10 trading days); volume and close-price outliers (IQR method,
reported only when > 5% of rows are affected); single-day extreme price changes (> 50%); and
OHLC logic consistency (e.g., High ≥ Low, Close within range). Problems are flagged and logged
to the caller rather than silently dropped, so the analysis can document which tickers were
excluded and why. (The validator does not check for duplicate dates or price monotonicity;
an earlier draft incorrectly listed those.) A full `DataValidator` quality report
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
(seed = 42 set once before the first shuffle). This re-orders the realized returns of the
*already-selected* trades, so it tests whether the observed return *sequence* (and hence the
equity path / drawdown) is unusual — it is a sequencing/path null, **not** a random-ticker or
random-entry null, and it leaves the order-invariant composite components (Profit Factor,
Sharpe, Win Rate) unchanged. Each permuted version's composite score forms the null
distribution; the p-value is `sum(null ≥ observed) / 500`. §7.5 documents the resulting
degeneracy when the observed drawdown exceeds the composite's 30% cap, and reports the
correct **portfolio-level random-entry** test we ran in its place (rule at the 62nd percentile
of the random-entry null, p = 0.378 — no significant selection skill).

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
signal universe for supervised learning (**801 rows × 21 features**; positive class rate
66.0%; 718 rows fall in the in-sample window). This same full-universe matrix is used to
**re-test H₂ at adequate sample size** (§5.2): the High/Low-VIX win-rate z-test and a
continuous logistic regression of win probability on the raw VIX level are computed on all
triggered signals, rather than on the 22-trade High-VIX portfolio subset that produced the
original H₂ result. The small-sample vs. large-sample comparison is implemented and plotted
in `ml_validation.ipynb` §3b.

Three models are trained on each IS window and evaluated on the corresponding OOS window:
1. **Logistic Regression** (C = 0.1, standardised features): coefficient table
2. **Random Forest** (300 trees, max_depth = 4): permutation importance
3. **XGBoost** (200 trees, max_depth = 3, lr = 0.05): SHAP mean absolute values

H₃ is tested using the full IS 2015–2025 feature matrix: H₃ is supported if RSI or
Consecutive_Count appears in the top-3 importances for ≥ 2 of 3 models.

### 3.7 Market-Beta Benchmarks

Three benchmarks separate market beta from stock-selection alpha. All are implemented in
`outputs/compute_benchmarks.py`, which reads the canonical `trades_df` and is re-runnable
end-to-end. **(1) SPY matched-window benchmark.** Each of the 177 IS trades is matched to
SPY's return over its identical entry→exit window; the paired difference (stock − SPY) is the
per-trade selection excess, tested with a paired *t*-test, and the two return series are also
compounded over the same windows for a capital-comparable cumulative comparison. SPY daily
data in the cache begins 2018-01-02, so calendar buy-and-hold figures cover 2018–2026.
**(2) Portfolio-level random-entry Monte Carlo.** 1,000 random portfolios (seed = 42) are
drawn using the rule's own entry dates and per-date trade counts, each random ticker held the
same 6 days net of the same cost; the rule's composite is located in this null distribution.
This is the valid replacement for the degenerate return-shuffle permutation (§7.5).
**(3) Expiry vs. non-expiry premium.** The forward 6-day net return of every RSI < 22 &
≥ 3-red signal in the IS period is split by whether the signal day immediately precedes a
monthly expiry; means are compared both naively and with each expiry day treated as a single
clustered observation (§8.2).

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
evaluation. The Monte Carlo and walk-forward tests below are therefore applied only after this
freeze; the IS optimisation is contained inside the IS window and does not contaminate the OOS
evidence.
**(2)** Monte Carlo permutation testing is retained as a pre-registered return-ordering
robustness diagnostic, but §7.5 shows that it is degenerate for this composite and should not be
read as a random-entry selection test. **(3)** Walk-forward validation imposes strict temporal
discipline by requiring positive performance in 8 independent out-of-sample periods.

Unlike generic technical-rule tests, the RSI-reversal signal is theoretically motivated
by an event-driven mechanism (charm-driven hedging flows at monthly expiry), providing a
structural rationale that is falsifiable and not merely a retrospective pattern.

McLean and Pontiff (2016) provide the complementary cautionary result that publicly
documented return predictors decay substantially after publication — by roughly a third
out-of-sample and more in liquid, easily arbitraged stocks — as informed capital trades
against them. For a simple, widely known oversold pattern in S&P 500 large-caps, this
predicts that any historical edge is likely to have been competed away by the sample period
studied here, framing a null result as the *expected* outcome under market efficiency rather
than a surprising one. This study's contribution is to test that expectation rigorously and
to explain the mechanism-versus-tradeability gap when the null is confirmed.

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

The three pre-registered hypotheses are reported individually in §5.1–§5.3, and they are
best read as a single concordant story: **all three are not supported.** H₁ (the primary
robustness gate) is **not supported** (Monte Carlo p = 1.000); H₂ (VIX-regime moderation)
is **not supported** once tested on the full signal universe rather than the 22-trade
portfolio subset (z = 0.065, p = 0.474); and H₃ (rule features dominate ML importance) is
**not supported** (RSI in top-3 for only 1/3 models, Consecutive_Count for 0/3). The three
results reinforce one another: the pre-registered permutation criterion fails (H₁), though
that test is degenerate (§5.1, §7.5) and the load-bearing evidence is the valid random-entry
Monte Carlo (rule at the 62nd percentile, p = 0.378) together with the SPY benchmark, which
show the rule's picks earn no significant excess over random selection or over simply holding
SPY on the same dates (§5.5); the
one apparent exception — a high-VIX edge — dissolves under proper sample size and is exposed
as an artifact of selecting the three most-oversold names per expiry (H₂); and an independent
supervised-learning view confirms the rule features are not the predictive drivers, which are
instead VIX_Level, MA_Distance, and MACD_Hist (H₃). §5.5 explains *why* — gross profit in this
long-only backtest is dominated by market beta, not stock-selection alpha — and §8 develops the
implication: in the modern liquid S&P 500 universe, this expiry mean-reversion rule is not a
deployable edge, consistent with high-arbitrage market efficiency.

### 5.1 H₁ — Primary Signal Edge

The IS portfolio simulation over 2015 — mid-2025 produced **177 trades** with the
following performance metrics (current-cache run):

| Metric | IS Value | OOS Value | Pass Criterion |
|--------|----------|-----------|----------------|
| Profit Factor | 1.35 | 2.84 | > 1.0 |
| Sharpe Ratio | 1.43 | 5.46 | > 0 |
| Win Rate | 53.1% | 58.3% | > 50% |
| Max Drawdown | 54.1% | 21.4% | < 30% |
| **Composite Score** | **0.5567** | **0.8096** | IS/OOS gap < 0.05 |

(IS window 2015-01-01 → 2025-06-30, 177 trades; OOS window 2025-07-01 → 2026-05-22,
24 trades — the canonical current-cache run in `spy_expiry_analysis_v2.ipynb`.)

The Monte Carlo permutation test (500 shuffles, seed = 42) produced a p-value of
**1.000**, so H₁'s pre-registered MC sub-criterion (p < 0.05) **fails**. We are careful not
to over-read this number. The test shuffles the `net_return` column of the *already-selected*
trades; it therefore holds the set of selected trades fixed and only re-orders their returns,
which changes the equity path (and hence Max Drawdown) but leaves Profit Factor, Sharpe, and
Win Rate unchanged (these are permutation-invariant). Because the observed Max Drawdown
(54.1%) already exceeds the composite's 30% drawdown cap — saturating that sub-score to its
floor of 0 — every permutation necessarily scores ≥ the observed composite, forcing
p = 1.000 as a near-mathematical certainty (§7.5). This permutation test is thus a
return-sequencing/path null, **not** a test of random ticker/entry selection, and it is
degenerate for the present composite.

Because the pre-registered permutation gate is uninformative here, the substantive test of
H₁ is the **portfolio-level random-entry Monte Carlo** described in §7.5: 1,000 random
portfolios drawn on the rule's own entry dates and per-date trade counts, so that Profit
Factor, Sharpe, Win Rate, and Max Drawdown all vary under the null. On that valid test the
rule's composite (0.557) sits at the **62nd percentile** of the random-entry distribution
(null median 0.494), **p = 0.378** — the rule is statistically **indistinguishable from
random stock-picking** on the same dates. This is the load-bearing H₁ result, and it agrees
with the SPY benchmark (§5.5: the rule's picks earn no significant excess over simply buying
SPY for the same holding windows), the regime analysis (§5.2), and the ML feature analysis
(§5.3). **H₁ is not supported.**

The IS/OOS composite gap is **0.253** (**FAILS** threshold < 0.05). Notably, the OOS
composite (0.810) exceeds the IS composite (0.557), so the gap reflects the brevity and
small trade count of the OOS window (~10 months, 24 trades), not in-sample overfitting —
a small sample simply produces high metric volatility. This same small-sample fragility,
seen here in the OOS window, is exactly what inflated the original H₂ result (§5.2).

H₁ has two sub-criteria: MC p < 0.05 **and** ≥ 6 of 8 walk-forward windows with positive
composite. The walk-forward criterion **passes** (8/8 positive; §5.4), but the MC criterion
**fails** (p = 1.000, for the degenerate reason explained in §7.5). Because both must hold
for H₁ to be supported, H₁ overall is **not supported**. The walk-forward positivity is itself explained in §5.5: in a rising
market, long-only entries are positive on average regardless of the entry signal.

### 5.2 H₂ — VIX Regime Moderation

The original test was run on the **177 portfolio trades** (top-3 most-oversold names per
expiry). On that subset the High-VIX win rate looked much higher than Low-VIX, and the
two-proportion z-test was marginally significant:

| Regime | VIX Range | Portfolio top-3 (n) | Win Rate |
|--------|-----------|---------------------|----------|
| Low | VIX < 15 | 63 | 52.4% |
| Medium | 15 ≤ VIX ≤ 25 | 92 | 48.9% |
| High | VIX > 25 | **22** | **72.7%** |

Two-proportion z-test on this subset (High vs. Low, one-sided): z = 1.663, p = 0.048 — which
*would* support H₂. **However, the High-VIX cell contains only 22 trades**, and the same
small-sample volatility that produced the inflated OOS metrics in §5.1 applies here. Because
the pre-registered hypothesis test is restricted to this portfolio subset, we conduct a
**post-hoc robustness expansion** to evaluate the mechanism on the **full triggered-signal universe**
(every ticker×expiry that fired the rule, before top-3 selection; `ml_validation.ipynb` §3b).
On that population the High-VIX sample grows from 22 to 207, and the regime gap **disappears**:

| Regime | VIX Range | Full universe (n) | Win Rate | Portfolio top-3 win rate |
|--------|-----------|-------------------|----------|--------------------------|
| Low | VIX < 15 | 110 | 67.3% | 52.4% |
| Medium | 15 ≤ VIX ≤ 25 | 401 | 66.6% | 48.9% |
| High | VIX > 25 | 207 | 67.6% | 72.7% |

Two-proportion z-test on the full universe (High vs. Low, one-sided): **z = 0.065, p = 0.474**.
A continuous treatment (logistic regression of win probability on the raw VIX level) leaves
only a faint association (coefficient +0.145, likelihood-ratio p = 0.076; Spearman ρ between
VIX and net return = 0.076, p = 0.042). **H₂ is NOT SUPPORTED.** The apparent High-VIX edge
was an artifact of evaluating only the 22 most-oversold High-VIX names; across all High-VIX
signals the win rate (67.6%) is indistinguishable from the Low-VIX win rate (67.3%) and from
the unconditional base rate (~66%). Figure (notebook §3b) plots the two samples side by side.

### 5.3 H₃ — ML Feature Validation

Feature importance results across three models (full IS period 2015 — mid-2025, feature
matrix 801 rows × 21 features, positive class rate 66.0%):

| Feature | LR \|Coefficient\| Rank | RF Permutation Rank | XGBoost SHAP Rank |
|---------|------------------------|---------------------|-------------------|
| RSI | outside top 3 | — (outside top 3) | **2** |
| Consecutive_Count | outside top 3 | — (outside top 3) | — (outside top 3) |

Top-3 features by model — **LR**: VIX_Level, MA_Distance, VIX_Regime_enc;
**RF**: MACD_Hist, MA_Distance_20, BB_Middle; **XGBoost (SHAP)**: VIX_Level, RSI, Volume_Ratio.

RSI appeared in top-3 for only **1/3 models** (XGBoost);
Consecutive_Count appeared in top-3 for **0/3 models**.

H₃ is **NOT SUPPORTED** under the pre-registered criterion (≥ 1 rule feature in top-3 for
≥ 2 of 3 models): only XGBoost ranks RSI in its top-3, and no model ranks Consecutive_Count.
Notably, in the earlier (frozen) cache RSI was borderline top-3 in 2 models; on the current
cache it falls out of the Random Forest top-3 (replaced by BB_Middle). That an indicator's
top-3 membership flips with a minor data refresh is itself evidence that RSI's importance is
**not robust** — it hovers near rank 3 rather than being a stable driver. The consistently
dominant features across both runs are VIX_Level, MA_Distance, and MACD_Hist; the rule
features are not the predictive structure the ML models rely on.

ML walk-forward OOS composites (8-window mean): **Logistic Regression 0.799**,
**Random Forest 0.812**, **XGBoost 0.790** — higher than the rule-based 0.557 baseline
(§5.1). This does **not** indicate a tradeable ML edge: the elevated composites are
produced on the same long-only, top-3 portfolio mechanics in a rising market (the beta
effect explained in §5.5), and the ML models' predictions are driven by VIX_Level and
MA_Distance, not by the pre-registered RSI/consecutive-candle rule.

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
| 2025 | 2022–2024 | 0.781 |
| **Mean** | | **0.739** |

Number of windows with composite > 0 (positive performance): **8/8**.
Number of windows with composite > 0.5: **7/8**.

Walk-forward positivity is **not** in tension with the H₁ null. A positive composite means
the long-only portfolio made money in that OOS year — but so would almost any long-only
entry rule in a rising market (§5.5). The Monte Carlo test (§5.1) fails the pre-registered
criterion, but §7.5 shows it is a return-ordering diagnostic rather than a valid selection-edge
test. Walk-forward confirms the strategy is not unstable or catastrophic, not that it has a
stock-selection edge.

### 5.5 Why the Null — Beta, Not Alpha

The single fact that reconciles every result above is that **the strategy's gross profit is
market beta, not stock-selection alpha.** The simulation is long-only over 2015–2026, a period
in which the S&P 500 rose substantially. In such a market, buying almost any large-cap stock
and holding it six trading days has a positive expected return. Four observations make this
concrete:

1. **The most-oversold selection does not add skill — if anything it subtracts it.** The 177
   top-3 selected trades win only **53.1%** of the time (§5.1), which is *below* the **~67%**
   win rate of the full triggered-signal universe (§5.2). Ranking by lowest RSI picks names
   that recover *less* often than the average signal, so the selection rule shows no positive
   stock-selection skill over the signal set it draws from.
2. **The rule's picks earn no significant excess over SPY on the same dates.** Matching every
   one of the 177 trades to SPY's return over the identical entry→exit window, the per-trade
   excess is **+0.21%** gross (mean stock 0.78% vs SPY 0.58%), **not significant** (paired
   t = 0.49, p = 0.627), and essentially zero after the 0.2% round-trip cost. Compounded over
   the same 177 windows, putting the capital into the rule's picks returns **+101%** versus
   **+166%** for simply buying SPY on those dates, and the rule beats SPY in only **45%** of
   the windows. Over the 2018–2026 span where daily SPY data is available, buy-and-hold SPY
   returned **+151%** (CAGR **11.9%**) with a **34.2%** maximum drawdown — versus the rule's
   **54.1%** in-sample drawdown. The rule delivers *less* return for *more* risk than doing
   nothing.
3. **Regime conditioning adds nothing** once sample size is adequate (H₂ flat at ~67%, §5.2).
4. **ML attributes predictions to VIX_Level and MA_Distance, not the rule** (H₃, §5.3), and
   even the ML composites are earned on the same long-only mechanics in a rising market.

The 54% in-sample max drawdown further shows that the equal-weight, sequentially-compounded
return series is not an investable curve so much as an accounting artifact of a high-beta,
long-only sleeve. Distinguishing beta from alpha is the crux: the rule "makes money" only in
the trivial sense that long exposure to rising large-caps makes money, which is not what H₁
claimed and not a stock-selection edge. The SPY-benchmark and matched-window figures are
produced by `outputs/compute_benchmarks.py` from the same `trades_df`.

---

## 6. Robustness Validation

All thresholds applied in this section — IS/OOS composite gap < 0.05 (§6.1), Monte Carlo
p < 0.05 (§6.2), ≥ 6 of 8 walk-forward windows with positive composite (§6.3), and
parameter sensitivity degradation < 15% (§6.4) — were pre-registered in the project
proposal (§5.6 / §7.6) before any analysis was run, and are cited back to that document
rather than re-derived here.

### 6.1 IS/OOS Holdout Split

The in-sample period (2015-01-01 → 2025-06-30) produced composite score 0.5567.
The out-of-sample period (2025-07-01 → 2026-05-22) produced 0.8096.
The absolute gap of **0.253** **FAILS** the pre-specified threshold of 0.05.

The OOS composite is higher than the IS composite, so the gap is not in-sample overfitting;
it is driven by the very small OOS trade count (24 trades over ~10 months), which produces
high metric volatility. This same small-sample volatility is what produced the spurious H₂
result on the 22-trade High-VIX subset (§5.2) — a recurring caution throughout this study
that sub-100-trade samples cannot support confident inference.

### 6.2 Monte Carlo Permutation Test

The observed composite 0.557 placed at the bottom of the null distribution, yielding
p = 1.000 (all 500 permutations achieved an equal or higher score), so the pre-registered
MC criterion is not met. **This p-value is, however, a degenerate artifact of the composite
design rather than independent evidence of randomness**, and should not be read as "the
signal is statistically indistinguishable from random entry." As §7.5 derives, three of the
four composite components (Profit Factor, Sharpe, Win Rate) are invariant under shuffling
`net_return`, so the only component that varies is Max Drawdown; because the observed
drawdown (54.1%) already saturates that component to its floor (0), every permutation must
score ≥ the observed composite, forcing p = 1.000. The permutation test therefore neither
supports nor refutes a selection edge here — it is uninformative for this composite. The
substantive conclusion that no tradeable edge exists rests on §5.2 (the High-VIX advantage
disappears on the full signal universe), §5.3 (ML does not rely on the rule features), and
§5.5 (gross return is market beta; the full-universe win rate equals the base rate).

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

Parameter perturbation results, baseline composite **0.5567** (same baseline as §5.1):

| Parameter | Baseline | −20% | −10% | +10% | +20% | All Pass? |
|-----------|----------|------|------|------|------|-----------|
| rsi_threshold (22→18/20/24/26) | 0.5567 | 0.6979 | 0.6875 | 0.5935 | 0.5577 | ✅ Yes |
| min_consecutive (3→2/3/3/4) | 0.5567 | 0.7001 | 0.5567 | 0.5567 | 0.7134 | ✅ Yes |
| hold_days (6→5/5/7/7) | 0.5567 | 0.6406 | 0.6406 | 0.5808 | 0.5808 | ✅ Yes |

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
| Full 2026 (baseline) | 503 | 177 | 53.1% | 1.35 | 1.43 | 54.1% | 0.5567 |
| Frozen 2015 proxy | 453 | 171 | 55.0% | 1.44 | 1.71 | 46.6% | 0.6176 |

The frozen 2015 universe produced a **higher** composite score (+0.061, +11.0%) than the
full universe. This result is the **opposite** of the expected survivorship-bias inflation:
the 50 post-2015 additions (high-growth, high-volatility names) drag the strategy's
composite score down rather than inflating it. This is plausible because RSI < 22 oversold
signals in highly volatile post-IPO stocks tend to be false recoveries, not mean-reversion
opportunities driven by expiry hedging flows.

**Conclusion (appropriately scoped).** This check shows only that the **post-2015 additions**
present in the current cache do not inflate the result — if anything they drag it down. It does
**not** establish that survivorship bias is absent, because the "2015 proxy" is **not** a
point-in-time S&P 500 membership list: it is built from the *current* cache and therefore still
omits the very companies that survivorship bias is about — constituents that were removed or
delisted between 2015 and 2026 (the failures). Those names leave no CSV in the cache and so
cannot enter either universe. A rigorous treatment requires a true point-in-time constituent
history with delisted tickers (§8.3); the present check should be read as "current additions do
not flatter the result," not as "returns are free of survivorship bias." Walk-forward evaluation
partially mitigates the concern, since OOS returns are earned after each training window closes.

### 7.2 Take-Profit Calibration — Recalibrated with RSIReversalStrategy

The original `take_profit_research.ipynb` used **EMAScanner** — an exploratory entry
generator from an earlier project iteration that triggers on EMA crossover patterns,
predating the capstone's RSI-based signal rule and unrelated to the pre-registered
hypotheses tested in this report — to produce entry signals. The resulting exit
parameters could not legitimately be applied to `RSIReversalStrategy`, since exit
calibration is sensitive to the entry distribution. The notebook has been fully
recalibrated for this capstone using RSIReversalStrategy entries from the IS period
(807 raw RSIReversalStrategy signals in this exit-grid scan, 500 randomly sampled for the
grid search, IS window 2015–2025). This 807 is a separate raw-signal count for the
take-profit grid and is not the same population as the §3.6 ML feature matrix (801 rows
across 2015–2026, of which 718 fall in the in-sample window after dropping rows with any
missing feature).

**Grid search results across six exit strategies:**

| Exit Strategy | Avg Return | Win Rate | Sharpe | Profit Factor | Avg Hold Days |
|---------------|-----------|----------|--------|---------------|---------------|
| rsi_exit (RSI > 75) | **2.73%** | 51.4% | 0.230 | **1.90** | 25.9 |
| rsi_exit (RSI > 65) | 2.45% | **59.0%** | 0.241 | **1.99** | 21.0 |
| time_based (20-day cap) | 2.27% | 57.4% | 0.255 | 2.08 | 16.5 |
| fixed_pct (tp=5%) | 1.50% | **63.0%** | 0.242 | 1.65 | 15.1 |
| trailing_stop (10%) | 2.27% | 44.4% | 0.219 | 1.75 | 35.2 |
| fixed_pct (tp=15%) | 2.41% | 44.6% | 0.236 | 1.71 | 31.3 |

**Note on the time-based row.** `TakeProfitBacktester` defines `time_based` with a default
`hold_days = 20` (and a 60-day max-hold backstop), so its average realized hold is 16.5 days —
*not* the capstone's frozen `HOLD_DAYS = 6`. This exploratory grid therefore compares exit
*styles* against a 20-day time-based exit as its internal baseline; it is **not** a direct
6-day benchmark. (An earlier draft mislabeled this row "6 days, baseline" and left its Sharpe
and Profit Factor blank; both are corrected here from the notebook output: Sharpe 0.255,
Profit Factor 2.08.) A clean comparison against the capstone's true 6-day hold would require
re-running `time_based` with `hold_days = 6`, which is noted as a follow-up.

**Recommended exit upgrade:** `rsi_exit` with threshold = 65–70 delivers the best
win-rate-adjusted performance: 59.0% win rate at 2.45% average return (vs. 57.4% /
2.27% for the 20-day time-based exit). This exit type is supported by `run_simulation(exit_params=
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

### 7.5 Degeneracy of the Monte Carlo Permutation Test

The Monte Carlo test as implemented (§3.5) shuffles the `net_return` column of the selected
trades. This has two consequences that limit what the resulting p-value can support, and both
are material to how H₁ should be read.

**(1) Three of the four composite components are permutation-invariant.** Profit Factor, the
per-trade Sharpe ratio (mean/std of returns), and Win Rate depend only on the *set* of returns,
not their order; shuffling leaves them unchanged. Only Max Drawdown — computed on the
sequentially compounded equity curve — varies across permutations. The permutation test is
therefore, in effect, a test of the return *ordering* (drawdown) alone, and carries no
discriminating power over the other three-quarters of the composite.

**(2) The p-value is forced to 1.000 whenever observed drawdown exceeds the 30% cap.** The
composite's drawdown term is `(1 − min(|MDD|/0.30, 1)) × 0.15`, which saturates to 0 for any
drawdown ≥ 30%. The observed in-sample drawdown is 54.1%, so the observed composite already
takes the minimum possible drawdown sub-score (0). Every permutation's drawdown sub-score is
≥ 0, and the other three components are identical, so **every** permuted composite is ≥ the
observed composite — making p = 1.000 a near-mathematical certainty rather than a measurement
of randomness. Concretely, the observed composite decomposes as PF 0.236 + Sharpe 0.215 +
Win Rate 0.106 + Drawdown 0.000 = 0.557, and no permutation can score below 0.557.

**Implication.** The p = 1.000 in §5.1/§6.2 should not be cited as evidence that "the signal
is no better than random entry." It is an artifact of (i) saturating the drawdown sub-score and
(ii) the permutation null only varying that one sub-score. A correct test of selection edge
resamples the *entries themselves* — drawing random (ticker, expiry-date) pairs on the
strategy's actual entry dates and re-running the portfolio, so that Win Rate, Profit Factor,
and Sharpe also vary under the null.

**We implemented this portfolio-level random-entry Monte Carlo** (1,000 iterations, seed = 42,
same entry dates and per-date trade counts as the rule, same hold and cost; re-runnable via
`outputs/compute_benchmarks.py`). The rule's composite (0.557) sits at the **62nd percentile**
of the random-entry null (null median 0.494), **p = 0.378** — modestly above the random median
but **not statistically significant**. So the
proper test, unlike the degenerate shuffle-MC, *can* discriminate, and its verdict is the same
in substance: **no demonstrated stock-selection skill** (the rule does not beat random picking
at any conventional significance level). This same procedure is the core of the public
**Edge or Beta?** tool (`doc/design/edge_or_beta_tool_design.md`), which races any rule against
random stock picks, random ETF timing, and buy-and-hold SPY. The no-edge conclusion is thus
corroborated by this proper random-entry test (p = 0.378) and, independently, by the
large-sample H₂ result (§5.2), the ML feature analysis (§5.3), and the beta-vs-alpha
decomposition (§5.5).

---

## 8. Practical Implications

### 8.1 No Deployable Edge — the Honest Practical Takeaway

The most important practical implication is a negative one, and it should be stated plainly:
**this rule is not a deployable trading edge in the S&P 500 universe.** All three hypotheses
were not supported — the primary H₁ permutation gate fails and is degenerate rather than a
valid random-entry test, the apparent High-VIX advantage is a small-sample artifact that
vanishes on the full signal universe (H₂, p = 0.474), and ML attributes predictive structure
to volatility/position features rather than the rule (H₃). What positive return the backtest
shows is market beta from long exposure to a rising market, not stock-selection alpha (§5.5).

For a practitioner, the actionable conclusion is therefore: **do not trade this rule as a
source of alpha.** An investor seeking the return the backtest displays should simply hold a
low-cost broad-market index fund, which delivers the same beta with lower turnover, lower
transaction costs, and a far smaller drawdown than the 54% equal-weight in-sample figure here.
The earlier draft of this report recommended a "VIX-conditional filter"; that recommendation
is **withdrawn**, because the H₂ result it rested on does not survive an adequately powered test.

### 8.2 Why a Real Mechanism Need Not Yield a Tradeable Rule

The expiry hedging-pressure mechanism (§4.1) is real and well documented — but at the **index
and option level**, and as a property of *derivative payoffs* rather than of a retail-executable
long position in individual stocks. Three reasons explain why a genuine mechanism does not
translate into a profitable individual-stock rule here:

1. **Level and instrument mismatch.** Baltussen et al. (2025) and Golez & Jackwerth (2012)
   identify the effect in index/futures/option prices; capturing it generally requires
   options or index instruments, not a long-only equity screen.
2. **Magnitude vs. cost.** Even where present in individual names, the distortion is small
   relative to the 0.2% round-trip cost and the idiosyncratic noise of single stocks; it is
   not separable from that noise by an RSI/consecutive-candle proxy.
3. **Arbitrage and decay.** Large-cap U.S. equities are among the most heavily arbitraged
   assets; documented anomalies decay after publication, particularly in liquid names
   (McLean & Pontiff, 2016; Sullivan et al., 1999; Fama, 1991). A simple, widely known oversold
   rule is exactly the kind of signal that competition removes.

The data show the mechanism's footprint without its tradeability. Splitting the forward 6-day
net return of every RSI < 22 & ≥ 3-red signal by expiry timing, signals on the day **before**
a monthly expiry returned **+1.55%** on average versus **+0.39%** on all other days (a +1.16%
gap; 66.9% vs. 55.5% win rate). Taken naively across 14,508 signals the gap is highly
significant, but the 815 pre-expiry signals cluster within only **84 independent expiry days**;
treating each expiry day as one observation, the difference is **not significant** (clustered
*t* = 1.67, p = 0.10). So a real, directionally-correct expiry footprint exists in the raw
returns — consistent with the documented hedging mechanism — yet it is statistically fragile
and, once funnelled through top-3 selection and transaction costs, produces no edge over SPY
(§5.5) or random entry (§7.5). **Mechanism existence ≠ ex-ante tradeability.**

This is the central interpretive contribution: distinguishing **mechanism existence** (true)
from **ex-ante tradeability at the stock level, net of costs** (not supported here).

### 8.3 Limitations for Live Deployment

This study evaluates a backtested rule on historical data and finds no deployable edge.
The constraints below are recorded for methodological completeness — they are reasons the
backtest, even where positive, cannot be read as a live-tradeable result, and they would
have to be resolved before *any* future variant of this work could be considered for capital:
- Apply the strategy to a point-in-time constituent list to remove survivorship bias
- Validate performance in the most recent 12–18 months of live paper trading
- Assess liquidity: the strategy targets S&P 500 large-caps, which are generally liquid,
  but individual names at extreme oversold readings may have wider bid-ask spreads
- Monitor for regime changes that could invalidate the underlying microstructure mechanism
  (e.g., regulatory changes to options market structure, or significant shifts in SPY
  options open interest distribution)

### 8.4 Broader Implications for Quantitative Research

The study demonstrates a methodology for bridging options microstructure theory
with cross-sectional technical analysis research. The walk-forward framework provides a
useful template for evaluating event-conditioned technical rules under temporal discipline.
The Monte Carlo component, however, should not be reused in its present return-shuffle form
as a selection-edge test; §7.5 shows that it is degenerate when drawdown saturation makes the
composite score order-insensitive. We replaced it with portfolio-level random-entry resampling
(§7.5: rule at the 62nd percentile, p = 0.378) so that Profit Factor, Sharpe, Win Rate, and Max
Drawdown all vary under the null — the form future templates (and the Edge or Beta? tool) should
use. The composite scoring approach — combining Profit Factor, Sharpe, Win Rate,
and Max Drawdown into a single bounded metric — remains useful for comparing strategies that
may trade off different aspects of risk-adjusted performance, provided the null test varies
the underlying entries rather than merely reordering realized returns.

---

## 9. References

Baltussen, G., van Dijk, M., & Zhu, L. (2025). The third Friday price spike. *Working Paper*.

Brock, W., Lakonishok, J., & LeBaron, B. (1992). Simple technical trading rules and the
stochastic properties of stock returns. *Journal of Finance, 47*(5), 1731–1764.

Fama, E. F. (1991). Efficient capital markets: II. *Journal of Finance, 46*(5), 1575–1617.

Golez, B., & Jackwerth, J. C. (2012). Pinning in the S&P 500 futures. *Journal of
Financial Economics, 106*(3), 566–585.

Kim, M. J., Nelson, C. R., & Startz, R. (1991). Mean reversion in stock prices? A reappraisal
of the empirical evidence. *Review of Economic Studies, 58*(3), 515–528.

McLean, R. D., & Pontiff, J. (2016). Does academic research destroy stock return
predictability? *Journal of Finance, 71*(1), 5–32.

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
                                   #   §3b: large-sample H₂ re-test, continuous-VIX,
                                   #   small-vs-large-sample comparison chart
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
