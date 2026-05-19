# Capstone Project Purpose

**Program**: Master of Data Analytics
**Course**: DAMO-699-3 — Spring 2026 Capstone Project
**Institution**: University of Niagara Falls Canada
**Project Title**: SPY Options Expiry Day Pattern Analysis — Predicting S&P 500 Stock Returns Using Technical Patterns and Machine Learning
**Last Updated**: 2026-04-18 (rubric-aligned revision)

### Team Members

| Name | Student ID | Email |
|------|-----------|-------|
| Bicieg Vazquez Del Mercado, Diego | NF100XXX | Diego.Bicieg@myunfc.ca |
| Dos Santos Rabelo, Ícaro | NF1009242 | icaro.dos9242@myunfc.ca |
| Chundru, Sai Bhaskar | NF1010427 | sai.chundru0427@myunfc.ca |
| Xin, Osmond (Yi) | NF1007319 | yi.xin7319@myunfc.ca |

---

## Abstract

This capstone asks a single, bounded question: does a pre-specified rule combining RSI oversold conditions and consecutive bearish candles generate a statistically significant and temporally stable positive-return edge in S&P 500 individual stocks on SPY monthly options expiry days (the third Friday of each month)? Despite extensive research on mean reversion and on options-expiry microstructure, little empirical work has tested whether expiry-driven hedging flows create exploitable stock-level mean-reversion opportunities identifiable from observable technical conditions — and whether VIX regime moderates this effect.

Using daily OHLCV data for S&P 500 constituents over 2015–2026, enriched with VIX, FOMC, and earnings calendars, a rule-based strategy is evaluated via a realistic T+1 portfolio simulation with 0.2% commission, a maximum of three entries per expiry day, and at most fifteen concurrent positions. Statistical validity is assessed through 500-iteration Monte Carlo permutation testing, 8-window walk-forward analysis, and parameter-sensitivity perturbation. Supervised machine learning models (Logistic Regression, Random Forest, XGBoost) independently verify whether RSI and consecutive-candle features drive the predictive structure. Three pre-registered hypotheses test edge significance, VIX-regime moderation, and ML feature dominance.

*Keywords:* mean reversion, options expiry, RSI, VIX regime, machine learning, walk-forward validation

---

## I. Problem Statement

Despite extensive research on mean reversion in equity markets (Poterba & Summers, 1988; Kim et al., 1991) and on options expiry microstructure effects (Golez & Jackwerth, 2012; Baltussen et al., 2025), these two bodies of literature have largely developed in isolation. Mean-reversion studies typically model price behaviour as a statistical property of returns over time, without incorporating event-driven mechanical catalysts. In contrast, options microstructure research documents systematic price distortions around expiry dates but focuses primarily on index-level dynamics, rather than cross-sectional predictability at the individual stock level.

Furthermore, while option-implied measures — particularly the implied volatility spread — have been shown to embed forward-looking information about expected volatility and information flow (DeLisle et al., 2021; Chen & Li, 2023), limited research has incorporated volatility regimes as conditioning variables for technical trading signals within an expiry-based framework.

As a result, there is limited empirical evidence on whether options market microstructure events systematically generate exploitable, stock-level mean-reversion opportunities, identifiable through observable technical conditions prior to expiry, and whether the effectiveness of such signals is moderated by the prevailing volatility environment.

This gap has direct real-world consequences. Quantitative hedge funds, systematic portfolio managers, execution desks, and sophisticated retail investors must routinely decide whether to enter, exit, or defer trades in the days surrounding monthly expiry; without evidence on whether technical signals are genuinely predictive — or merely reflect random noise — such decisions rely on heuristics that may embed economically significant bias. Closing this gap enables evidence-based timing rules for short-term portfolio rebalancing, informs risk overlays around expiry weeks, and clarifies when institutional hedging flows create exploitable inefficiencies versus when they do not (Baltussen et al., 2025).

---

## II. Solution Statement

This project addresses this gap by integrating options expiry microstructure with mean reversion theory in a unified, testable analytical framework at the individual stock level. Using S&P 500 constituents over 2015–2026, the project operationalises the intersection of options market hedging pressure associated with option time decay ("charm") and technical oversold conditions (RSI + consecutive bearish candles) as a signal for short-term mean reversion around monthly expiry dates.

VIX regime is introduced as an explicit moderating variable, motivated by Kim et al.'s (1991) finding that mean reversion concentrates in high-volatility periods. Machine-learning models (Logistic Regression, Random Forest, XGBoost) are applied to validate whether the hand-crafted rule approximates the statistically optimal decision boundary, bridging interpretable finance theory with data-driven model validation. The framework is evaluated with rigorous walk-forward validation and Monte Carlo permutation testing to distinguish genuine edge from noise.

**Academic contribution**: The study's contribution is targeted: it tests whether a single, interpretable technical rule captures the stock-level mean-reversion effect of options-expiry hedging flows, and whether VIX regime systematically conditions that effect. This is a feasible, well-scoped capstone question that is directly answerable within the available 2015–2026 sample.

---

## III. Analytical Objective

**Primary objective:** Quantify whether a pre-specified technical signal — RSI oversold combined with a sustained sequence of consecutive bearish candles — generates a statistically significant and temporally stable positive-return edge in S&P 500 individual stocks on SPY monthly options expiry days (the third Friday of each month), evaluated against a random-entry null via Monte Carlo permutation testing and assessed for temporal stability across walk-forward out-of-sample windows.

**Secondary objective:** Use supervised machine learning models (Logistic Regression, Random Forest, XGBoost) trained on the same feature space to (a) independently verify that RSI and consecutive-candle features dominate the predictive structure, and (b) quantify whether model-derived signals improve portfolio-level Sharpe, Win Rate, and Profit Factor beyond the rule-based baseline.

**Moderator analysis:** Test whether the implied-volatility environment (VIX regime) systematically conditions signal effectiveness, in line with Kim et al.'s (1991) evidence that mean reversion concentrates in high-volatility periods.

The investigation follows the full data-analytics lifecycle — data acquisition and enrichment, feature engineering, predictive modelling (rule-based and ML), portfolio simulation, robustness validation, and communication — as elaborated in Section VII.

---

## IV. Research Questions

**Primary research question:**
> Do RSI oversold conditions combined with consecutive bearish candles, observed in S&P 500 stocks in the days preceding SPY monthly options expiry, produce a positive-return edge that is statistically distinguishable from random entry and temporally stable across out-of-sample windows?

**Secondary research questions:**
- RQ2 — Does the implied-volatility environment (VIX regime: Low < 15, Medium 15–25, High > 25) moderate signal effectiveness?
- RQ3 — Do supervised ML models (Logistic Regression, Random Forest, XGBoost) independently assign dominant feature importance to RSI and Consecutive_Count?

---

## V. Hypotheses

All hypotheses are evaluated at a significance level of α = 0.05. Each hypothesis is paired with a pre-specified statistical test:

**H₀ (Null)**: Portfolio returns generated by the RSI-oversold + consecutive-red-candle signal on SPY monthly expiry days are indistinguishable from a random-entry strategy on the same dates.
*Test:* Monte Carlo permutation test over 500 random signal-date shuffles; composite-score percentile of the observed strategy against the null distribution.

**H₁ (Primary alternative)**: The signal produces a statistically significant positive edge that is temporally stable across out-of-sample periods. Significance is evaluated via Monte Carlo permutation testing; temporal stability is assessed across walk-forward out-of-sample windows. Specific evaluation thresholds are defined in Section 7.6.

**H₂ (VIX moderator)**: Signal win rate is materially higher in high-VIX environments (VIX > 25) than in low-VIX environments (VIX < 15).
*Test:* Two-proportion z-test comparing win-rate in the two VIX sub-samples; one-sided alternative (high > low); α = 0.05.

**H₃ (ML feature validation)**: Supervised models trained on the full feature matrix assign dominant importance to RSI and Consecutive_Count.
*Test:* Random Forest permutation importance and XGBoost SHAP mean-absolute-value ranking; H₃ supported if both features appear in the top-3 most important features for at least two of the three models.

---

## VI. Industry and Data Context

**Industry/Sector**: U.S. Equity Markets — S&P 500 large-cap constituents (~500 stocks)

**Analytical Context**: Options expiry dates produce abnormal volume and price activity due to the mechanics of options pinning, delta hedging, and gamma unwinding. Market makers who have sold options must dynamically rebalance their delta hedges; as options approach zero days-to-expiry (0DTE), the time decay of these hedges (charm) creates systematic buy-side pressure in individual stocks that were pushed below fair value by prior selling flows. This creates a predictable mean-reversion window around each monthly expiry date.

### Data Sources

| Source | Dataset | Format | Coverage | Access |
|--------|---------|--------|----------|--------|
| Stooq.com | S&P 500 OHLCV (price + volume) | Per-ticker CSV | 2010–present, ~500 tickers | Free, no API key |
| Alpaca Markets API | Incremental OHLCV updates | REST API | Daily updates | Free tier, API key |
| FRED (St. Louis Fed) | VIX daily close (`VIXCLS`) | `cache/vix.csv` (9,468 rows) | 1990–2026 | Free, no API key |
| Federal Reserve | FOMC meeting dates 2015–2026 | `cache/fomc_dates.csv` (97 rows) | 2015–2026 | Public calendar |
| Nasdaq Earnings Calendar API | Earnings announcement dates | `cache/earnings_dates.csv` (21,326 events) | 2015–present, 500 tickers | Free, no auth |
| S&P 500 constituent list | Ticker universe | `cache/sp500_list.csv` | Current ~500 symbols | Cached |
| Derived | Monthly expiry dates (3rd Friday) | `cache/expiry_dates.csv` | 2010–2026 | ExpiryCalculator module |

**Data acquisition tool**: `tools/download_enrichment_data.py` — fetches VIX, FOMC, and earnings data in two phases (Phase 1: ~5 sec; Phase 2: ~25 min, resumable).

### Data Limitations and Mitigations

The project acknowledges four categories of data-quality risk. Each is documented transparently and addressed through explicit mitigation steps:

| Source | Limitation | Mitigation |
|--------|-----------|-----------|
| S&P 500 constituent list | **Survivorship bias** — the universe uses the current (2026) S&P 500 membership list rather than point-in-time historical constituents, potentially inflating returns by excluding delisted/removed names | Reported as a known bias in the final findings; walk-forward OOS design limits in-sample optimisation artefacts; sensitivity analysis compares results on a constant-universe subset |
| Stooq OHLCV | Free data without a formal service-level guarantee; occasional missing bars or split/dividend adjustment errors possible | `DataValidator` module flags and logs gaps, duplicate dates, and non-monotonic prices; incremental Alpaca API updates cross-verify recent closes |
| Alpaca API | Free-tier rate limits; potential update gaps during market holidays or provider outages | `CacheManager` supports resumable incremental updates and fall-back to Stooq for missing sessions |
| Nasdaq Earnings Calendar | Third-party accuracy not contractually guaranteed; earnings date reschedules may not always be reflected | Earnings-proximity filter is applied as a sensitivity layer (Section 7.6), not as a hard filter, so missed events degrade signal-to-noise but do not cause look-ahead bias |
| FRED VIX | Official end-of-day closing value; no intraday granularity | Acceptable for regime classification; intraday VIX dynamics are outside the research scope |
| FOMC calendar | Only 97 events across 2015–2026 | Sufficient for binary proximity flagging; sample too small for standalone FOMC-conditional subgroup analysis, which is excluded a priori |

**Survivorship bias deserves particular attention.** The universe uses the current (2026) S&P 500 membership list rather than point-in-time historical constituents. This may inflate returns by excluding stocks that were removed or delisted during the study period. Two mitigations are applied: (1) the walk-forward out-of-sample windows use forward returns earned after the model is trained, which limits in-sample selection artefacts; and (2) a sensitivity check using a frozen 2015 constituent snapshot is planned as an optional extension to bound the magnitude of the bias. This limitation is reported transparently in the final findings.

These limitations are revisited in Section XI and in the robustness suite (Section 7.6).

### Sample Size and Statistical Power

An *ex ante* sample-size estimate supports the feasibility of the planned statistical tests:

- **Backtest window**: 11 years × 12 monthly expiries ≈ **132 expiry events**
- **Raw feature rows**: 132 expiries × ~500 tickers = **~66,000 (ticker, expiry) observations**
- **Rule-based signals (RSI < 30 with ≥ 3 consecutive red candles)**: expected at ~3–5% of observations → **≈ 2,000–3,300 signals** for ML training
- **Portfolio trades (top-3 ranked per expiry)**: 132 × 3 = **≈ 396 trades**

Under these sample sizes, a two-proportion z-test (H₂) can detect a win-rate gap of ≥ 10 percentage points between VIX regimes with ≥ 0.80 power at α = 0.05 if each regime contains ≥ 150 trades. Walk-forward windows (~50 trades each) are sufficient for stable composite estimates; Monte Carlo permutation (500 shuffles) provides a resolution of p ≈ 0.002, well below the 0.05 threshold.

---

## VII. Analytical Methods

The analytical workflow is structured to align explicitly with the standard data-analytics lifecycle. Each subsection below corresponds to one lifecycle stage:

| Lifecycle Stage | Project Section |
|-----------------|-----------------|
| 1. Data Acquisition & Enrichment | Section VI — Industry and Data Context |
| 2. Feature Engineering | Section 7.1 |
| 3. Predictive Modelling (Rule-based) | Section 7.2 |
| 4. Predictive Modelling (Machine Learning) | Section 7.3 |
| 5. Portfolio Simulation & Backtesting | Section 7.4 |
| 6. Model Evaluation | Section 7.5 |
| 7. Robustness Validation & Diagnostics | Section 7.6 |
| 8. Visualisation & Communication | Section VIII |

### 7.1 Feature Engineering

All features are computed from data available at signal date only (no look-ahead). The full feature matrix covers every (stock, expiry_date) pair in the universe.

| Feature | Definition | Justification |
|---------|------------|---------------|
| `RSI(14)` | Relative Strength Index, 14-period | Industry-standard momentum oscillator (Wilder, 1978); values below the threshold indicate oversold conditions |
| `Consecutive_Count` | Count of successive days where Close < Open | Captures duration of selling pressure; exhaustion patterns require sustained directional moves |
| `MA_Distance` | % deviation of price below a short-period SMA | Measures mean-reversion potential (Poterba & Summers, 1988) |
| `VIX_Level` / `VIX_Regime` | CBOE VIX close; categorical Low/Medium/High | Measures market fear; high VIX amplifies mean-reversion tendency (Kim et al., 1991); see H₂ |
| `FOMC_Proximity` | Binary: FOMC meeting within ±5 calendar days | Controls for macro uncertainty that can override technical patterns |
| `Earnings_Proximity` | Binary: earnings release within ±3 days of signal | Identifies fundamental-driven moves orthogonal to technical mean-reversion |

Additional features available for ML training — Bollinger Band position, MACD histogram, and volume ratio — are engineered but not part of the core signal rule. They are included in the ML feature matrix to test whether the rule-based signal approximates a richer predictive structure.

### 7.2 Rule-Based Predictive Model

The primary strategy (`RSIReversalStrategy`) applies a transparent signal rule with thresholds determined through systematic parameter search. A stock receives a signal of 1 when its 14-period RSI falls below a pre-specified oversold threshold AND its consecutive bearish candle count meets a minimum. Both AND and OR logic are evaluated; the better-generalising form is selected based on out-of-sample performance. An upper bound on consecutive candles is tested to exclude free-falling stocks where recovery is less reliable. Signals are ranked by ascending RSI; the top N per expiry day enter the portfolio.

**Justification**: The rule directly operationalises the research question. Its transparency makes it interpretable for academic review and directly testable against a random-entry null via Monte Carlo permutation.

### 7.3 Machine Learning Models

ML models are applied to the **full unfiltered signal universe** (all stocks satisfying RSI < threshold on expiry windows, not just the top-3 portfolio selection). This produces a larger labelled dataset suitable for supervised learning.

**Task formulation**:
- Input X: feature vector at signal date (all features in 7.1)
- Output y: binary label — is the return over the hold period positive? (1 = win, 0 = loss)

| Model | Role | Justification |
|-------|------|---------------|
| **Logistic Regression** | Interpretable baseline | Coefficients directly show each feature's directional contribution; most comparable to the rule-based threshold logic |
| **Random Forest** | Feature importance analysis | Captures non-linear interactions; built-in importance scores answer which technical indicator is most predictive |
| **XGBoost** | Performance benchmark | Typically best on tabular financial data; SHAP values provide per-prediction explanations |

**Models excluded**:
- Neural networks / LSTM: sample count (estimated few thousand rows) is insufficient
- K-Means: task is prediction, not segmentation
- ARIMA / Prophet: cross-sectional prediction task, not single time-series forecasting

**Validation discipline for ML**: All ML models use walk-forward time-ordered splits identical to the rule-based validation. No random train/test split — financial time series requires temporal discipline to prevent look-ahead bias.

**Scope note**: ML validation is a secondary, confirmatory analysis. Feature importance results from a single Random Forest model are sufficient to evaluate H₃; XGBoost with SHAP values is an optional extension if the project timeline permits.

### 7.4 Portfolio Simulation (Backtesting)

The rule-based strategy is evaluated through a realistic portfolio simulation. Entry is at the T+1 open (signal generated at close, executed next morning — no look-ahead bias). Positions are exited after a fixed hold period or on stop-loss / take-profit triggers determined via parameter search. Constraints include a maximum of three new entries per expiry day (ranked by lowest RSI), at most fifteen concurrent positions, equal-weight sizing, and a 0.2% per-trade commission; no leverage or short selling is used.

### 7.5 Model Evaluation

**Rule-based strategy** — composite score combining four metrics with pre-registered weights:

| Metric | Weight | Transformation | Rationale |
|--------|--------|----------------|-----------|
| Profit Factor | 0.35 | `min(PF / 2.0, 1.0)` | Gross profit / gross loss — primary measure of edge quality |
| Sharpe Ratio | 0.30 | `min(max(Sharpe, 0) / 2.0, 1.0)` | Risk-adjusted return |
| Win Rate | 0.20 | `Win_Rate` (already 0–1) | Signal reliability |
| Max Drawdown | 0.15 | `1 − min(\|MaxDD\| / 0.30, 1.0)` | Downside risk control (inverted — smaller drawdowns score higher) |

All four sub-scores are bounded to [0, 1] and combined as a weighted sum, producing a composite in [0, 1]. **Weights were fixed before any optimisation began and held constant across every experiment** to prevent ex post metric selection (a form of p-hacking). The weight vector (0.35 / 0.30 / 0.20 / 0.15) reflects the study's emphasis on edge quality (PF) and risk-adjusted return (Sharpe) over raw hit rate or drawdown.

**ML models** — standard classification metrics plus financial performance:

- Accuracy, Precision, Recall, F1-score, AUC-ROC
- Portfolio Sharpe Ratio, Win Rate, Profit Factor (using ML signals as entry triggers)
- Feature importance / SHAP summary

### 7.6 Robustness Validation and Diagnostics

| Test | Description | Threshold | Priority |
|------|-------------|-----------|----------|
| IS / OOS split | Train 2015–2025, test 2025 H2–2026 | OOS gap < 0.05 | Essential |
| Walk-Forward Analysis | 8 rolling windows (3yr IS / 1yr OOS, 2015–2026) | Mean OOS composite > 0.45 | Essential |
| Monte Carlo Permutation | 500 random signal shuffles; compare strategy composite to null distribution | p < 0.05 | Essential |
| Parameter Sensitivity | ±10% and ±20% perturbation of each parameter individually | Composite degrades < 15% | Essential |
| Market Regime Segmentation | Bull / Sideways / Bear classification by SPY 200-day SMA; VIX regime (Low / Medium / High) | Positive Profit Factor in ≥ 2 of 3 regimes | Optional |
| Earnings Filter Impact | Compare win rate with and without earnings proximity filter | Improvement in Win Rate and Profit Factor | Optional |

**Justification for walk-forward over simple train-test split**: Financial time series exhibit regime changes and non-stationarity. Walk-forward analysis simulates realistic live deployment — the model is repeatedly retrained on expanding windows — preventing look-ahead bias and measuring temporal stability across different market environments.

### 7.7 Ethics, Data Governance, and Reproducibility

- **Data ethics**: All data sources are public-domain market data (exchange-traded prices, publicly released macro calendars, and published VIX values). No personally identifiable information, proprietary dataset, or non-public material is used.
- **Regulatory scope**: The project is academic in nature and does not constitute investment advice; no live capital is deployed.
- **Reproducibility**: All analysis code is version-controlled in Git; the Python environment is pinned via `requirements.txt`; random seeds are fixed (seed = 42) for every Monte Carlo, train/test split, and Random Forest / XGBoost run; cache files are content-hashed to detect silent data drift.
- **Computational environment**: Python 3.11, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`. Full reproduction instructions are documented in `CLAUDE.md` and `README.md`.
- **Pre-registration discipline**: Hypotheses (Section V), composite-score weights (Section 7.5), robustness thresholds (Section 7.6), and the feature set (Section 7.1) are fixed in this proposal before evaluation begins; deviations from the pre-registered plan will be explicitly reported in the final write-up.

---

## VIII. Data Visualisation and Communication

| Visualisation | Purpose |
|---------------|---------|
| Equity curve vs. SPY buy-and-hold | Primary performance communication |
| RSI distribution at signal dates | Shows the statistical properties of the entry condition |
| Consecutive candle count distribution | Validates the ≥ 3 threshold choice |
| Walk-forward window heatmap | Demonstrates temporal stability across 8 windows |
| Monte Carlo permutation histogram | Shows strategy result vs. null distribution with p-value |
| Win rate by VIX regime (bar chart) | Tests H₂; visualises the volatility moderator hypothesis |
| Feature importance bar chart (Random Forest) | Core ML output — validates rule-based feature selection |
| SHAP summary plot (XGBoost) | Per-feature directional contribution to predictions |
| Rule-based vs. ML model comparison table | Answers whether ML improves on the rule |

All visualisations are produced in `notebooks/spy_expiry_analysis_v2.ipynb` and `notebooks/pattern_distribution_scanner.ipynb`.

---

## IX. Literature Review

### 9.1 Options Expiry Microstructure and Price Distortions

Price behaviour around options expiry has been studied extensively within the market-microstructure literature, primarily at the aggregate index level. Golez and Jackwerth (2012) provide a foundational analysis of pinning effects, demonstrating that S&P 500 futures prices systematically gravitate toward or deviate from at-the-money strike prices on expiration dates. These dynamics are driven by the interaction between the options market and the underlying asset, particularly through the rebalancing of delta hedges and investor trading behaviour, which together generate systematic price pressure.

More recent evidence from Baltussen et al. (2025) documents a persistent "third Friday price spike" in U.S. equity index markets: prices systematically increase in the period leading up to options expiry and subsequently revert, creating economically significant distortions in derivative payoffs. The authors attribute this effect to dealer hedging behaviour driven by option time decay (charm), rather than to traditional delta-hedge unwinding.

Complementing these index-level findings, Ni, Pearson, and Poteshman (2005) document a related phenomenon at the individual-stock level: stock prices cluster around option strike prices on expiration dates, consistent with market-maker delta-hedging activity creating systematic price pressure in individual names. While Golez and Jackwerth (2012) and Baltussen et al. (2025) establish the aggregate dynamics, Ni et al. (2005) confirm these effects extend to individual stocks, directly motivating the present study's cross-sectional approach.

However, none of these studies examines whether the individual-stock expiry effect can be identified ex ante from observable technical conditions, nor whether its magnitude depends on the prevailing volatility environment. These gaps define the specific empirical question this capstone addresses.

### 9.2 Mean Reversion, Volatility Regimes, and Implied Volatility as a Moderator

The expectation of positive returns following extreme price declines is grounded in the mean-reversion literature. Poterba and Summers (1988) provide early empirical evidence that stock prices contain substantial transitory components, implying that deviations from fundamental value tend to correct over time; their findings indicate negative autocorrelation in long-horizon returns, consistent with mean-reverting behaviour in equity markets.

Kim et al. (1991) qualify this finding, showing that mean reversion is not a universal phenomenon but rather depends on the underlying market environment: reversion is more pronounced during periods of elevated volatility and weaker in calmer regimes. This insight motivates the incorporation of volatility regimes as an explicit moderating variable in the present study.

The options literature complements this perspective by demonstrating that implied volatility contains forward-looking information relevant to asset pricing. DeLisle et al. (2021) show that variation in the implied-volatility spread — capturing deviations between call- and put-implied volatilities — has significant predictive power for future stock returns, particularly when this spread exhibits higher variability. Chen and Li (2023) further establish that approximately one third of implied volatility's predictive power for realised volatility stems from its ability to anticipate news-arrival intensity, reinforcing the interpretation that elevated VIX before expiry implies larger subsequent realised moves and amplifies the mean-reversion window for deeply oversold stocks.

At the individual-stock level, Jegadeesh and Titman (1993) document that short-horizon return reversals coexist with intermediate-horizon momentum in equities: stocks that underperform over the prior month tend to recover over the following month. This short-horizon reversal is directly consistent with the 9-day mean-reversion window studied here, and distinguishes the present analysis from momentum strategies — the mechanism is exhaustion recovery, not trend continuation. Unlike Jegadeesh and Titman (1993), however, who focus on calendar-period sorting, the present study conditions entry on a specific microstructure event (monthly options expiry), which provides an explicit mechanical catalyst for the expected mean reversion.

Taken together, this literature suggests that mean reversion at the individual-stock level is both conditional and event-driven. Existing studies do not integrate these insights within an expiry-based framework, nor do they examine whether volatility regimes systematically moderate the effectiveness of technical mean-reversion signals around options-expiry events. The present study addresses both gaps.

### 9.3 Technical Signal Validity

The academic standing of technical analysis as a predictive tool remains contested. Brock, Lakonishok, and LeBaron (1992) provide one of the most cited empirical tests of technical trading rules, finding that simple moving-average and trading-range breakout rules generated statistically significant returns in U.S. equity markets over 1897–1986. However, subsequent work has questioned whether such results survive transaction costs, data snooping corrections, and out-of-sample periods — consistent with gradually increasing market efficiency.

The present study engages with this debate in two ways. First, rather than asserting that technical signals are generically predictive, it tests a single pre-specified rule against a Monte Carlo permutation null, which controls for the data-mining bias that afflicts exploratory technical analysis research. Second, the rule is theoretically grounded in an event-driven mechanism (options-expiry hedging flows), rather than applied as a generic price pattern, which provides a falsifiable structural rationale that is absent from purely empirical rule-testing.

---

## X. Objectives, Milestones, and Timeline

The project is scheduled across the Spring–Summer 2026 capstone term. Stages 1–2 are substantially complete as of the proposal submission; remaining milestones are scheduled as follows:

| # | Milestone | Deliverable | Target Window | Status |
|---|-----------|-------------|---------------|--------|
| 1 | **Data Infrastructure** | Clean, validated dataset of S&P 500 OHLCV, VIX, FOMC, and earnings data covering 2015–2026 (~500 tickers); documented limitations (Section VI) | Mar 2026 | ✅ Complete |
| 2 | **Feature Engineering** | Full feature matrix (RSI, Consecutive_Count, MA_Distance, BB_Position, MACD_Hist, Volume_Ratio, VIX_Level/Regime, FOMC_Proximity, Earnings_Proximity) for every (stock, expiry_date) pair with no look-ahead | Apr 2026 | ✅ Complete |
| 3 | **Rule-Based Model & Backtesting** | Implemented `RSIReversalStrategy`, parameter-optimised thresholds, in-sample and out-of-sample performance report (Profit Factor, Sharpe, Win Rate, Max Drawdown) | May 2026 | ✅ Complete |
| 4 | **Statistical Validation** | Monte Carlo permutation result (500 shuffles, p-value), 8-window walk-forward composite, temporal-stability heatmap | Jun 2026 | ✅ Complete |
| 5 | **Volatility Regime Analysis** | VIX-segmented win-rate analysis (Low / Medium / High); formal H₂ test and supporting visualisations | Jul 2026 | ✅ Complete |
| 6 | **Machine Learning Validation** | Trained Logistic Regression, Random Forest, and XGBoost models with walk-forward splits; SHAP summaries; H₃ evaluation and ML-vs-rule comparison | Jul–Aug 2026 | ✅ Complete |
| 7 | **Final Report & Defence** | Complete visualisation set (equity curve, Monte Carlo histogram, walk-forward heatmap, regime win rates, SHAP summary); written capstone report; editable presentation deck | Aug–Sep 2026 | ✅ Complete for submission package |

**Submission package status (2026-05-04):** The implementation, notebooks, final report, survivorship-bias proxy artifact, and editable presentation deck have been completed. The repository validation suite passes with `69 passed, 4 subtests passed`.

---

## XI. Expected Findings and Practical Implications

**Expected outcomes** (to be determined through formal analysis):

- The Monte Carlo permutation test will determine whether strategy returns are statistically distinguishable from random signal placement (H₀ vs H₁)
- Walk-forward analysis across 8 windows (2015–2026) will establish whether any edge is temporally stable or concentrated in a single market regime
- VIX regime segmentation will test H₂ — whether the signal win rate is materially higher during high-fear environments
- ML feature importance will test H₃ — whether RSI and Consecutive_Count dominate predictive power as embedded in the rule design
- Earnings proximity filtering will quantify how much fundamental-driven noise contaminates the technical signal

**Anticipated practical implication**: If H₁ is supported, the expiry-day mean-reversion effect would be consistent with institutional options hedging activity creating predictable short-term price recovery windows for deeply oversold stocks around monthly expiry dates. The VIX and earnings enrichment layers will allow the analysis to distinguish genuine technical mean-reversion from confounded macro or fundamental-driven moves.

**Known limitations**:
- Monthly expiry frequency generates relatively few signals per year, limiting statistical power for subgroup analyses.
- Commission and slippage assumptions are simplified relative to live trading conditions.
- Results may not persist under structural market changes (e.g., the shift toward weekly and 0DTE options dominance).
- ML model performance is constrained by sample size; complex models risk overfitting despite walk-forward discipline.
- Survivorship bias in the constituent universe (Section VI, Data Limitations) may inflate returns on delisted names.

---

## XII. References

Appel, G. (2005). *Technical analysis: Power tools for active investors*. Financial Times Prentice Hall.

Brock, W., Lakonishok, J., & LeBaron, B. (1992). Simple technical trading rules and the stochastic properties of stock returns. *Journal of Finance*, *47*(5), 1731–1764. https://doi.org/10.1111/j.1540-6261.1992.tb04681.x

Baltussen, G., Terstegge, J., & Whelan, P. (2025). *The derivative payoff bias* [Conference paper]. American Economic Association Annual Conference 2025. https://www.aeaweb.org/conference/2025/program/paper/N7rsBN2N

Bollinger, J. (2001). *Bollinger on Bollinger Bands*. McGraw-Hill.

Chen, S., & Li, G. (2023). Why does option-implied volatility forecast realized volatility? Evidence from news events. *Journal of Banking & Finance*, *156*, 107019. https://doi.org/10.1016/j.jbankfin.2023.107019

DeLisle, R. J., Diavatopoulos, D., Fodor, A., & Kassa, H. (2021). Variation in option implied volatility spread and future stock returns. *Research in International Business and Finance*, *58*, Article 101453. https://doi.org/10.1016/j.ribaf.2021.101453

Golez, B., & Jackwerth, J. C. (2012). Pinning in the S&P 500 futures. *Journal of Financial Economics*, *106*(3), 566–585. https://doi.org/10.1016/j.jfineco.2012.07.005

Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers: Implications for stock market efficiency. *Journal of Finance*, *48*(1), 65–91. https://doi.org/10.1111/j.1540-6261.1993.tb04702.x

Kim, M. J., Nelson, C. R., & Startz, R. (1991). Mean reversion in stock prices? A reappraisal. *Review of Economic Studies*, *58*(3), 515–528. https://doi.org/10.2307/2297955

Ni, S. X., Pearson, N. D., & Poteshman, A. M. (2005). Stock price clustering on option expiration dates. *Journal of Financial Economics*, *78*(1), 49–87. https://doi.org/10.1016/j.jfineco.2004.09.003

Poterba, J. M., & Summers, L. H. (1988). Mean reversion in stock prices: Evidence and implications. *Journal of Financial Economics*, *22*(1), 27–59. https://doi.org/10.1016/0304-405X(88)90021-9

Wilder, J. W. (1978). *New concepts in technical trading systems*. Trend Research.
