# Team Work & Review Assignment
**Project:** DAMO-699-3 MDA Capstone — SPY Expiry Signal Pipeline  
**University:** University of Niagara Falls Canada  
**Date:** 2026-04-27  
**Submission readiness update:** 2026-05-04  
**Source of record:** `doc/capstone/architecture.md` §10 Remaining Work

---

## Context

All coding in this project is completed by Claude Code (AI-assisted development).  
Each team member owns one **implementation stream**: they review every piece of code Claude Code produces within that stream, validate outputs against the success criteria below, and sign off before results are used in the final submission.

**Do not approve code that passes tests but produces wrong research conclusions.**  
If logic is questionable, raise it to the full group before the work is marked complete.

## Submission Readiness Status — 2026-05-04

The implementation streams listed below have been completed for the submission
package. This section supersedes the original "needs to be built" language in
the stream assignments while preserving those details as review traceability.

| Stream | Required output | Review status |
|---|---|---|
| Stream 1 — Data Enrichment, Configuration & Bias | Frozen params, ML dependencies, VIX/FOMC/Earnings enrichments, survivorship-bias proxy universe | Complete |
| Stream 2 — Portfolio Simulation & Core Metrics | RSI-based take-profit recalibration, `run_simulation()`, `trades_df`, VIX regime at signal, composite score, IS/OOS split | Complete |
| Stream 3 — Statistical Robustness Validation | Monte Carlo permutation test, 8-window walk-forward, H₂ z-test, parameter sensitivity | Complete |
| Stream 4 — ML Validation, Visualisations & Report | Six core charts, ML validation notebook, H₃ verdict, benchmark table, final report, editable PPTX deck | Complete |

Submission package artefacts:
- `doc/capstone/final_report.md`
- `doc/capstone/spy_expiry_capstone_presentation.pptx`
- `cache/sp500_constituents_2015.csv` (2015 proxy universe used for survivorship sensitivity)
- `notebooks/spy_expiry_analysis_v2.ipynb`
- `notebooks/ml_validation.ipynb`
- `notebooks/take_profit_research.ipynb`

Validation status: `python -m pytest tests/ -q` passes with 69 tests and 4 subtests.

---

## Team Members

| Name | Email | Student ID |
|---|---|---|
| Diego Bicieg | diego.bicieg9519@myunfc.ca | NF1909519 |
| ÍCARO DOS SANTOS RABELO | icaro.dos9242@myunfc.ca | NF1009242 |
| Sai Bhaskar Chundru | sai.chundru0427@myunfc.ca | NF1010427 |
| Osmond Xin | yi.xin7319@myunfc.ca | NF1007319 |

---

## Dependency Map

Not all streams are blocked equally. `capstone_v4_params.py` is the only immediate gate.  
VIX/FOMC/Earnings joins are parallel work that unblocks H₂ and ML, not the core simulation.

```
Stream 1 (Diego)
  ├─ capstone_v4_params.py ────────────────────────────────▶ Stream 2 Phase A starts
  ├─ VIX/FOMC/Earnings joins ──────────────────────────────▶ Stream 3 H₂ test
  │                                                            Stream 4 ML features
  └─ Survivorship bias check ◀── needs Stream 2 Phase B ──▶ Stream 4 Limitations section
     (cannot start until portfolio_simulator.py exists)

Stream 2 (ÍCARO)
  ├─ Phase A: take_profit re-run → exit_params dict ───────▶ feeds Phase B only
  ├─ Phase B: portfolio_simulator.py (uses exit_params) ───▶ trades_df (with vix_regime_at_signal)
  │                                                            run_simulation() callable
  ├─ calculate_composite_score() ──────────────────────────▶ Stream 3 starts
  └─ IS/OOS split

Stream 3 (Sai) — imports portfolio_simulator and calculate_composite_score
  ├─ Monte Carlo (permutes return values in trades_df, 500×)
  ├─ Walk-forward (calls run_simulation() with 8 date windows)
  ├─ Parameter sensitivity (calls run_simulation() ×12)
  └─ H₂ z-test (uses vix_regime_at_signal from trades_df)

Stream 4 (Osmond) — after Streams 2 + 3
  ├─ Report sections (Literature Review, Methods, Data Sources) ← can start NOW
  ├─ Core visualisations (needs trades_df + walk-forward + Monte Carlo)
  ├─ ML validation notebook (needs enrichment columns from Stream 1)
  ├─ ML benchmark: call run_simulation() with ML-predicted signals
  └─ Final report Key Findings + Presentation slides
```

---

## Stream 1 — Data Enrichment, Configuration & Bias Assessment
**Owner:** Diego Bicieg (NF1909519)  
**Priority:** P0 — `capstone_v4_params.py` gates everything; other tasks are parallel

### What needs to be built

| Phase | Task | File(s) | Architecture ref |
|---|---|---|---|
| **A (immediate)** | Create `modules/config/capstone_v4_params.py` — frozen single source of truth for all pre-registered v4.1 parameters | New file | §8, §10 P1 |
| **A (immediate)** | Add `scikit-learn`, `xgboost`, `shap` with pinned versions to `requirements.txt` | `requirements.txt` | §10 P0 |
| **B (parallel)** | Join `cache/vix.csv` onto the feature DataFrame by date; derive `VIX_Level` (raw close) and `VIX_Regime` (Low < 15 / Medium 15–25 / High > 25) | New utility in `modules/features/` | §3.4, §10 P0 |
| **B (parallel)** | Join `cache/fomc_dates.csv`; add `FOMC_Proximity` binary flag: FOMC meeting within ±5 calendar days of the signal date | Same location as VIX join | §3.4, §10 P0 |
| **B (parallel)** | Join `cache/earnings_dates.csv`; add `Earnings_Proximity` binary flag: earnings release for that ticker within ±3 days of signal date | Same location as VIX join | §3.4, §10 P0 |
| **C (after Stream 2 Phase B)** | Survivorship bias sensitivity check: run the full strategy on the current 2026 S&P 500 constituent list vs. a frozen 2015 proxy snapshot; compute and compare composite scores for both universes. Result is available in the final report Limitations section. `cache/sp500_constituents_2015.csv` contains the 453-ticker proxy universe used for the analysis. | `cache/sp500_constituents_2015.csv`; summarized in `doc/capstone/final_report.md` | §10 P1 |

### Parameters to freeze in `capstone_v4_params.py`

```python
RSI_THRESHOLD             = 22
MIN_CONSECUTIVE           = 3
HOLD_DAYS                 = 6
COMMISSION                = 0.002        # 0.2% per trade
MAX_ENTRIES_PER_EXPIRY    = 3
MAX_CONCURRENT_POSITIONS  = 15
COMPOSITE_WEIGHTS         = (0.35, 0.30, 0.20, 0.15)  # PF, Sharpe, WinRate, MaxDD
RANDOM_SEED               = 42
```

### Success criteria

- `from modules.config.capstone_v4_params import *` succeeds; every value above is present and correct.
- After applying the enrichment joins, every feature DataFrame for any ticker contains `VIX_Level`, `VIX_Regime`, `FOMC_Proximity`, and `Earnings_Proximity` with no all-NaN rows inside the 2015–2026 window.
- `VIX_Regime` contains only `'Low'`, `'Medium'`, `'High'`; no `NaN` on days where VIX data exists.
- `FOMC_Proximity` and `Earnings_Proximity` are 0/1 integers; manually verify at least 3 known FOMC dates and 3 known earnings events.
- Survivorship bias analysis reports two composite scores (2026 universe vs. 2015 snapshot) and explicitly states the percentage difference; this difference is included in the final report Limitations section.
- `pip install -r requirements.txt` in a clean virtual environment installs `scikit-learn`, `xgboost`, and `shap` without conflicts.
- `python -m pytest tests/` passes after all additions.

---

## Stream 2 — Portfolio Simulation & Core Metrics
**Owner:** ÍCARO DOS SANTOS RABELO (NF1009242)  
**Priority:** P0 — produces `trades_df`, the central data artifact that every other stream consumes  
**Depends on:** Stream 1 Phase A (`capstone_v4_params.py` only; does NOT need VIX join to start)

### What needs to be built

**Resolved:** The portfolio simulation now exists in
`modules/evaluation/portfolio_simulator.py` and is called from
`spy_expiry_analysis_v2.ipynb`. `RSIReversalStrategy` is used to generate the
capstone backtest trades.

**Important: two-phase ordering within this stream.** The take-profit research must be re-run first (Phase A) so the optimal exit parameters it discovers can be incorporated into the final portfolio simulation (Phase B). Do not skip ahead.

| Phase | Task | File(s) | Architecture ref |
|---|---|---|---|
| **A** | Re-run `take_profit_research.ipynb` §2 substituting `RSIReversalStrategy` entries in place of `EMAScanner` entries; run exit strategy grid search. **Output: a dict `exit_params` with keys `exit_type` (one of `'fixed_pct'`, `'trailing_stop'`, `'rsi_exit'`, `'hold_only'`), `take_profit_pct`, `trailing_stop_pct`, `rsi_exit_threshold` — only the relevant keys populated for the winning type.** Record this dict in a comment cell and pass it to Phase B. | `notebooks/take_profit_research.ipynb` | §10 P1 |
| **B** | Implement `run_simulation(rsi_threshold, min_consecutive, hold_days, start_date, end_date, exit_params=None) → trades_df` in `modules/evaluation/portfolio_simulator.py`. Logic: for each historical expiry date in `[start_date, end_date]`, apply `RSIReversalStrategy` to each ticker on the day before expiry; select top-3 signals ranked by lowest RSI; enter at next open (T+1); hold for `hold_days` or trigger exit per `exit_params`; deduct `COMMISSION` (round-trip 0.2%) per trade; enforce ≤ 15 concurrent positions. If `exit_params=None`, fall back to fixed `hold_days` only. The notebook calls this function; Stream 3 imports it. | New `modules/evaluation/portfolio_simulator.py`; called from `spy_expiry_analysis_v2.ipynb` | §10 P0 |
| **B** | Add `vix_regime_at_signal` column to `trades_df` by joining `cache/vix.csv` on `signal_date` (requires Stream 1 Phase B to be done first). Values: `'Low'`, `'Medium'`, `'High'` per the VIX thresholds in `capstone_v4_params`. This column is consumed by Stream 3's H₂ test. | `portfolio_simulator.py` or post-processing in notebook | §3.4 |
| **B** | Implement `calculate_composite_score(profit_factor, sharpe, win_rate, max_drawdown)`: `min(PF/2.0,1)×0.35 + min(max(Sharpe,0)/2.0,1)×0.30 + win_rate×0.20 + (1−min(|MaxDD|/0.30,1))×0.15` | `modules/evaluation/metrics.py` | §10 P0 |
| **B** | IS/OOS holdout split: call `run_simulation(..., start_date='2015-01-01', end_date='2025-06-30')` for IS and `run_simulation(..., start_date='2025-07-01', end_date=<latest_cache_date>)` for OOS (use the most recent date available in the cache, not a hardcoded future date); compute and print composite score for both; flag if gap > 0.05 | `spy_expiry_analysis_v2.ipynb` | §10 P0 |

**`trades_df` required columns:** `ticker, signal_date, entry_date, exit_date, entry_price, exit_price, net_return, gross_return, rsi_at_signal, consecutive_at_signal, vix_regime_at_signal`

**Expected row count:** With top-3 selection over ~132 expiry dates, `trades_df` will have approximately **200–400 rows** (fewer on dates where fewer than 3 signals fire, or where the 15-position cap is active). The 2,000–3,300 figure from the architecture refers to the ML feature matrix (all unfiltered signal pairs), not to `trades_df`.

### Success criteria

- `trades_df` row count is between 200 and 400 for the full 2015–2026 period.
- `trades_df` contains all required columns including `vix_regime_at_signal`; no NaN in that column for dates where VIX data exists.
- `net_return = gross_return − 0.002` for every row (commission is 0.2% round-trip per trade).
- For any 3 hand-verified (ticker, expiry_date) pairs, `net_return` matches manual calculation within 0.01%.
- `signal_date` is always strictly before `entry_date` for every row — no look-ahead bias.
- `calculate_composite_score(1.5, 1.0, 0.6, 0.1)` returns `0.6325` (verify: `0.75×0.35 + 0.50×0.30 + 0.60×0.20 + 0.667×0.15`).
- IS and OOS composite scores are both printed and the gap is explicitly labelled; OOS `end_date` is the actual latest cache date, not a hardcoded future date.
- `run_simulation()` is importable from `modules/evaluation/portfolio_simulator.py`; signature: `(rsi_threshold, min_consecutive, hold_days, start_date, end_date, exit_params=None) → pd.DataFrame`.

---

## Stream 3 — Statistical Robustness Validation
**Owner:** Sai Bhaskar Chundru (NF1010427)  
**Priority:** P0 (Monte Carlo, walk-forward) / P1 (H₂ test, sensitivity)  
**Depends on:** Stream 2 Phase B complete (`trades_df` and `calculate_composite_score()` must exist); Stream 1 Phase B complete (VIX join needed for H₂ test)

### What needs to be built

Neither the Monte Carlo permutation test nor the walk-forward analysis exists anywhere in the codebase. Both must be written from scratch in a new module.

| Task | File(s) | Architecture ref |
|---|---|---|
| New module `modules/evaluation/robust_validation.py` containing `MonteCarlo` and `WalkForward` classes | `modules/evaluation/robust_validation.py` | §10 P0 |
| **Monte Carlo permutation test (500 iterations):** set `seed = 42` once before the first shuffle. **Permutation method: shuffle the `net_return` values within `trades_df` (break the strategy-signal-return link while preserving the return distribution). Each iteration: sample a random permutation of `trades_df.net_return`, recompute portfolio-level Sharpe, Win Rate, Profit Factor, Max Drawdown, then `calculate_composite_score()`. Output: null distribution array (500 values), p-value = `sum(null ≥ observed) / 500`.** Do NOT re-run `run_simulation()` inside the loop (500 full simulations would be prohibitively slow and is not what the architecture specifies). | `robust_validation.py` + wire into `spy_expiry_analysis_v2.ipynb` §8 | §10 P0 |
| **Walk-forward 8-window analysis:** 8 rolling windows of 3-year IS / 1-year OOS. Window dates (OOS year): 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025. Each window: call `run_simulation(rsi_threshold=22, min_consecutive=3, hold_days=6, start_date=is_start, end_date=oos_end)`, compute composite score on the OOS portion only. Output: 8-row DataFrame of per-window metrics. | `robust_validation.py` + wire into `spy_expiry_analysis_v2.ipynb` §8 | §10 P0 |
| **H₂ two-proportion z-test:** split `trades_df` rows by `vix_regime_at_signal` column (provided by Stream 2). Compare win rate in `'High'` sub-sample vs. `'Low'` sub-sample; report z-statistic and one-sided p-value. **This column is already in `trades_df` — no additional join needed.** | `spy_expiry_analysis_v2.ipynb` §8 | §10 P1 |
| **Parameter sensitivity analysis:** for each of `rsi_threshold`, `min_consecutive`, `hold_days` individually, call `run_simulation()` at ±10% and ±20% of v4.1 values (12 runs total, each with full `start_date='2015-01-01'`). Report composite score at each perturbation. Pass criterion: < 15% degradation at all levels. | `robust_validation.py` | §10 P1 |

**Implementation notes:**
- `WalkForward` and parameter sensitivity call `run_simulation()` from `portfolio_simulator.py` — import it, do not copy the logic.
- `MonteCarlo` operates on an existing `trades_df` and does NOT call `run_simulation()`.
- Confirm the `run_simulation()` signature with ÍCARO before writing any wrapper code.

### Success criteria

- `seed = 42` is set exactly once before the first Monte Carlo shuffle.
- Monte Carlo p-value printed as `sum(null ≥ observed) / 500`; confirm with a reproducibility check: running the test twice with `seed = 42` produces identical p-values.
- Walk-forward output DataFrame has exactly 8 rows and columns: `window, is_start, is_end, oos_start, oos_end, pf, sharpe, win_rate, max_drawdown, composite`.
- All 8 walk-forward OOS composite scores are non-null (no window is empty).
- H₂ z-test uses a one-sided alternative (High-VIX win rate > Low-VIX); formula matches standard two-proportion z-test.
- Sensitivity table shows composite score at each of 12 perturbation points; pass/fail against the < 15% criterion is explicitly flagged per parameter.
- `python -m pytest tests/` passes after adding `robust_validation.py`.

---

## Stream 4 — ML Validation, Visualisations & Report
**Owner:** Osmond Xin (NF1007319)  
**Priority:** P0 (visualisations) / P1 (ML, report, slides)  
**Depends on:** Streams 2 + 3 complete for charts and Key Findings; Stream 1 Phase B complete for ML features  
**Can start immediately:** Literature Review, Data Sources & Limitations, and Analytical Methods sections of the final report do not depend on any simulation output and can be drafted in parallel with Streams 1–3.

### What needs to be built

**Resolved:** ML validation exists in `notebooks/ml_validation.ipynb`; the six
core visualisations are in `spy_expiry_analysis_v2.ipynb`; the final report is
complete; and the editable presentation deck is in `doc/capstone/`.

**Core visualisations** (P0 — in `spy_expiry_analysis_v2.ipynb`)

| Chart | Source data | Architecture ref |
|---|---|---|
| Equity curve vs. SPY buy-and-hold | `trades_df` from Stream 2 | §10 P0 |
| RSI distribution at signal dates (histogram) | `rsi_at_signal` column of `trades_df` | §10 P0 |
| Consecutive candle count distribution (bar chart) | `consecutive_at_signal` column of `trades_df` | §10 P0 |
| Walk-forward heatmap (8 windows × 4 metrics) | Walk-forward DataFrame from Stream 3 | §10 P0 |
| Monte Carlo permutation histogram with observed composite marked | Null distribution array from Stream 3 | §10 P0 |
| Win rate by VIX regime bar chart (Low / Medium / High) | H₂ sub-sample split from Stream 3 | §10 P0 |

**ML validation** (P1 — new `notebooks/ml_validation.ipynb`)

| Task | Architecture ref |
|---|---|
| Build ML feature matrix: all (ticker, expiry_date) signal pairs as rows (unfiltered universe, not just top-3 picks); target `y = 1` if T+6 return > 0; features = all TechnicalIndicators columns + VIX/FOMC/Earnings enrichments | §10 P1 |
| Logistic Regression: coefficient table showing directional feature contributions | §10 P1 |
| Random Forest: permutation importance ranking (primary H₃ test) | §10 P1 |
| XGBoost + SHAP summary plot | §10 P1 |
| Walk-forward validation for all ML models using the **same 8 windows** as Stream 3 — no random train/test split. For each window: train on IS rows of the ML feature matrix, predict on OOS rows, use predicted signals as entry triggers in `run_simulation()`, compute composite score on OOS `trades_df`. | §10 P1 |
| Benchmark comparison table: Rule-based v4.1 vs. LR vs. RF vs. XGBoost — OOS Composite, Sharpe, Win Rate, Profit Factor. **All composite scores must be computed via `calculate_composite_score()` on OOS portfolio-level results, not on classification metrics (AUC, F1, etc.)** | §10 P1 |

**Final report** (P0 — `doc/capstone/final_report.md`)

| Section | Note |
|---|---|
| Abstract | Brief summary of question, method, and key finding |
| Problem Context | Why SPY expiry dates, what the hypotheses are |
| Data Sources & Limitations | Stooq/Alpaca source, survivorship bias magnitude (from Stream 1 Phase C), enrichment data coverage |
| Analytical Methods | **Method justification narrative**: why RSI(14), why AND logic over OR, why walk-forward over simple train-test split, why top-3 per expiry, why 0.2% commission. Must address Criterion 4 rubric explicitly |
| Literature Review | Integrate and critically evaluate the 5 cited papers: Golez & Jackwerth 2012, Baltussen et al. 2025, Ni et al. 2005, Poterba & Summers 1988, Kim et al. 1991. Evaluate findings critically — not just descriptive summaries |
| Key Findings | Results of H₁, H₂, H₃ tests; composite scores; benchmark table |
| Robustness Validation | IS/OOS split, walk-forward stability, Monte Carlo p-value, sensitivity table |
| Limitations | Survivorship bias (quantified from Stream 1), EMAScanner/RSIReversalStrategy inconsistency in take-profit research, enrichment data coverage gaps |
| Practical Implications | What the results mean for an options trader |
| References | All cited papers plus data sources |

**Presentation** (P1 — `doc/capstone/presentation.pdf` or equivalent, 12–15 slides)

Structure: problem → data → method → key findings → limitations → conclusion.

### Success criteria

- All 6 charts are sourced from the actual simulation/validation outputs, not from a single scan date or manually constructed examples.
- RSI histogram and consecutive candle bar chart use `trades_df` rows, not any per-scan-date data.
- ML feature matrix has no future-leaking columns: all features are computed using only data available on the signal date.
- For H₃: if RSI and `Consecutive_Count` do NOT appear in the top-3 importances for ≥ 2 of 3 models, the result is reported honestly with interpretation — do not tune models to force the outcome.
- ML walk-forward uses the same 8 window boundaries as Stream 3 (verify window start/end dates match exactly).
- Final report Limitations section explicitly quantifies the survivorship bias magnitude from Stream 1 Phase C.
- Final report Limitations section explicitly notes the take-profit calibration inconsistency (EMAScanner vs. RSIReversalStrategy) and explains what was done to address it.
- All notebooks run end-to-end without errors: `jupyter nbconvert --to notebook --execute notebooks/<name>.ipynb`.

---

## P2 Extensions (Optional — pursue only after all P0/P1 work is complete)

These are listed in `architecture.md` §10 P2. No stream is blocked on them and they are not required for the capstone grade. Assign to whoever finishes their P0/P1 work first.

| Task | Effort | Architecture ref |
|---|---|---|
| Market regime segmentation: classify each expiry date as Bull/Sideways/Bear by SPY 200-day SMA; report Profit Factor per regime | Medium | §10 P2 |
| Earnings filter impact: compare win rate and Profit Factor with and without the `Earnings_Proximity` filter | Low | §10 P2 |
| Power BI dashboard: interactive replica of equity curve, regime breakdown, parameter sensitivity | High | §10 P2 |
| SQLite data layer: migrate `cache/*.csv` into a SQLite database via `modules/data/` | Medium | §10 P2 |

---

## Cross-Stream Interface Contracts

These are the exact data handoffs between streams. Reviewers on both sides must agree the interface is correct before downstream work begins.

| Handoff | Produced by | Consumed by | Contract |
|---|---|---|---|
| `capstone_v4_params` module | Stream 1 | Streams 2, 3, 4 | All parameter constants importable; `RSI_THRESHOLD = 22`, `COMPOSITE_WEIGHTS = (0.35, 0.30, 0.20, 0.15)`, etc. |
| VIX/FOMC/Earnings columns in feature DataFrame | Stream 1 | Streams 3, 4 | Columns `VIX_Level`, `VIX_Regime`, `FOMC_Proximity`, `Earnings_Proximity` present in every enriched DataFrame |
| `trades_df` | Stream 2 | Streams 3, 4 | Columns: `ticker, signal_date, entry_date, exit_date, entry_price, exit_price, gross_return, net_return, rsi_at_signal, consecutive_at_signal, vix_regime_at_signal` |
| `run_simulation(...)` in `portfolio_simulator.py` | Stream 2 | Streams 3, 4 | Signature: `(rsi_threshold, min_consecutive, hold_days, start_date, end_date, exit_params=None) → pd.DataFrame`; returns `trades_df` including `vix_regime_at_signal` |
| `calculate_composite_score()` | Stream 2 | Streams 3, 4 | Signature: `(profit_factor, sharpe, win_rate, max_drawdown) → float` |
| Walk-forward 8-row DataFrame | Stream 3 | Stream 4 | Columns: `window, is_start, is_end, oos_start, oos_end, pf, sharpe, win_rate, max_drawdown, composite` |
| Monte Carlo null distribution array | Stream 3 | Stream 4 | NumPy array of 500 composite score values; `seed = 42` fixed |
| Survivorship bias delta | Stream 1 | Stream 4 | A single percentage-point difference in composite score between the two universes |

---

## Shared Review Checklist (apply to every Claude Code change)

- [ ] Logic matches the proposal specification in `doc/capstone/architecture.md` §10 — not just "runs without error".
- [ ] No look-ahead bias: features and signals only use data available at the time of the decision.
- [ ] All parameters sourced from `modules/config/capstone_v4_params` — no hardcoded values in notebooks.
- [ ] `seed = 42` set before any stochastic operation.
- [ ] No hardcoded absolute paths, credentials, or local usernames.
- [ ] `python -m pytest tests/ -v` passes.
- [ ] No `print()` debug statements left in production module code.
- [ ] Docstrings on any new public function explain the purpose and return type.
