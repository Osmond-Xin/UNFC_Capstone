# Revision Notes — project_purpose.md

**Source**: Instructor rubric feedback (87/100), received 2026-04-22
**Applies to**: `doc/capstone/project_purpose.md`

---

## Criterion 1 — Clarity of Problem and Relevance (18/20)

**Feedback**: Narrow the stated contribution so the project is framed less like a full trading research program and more like one clearly bounded capstone study with a specific decision problem.

**Changes made**:
- **Abstract**: Removed "bridges interpretable finance theory with data-driven model validation" and "three previously disjoint literatures" language. Reframed to state one specific decision problem: does a pre-specified rule (RSI oversold + consecutive bearish candles) generate a statistically detectable, temporally stable positive-return edge in S&P 500 stocks on SPY monthly expiry days? Trimmed to ≤ 200 words.
- **Section II (Solution Statement)**: Trimmed the "academic contribution" paragraph. Removed broad synthesis framing. Reworded to: this study tests whether a single observable technical rule captures expiry-driven mean-reversion, with VIX as one conditioning variable.
- **Section III (Analytical Objective)**: Removed embedded success thresholds from objective statements (those belong only in Section V Hypotheses and Section 7.6 Methods).

---

## Criterion 2 — Research Questions & Hypotheses (18/20)

**Feedback**: Simplify and prioritize the research questions and hypotheses. Too many technical thresholds and success conditions are fixed at the proposal stage.

**Changes made**:
- **Section IV**: Collapsed from five questions (Primary + RQ1a + RQ1b + RQ2 + RQ3) to three:
  - Primary RQ: Does the signal produce a statistically distinguishable and temporally stable positive-return edge?
  - RQ2: Does VIX regime moderate signal effectiveness?
  - RQ3: Do ML models independently validate RSI and Consecutive_Count as dominant predictors?
  - Removed RQ1a and RQ1b as separate numbered sub-questions; their content is incorporated into the primary RQ description.
- **Section V (Hypotheses)**: Removed specific numerical success conditions from H₁ (deleted "Profit Factor > 1.0 and Win Rate > 0.50", "p < 0.05", and "mean OOS composite > 0.45" from the hypothesis definition). H₁ now states the qualitative claim; thresholds appear only in Section 7.6 as evaluation criteria.

---

## Criterion 3 — Preliminary Literature Insight (17/20)

**Feedback**: Add recent academic sources on stock-level expiry effects, technical-signal validity, and event-driven return prediction. Compare studies more critically.

**Changes made**:
- **Section 9.1**: Added Ni, Pearson & Poteshman (2005) on stock price clustering at option expiration dates — the key missing stock-level paper. Added critical bridging sentence: "While Golez & Jackwerth (2012) and Baltussen et al. (2025) establish index-level dynamics, Ni et al. (2005) confirm these effects extend to individual stocks, directly motivating our cross-sectional approach."
- **New Section 9.3 — Technical Signal Validity**: Added Brock, Lakonishok & LeBaron (1992) on technical trading rules. Noted that technical signal validity is contested in efficient-market contexts, which motivates the Monte Carlo permutation test as the correct null-hypothesis test rather than a standard t-test.
- **Section 9.2**: Added Jegadeesh & Titman (1993) on short-horizon return reversals in individual stocks. Noted that their 1-month momentum reversal complements (rather than contradicts) the expiry-driven 9-day mean-reversion window studied here.
- **Section XII (References)**: Added three new APA-formatted entries: Brock et al. (1992), Jegadeesh & Titman (1993), Ni et al. (2005).

---

## Criterion 4 — Proposed Data Sources and Methods (18/20)

**Feedback**: Reduce methodological scope. Address survivorship bias more directly. Clarify which analyses are essential versus optional.

**Changes made**:
- **Section 7.6 (Robustness Validation)**: Added an "Essential / Optional" column to the table:
  - Essential: IS/OOS split, Walk-Forward Analysis, Monte Carlo Permutation, Parameter Sensitivity
  - Optional (time-permitting extensions): Market Regime Segmentation, Earnings Filter Impact
- **Section VI (Data Limitations — Survivorship Bias)**: Elevated survivorship bias from a table row into its own dedicated paragraph. Language makes the limitation explicit and names two mitigations: (1) walk-forward OOS windows reduce in-sample selection artefacts; (2) a constant 2015-universe sensitivity check is planned as an optional extension.
- **Section 7.3 (ML)**: Added sentence clarifying that ML validation is a secondary, confirmatory analysis. Feature importance from a single Random Forest model is sufficient to evaluate H₃; XGBoost + SHAP is an optional extension if time permits.

---

## Criterion 5 — Structure & APA Formatting (16/20)

**Feedback**: Shorten some sections, reduce technical density, review APA/style consistency.

**Changes made**:
- **Abstract**: Trimmed to ≤ 200 words.
- **Section 7.1 (Feature Engineering)**: Removed `BB_Position`, `MACD_Hist`, and `Volume_Ratio` rows from the feature table. These are engineered but not part of the core signal rule; replaced with a brief prose sentence noting them as "additional features available for ML training."
- **Section 7.2 (Rule-Based Model)**: Removed the code block. Signal rule described in plain English only; code belongs in the repository.
- **Section 7.4 (Portfolio Simulation)**: Condensed to three sentences; detailed constraints list replaced with concise prose.
- **APA fixes**:
  - Ensured DOI links use `https://doi.org/` format consistently.
  - Verified journal volume/issue formatting (volume in italics, issue in parentheses, not italicised) is consistent across all references.
  - Conference citation for Baltussen et al. (2025) uses the APA conference paper format.
