"""
Creates notebooks/eda.ipynb — Exploratory Data Analysis notebook.

Covers all 8 items from Diego Bicieg's 0428 feedback:
  1. Dataset shape and time coverage
  2. OHLCV descriptive statistics
  3. DataValidator quality summary
  4. Expiry calendar overview
  5. VIX regime distribution
  6. FOMC & earnings event density
  7. Technical feature distributions + signal frequency
  8. Feature correlation matrix + cross-sectional return baseline

Run from project root:
    python tools/create_eda_notebook.py
"""

import json, uuid

NB_PATH = "notebooks/eda.ipynb"


def code(src): return {"cell_type":"code","execution_count":None,"id":str(uuid.uuid4())[:8],"metadata":{},"outputs":[],"source":src}
def md(src):   return {"cell_type":"markdown","id":str(uuid.uuid4())[:8],"metadata":{},"source":src}


cells = []

# ── Title ───────────────────────────────────────────────────────────────────
cells.append(md("""\
# Exploratory Data Analysis — SPY Expiry Signal Pipeline
**Project**: DAMO-699-3 MDA Capstone — University of Niagara Falls Canada
**Purpose**: Systematic data introduction required by the "Data Collection and Preparation"
rubric criterion. Covers dataset shape, quality, enrichment coverage, feature distributions,
and cross-sectional return baseline before any modelling is introduced.

| Section | Content |
|---------|---------|
| §1 | Dataset Overview — ticker count, date range coverage |
| §2 | OHLCV Descriptive Statistics |
| §3 | Data Quality Report (`DataValidator`) |
| §4 | Expiry Calendar Overview |
| §5 | VIX Regime Distribution (2015–2026) |
| §6 | FOMC & Earnings Event Density |
| §7 | Technical Feature Distributions & Signal Frequency |
| §8 | Feature Correlation Matrix & Cross-Sectional Return Baseline |"""))

# ── Setup ────────────────────────────────────────────────────────────────────
cells.append(md("## 0. Setup"))
cells.append(code("""\
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

from modules.data.data_loader import DataLoader
from modules.data.data_validator import DataValidator
from modules.data.expiry_calculator import ExpiryCalculator
from modules.features.feature_pipeline import FeaturePipeline
from modules.features.technical_indicators import TechnicalIndicators
from modules.config.capstone_v4_params import (
    RSI_THRESHOLD, MIN_CONSECUTIVE,
    VIX_LOW_THRESHOLD, VIX_HIGH_THRESHOLD,
)

CACHE_DIR  = '../cache'
STUDY_START = '2015-01-01'
STUDY_END   = '2026-03-31'

plt.rcParams.update({'figure.dpi': 110, 'font.size': 11})
sns.set_style('darkgrid')
print("Setup complete.")"""))

# ── §1 Dataset Overview ──────────────────────────────────────────────────────
cells.append(md("""\
## §1 — Dataset Overview

Load all cached tickers and summarise the raw data universe: how many stocks, what
date ranges they cover, and how the coverage varies across the study window."""))

cells.append(code("""\
loader = DataLoader(cache_dir=os.path.join(CACHE_DIR, 'constituent_data'), auto_update=False)
tickers = loader.get_sp500_tickers()
print(f"Tickers in sp500_list.csv: {len(tickers)}")

# Load metadata for every ticker (first/last date, row count)
meta_rows = []
for tk in tickers:
    try:
        df = loader.load_ticker(tk, validate=False)
        if df is None or df.empty:
            continue
        meta_rows.append({
            'ticker':    tk,
            'first_date': df.index.min(),
            'last_date':  df.index.max(),
            'n_rows':     len(df),
        })
    except Exception:
        pass

meta = pd.DataFrame(meta_rows)
meta['first_date'] = pd.to_datetime(meta['first_date'])
meta['last_date']  = pd.to_datetime(meta['last_date'])
meta['years_covered'] = (meta['last_date'] - meta['first_date']).dt.days / 365.25

print(f"\\nSuccessfully loaded metadata: {len(meta)} tickers")
print(f"\\nDate range summary:")
print(f"  Earliest first_date : {meta['first_date'].min().date()}")
print(f"  Latest  first_date  : {meta['first_date'].max().date()}")
print(f"  Earliest last_date  : {meta['last_date'].min().date()}")
print(f"  Latest  last_date   : {meta['last_date'].max().date()}")
print(f"\\nRow count summary (trading days per ticker):")
print(meta['n_rows'].describe().round(0).to_string())"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1a. Distribution of first_date
axes[0].hist(meta['first_date'].dt.year, bins=range(2000, 2027),
             color='steelblue', edgecolor='white')
axes[0].set_title('Ticker First Available Year')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Ticker count')

# 1b. Distribution of years covered
axes[1].hist(meta['years_covered'], bins=20, color='steelblue', edgecolor='white')
axes[1].set_title('Years of Data per Ticker')
axes[1].set_xlabel('Years')
axes[1].set_ylabel('Ticker count')

# 1c. Tickers with data covering the full study window
study_start = pd.Timestamp(STUDY_START)
study_end   = pd.Timestamp(STUDY_END)
full_cover  = ((meta['first_date'] <= study_start) & (meta['last_date'] >= study_end)).sum()
partial     = len(meta) - full_cover
axes[2].bar(['Full 2015–2026', 'Partial coverage'], [full_cover, partial],
            color=['steelblue', 'darkorange'], edgecolor='white')
axes[2].set_title('Study-Window Coverage')
axes[2].set_ylabel('Ticker count')
for ax, v in zip([axes[2]], [full_cover, partial]):
    pass
axes[2].bar_label(axes[2].containers[0], padding=2)

plt.suptitle('§1 — Dataset Overview', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
print(f"\\nTickers with full study-window coverage (2015–2026): {full_cover}/{len(meta)}")"""))

# ── §2 OHLCV Descriptive Statistics ─────────────────────────────────────────
cells.append(md("""\
## §2 — OHLCV Descriptive Statistics

Aggregate price and volume statistics across the full universe.
A sample of 50 tickers is used for distribution plots to keep runtime reasonable;
summary statistics are computed across all available tickers."""))

cells.append(code("""\
# Sample 50 tickers for distribution plots (reproducible)
rng = np.random.default_rng(42)
sample_tickers = rng.choice(meta['ticker'].values,
                             size=min(50, len(meta)), replace=False).tolist()

sample_rows = []
for tk in sample_tickers:
    try:
        df = loader.load_ticker(tk, validate=False)
        if df is None or df.empty:
            continue
        # Filter to study window
        df = df[(df.index >= STUDY_START) & (df.index <= STUDY_END)]
        if df.empty:
            continue
        sample_rows.append({
            'ticker':      tk,
            'mean_close':  df['Close'].mean(),
            'std_close':   df['Close'].std(),
            'mean_volume': df['Volume'].mean(),
            'daily_ret_mean': df['Close'].pct_change().mean(),
            'daily_ret_std':  df['Close'].pct_change().std(),
        })
    except Exception:
        pass

sample_meta = pd.DataFrame(sample_rows)
print(f"Sample size for distribution plots: {len(sample_meta)} tickers")
print("\\nClose price statistics (sample):")
print(sample_meta[['mean_close','std_close']].describe().round(2).to_string())
print("\\nDaily return statistics (sample):")
print(sample_meta[['daily_ret_mean','daily_ret_std']].describe().round(5).to_string())"""))

cells.append(code("""\
fig, axes = plt.subplots(2, 2, figsize=(13, 8))

axes[0,0].hist(sample_meta['mean_close'].dropna(), bins=25,
               color='steelblue', edgecolor='white')
axes[0,0].set_title('Mean Close Price (per ticker, study window)')
axes[0,0].set_xlabel('USD')
axes[0,0].set_ylabel('Ticker count')

axes[0,1].hist(np.log10(sample_meta['mean_volume'].replace(0, np.nan).dropna()),
               bins=25, color='steelblue', edgecolor='white')
axes[0,1].set_title('Mean Daily Volume (log₁₀ scale)')
axes[0,1].set_xlabel('log₁₀(shares/day)')
axes[0,1].set_ylabel('Ticker count')

axes[1,0].hist(sample_meta['daily_ret_mean'].dropna() * 100, bins=30,
               color='steelblue', edgecolor='white')
axes[1,0].axvline(0, color='red', linewidth=1.2, linestyle='--')
axes[1,0].set_title('Mean Daily Return (per ticker)')
axes[1,0].set_xlabel('Daily return (%)')
axes[1,0].set_ylabel('Ticker count')

axes[1,1].hist(sample_meta['daily_ret_std'].dropna() * 100, bins=25,
               color='steelblue', edgecolor='white')
axes[1,1].set_title('Std Dev of Daily Return (per ticker)')
axes[1,1].set_xlabel('Daily return std (%)')
axes[1,1].set_ylabel('Ticker count')

plt.suptitle('§2 — OHLCV Descriptive Statistics (50-ticker sample)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()"""))

# ── §3 Data Quality ──────────────────────────────────────────────────────────
cells.append(md("""\
## §3 — Data Quality Report (`DataValidator`)

Run `DataValidator` across all tickers and report:
- Missing trading-day gaps (> 5 consecutive missing days)
- Duplicate date entries
- Non-monotonic price anomalies

This section provides the evidence base for the data-quality claim in the final report."""))

cells.append(code("""\
from modules.data.data_validator import DataValidator

validator = DataValidator()
quality_rows = []

# Use full ticker list but only report, do not filter
for tk in tickers:
    try:
        df = loader.load_ticker(tk, validate=False)
        if df is None or df.empty:
            continue
        df_study = df[(df.index >= STUDY_START) & (df.index <= STUDY_END)]
        if df_study.empty:
            continue
        is_valid, errors = validator.validate(df_study, tk)
        quality_rows.append({
            'ticker':    tk,
            'n_rows':    len(df_study),
            'is_valid':  is_valid,
            'n_issues':  len(errors),
            'issues':    '; '.join(errors[:3]) if errors else '',
        })
    except Exception as e:
        quality_rows.append({'ticker': tk, 'n_rows': 0,
                              'is_valid': False, 'n_issues': 1, 'issues': str(e)[:80]})

quality_df = pd.DataFrame(quality_rows)
n_valid   = quality_df['is_valid'].sum()
n_invalid = len(quality_df) - n_valid

print(f"DataValidator results ({len(quality_df)} tickers):")
print(f"  Valid (no issues):      {n_valid}  ({n_valid/len(quality_df):.1%})")
print(f"  Flagged (≥1 issue):     {n_invalid}  ({n_invalid/len(quality_df):.1%})")
print(f"\\nIssue count distribution:")
print(quality_df['n_issues'].value_counts().sort_index().head(10).to_string())
print("\\nSample of flagged tickers:")
display(quality_df[~quality_df['is_valid']][['ticker','n_rows','n_issues','issues']].head(10))"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Valid vs flagged
axes[0].bar(['Valid', 'Flagged'], [n_valid, n_invalid],
            color=['steelblue', 'firebrick'], edgecolor='white')
axes[0].set_title('Data Quality — Valid vs Flagged Tickers')
axes[0].set_ylabel('Ticker count')
axes[0].bar_label(axes[0].containers[0], padding=2)

# Issue count distribution
issue_counts = quality_df['n_issues'].value_counts().sort_index()
axes[1].bar(issue_counts.index.astype(str), issue_counts.values,
            color='steelblue', edgecolor='white')
axes[1].set_title('Distribution of Issue Count per Ticker')
axes[1].set_xlabel('Number of issues flagged')
axes[1].set_ylabel('Ticker count')

plt.suptitle('§3 — Data Quality Report', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()"""))

# ── §4 Expiry Calendar ───────────────────────────────────────────────────────
cells.append(md("""\
## §4 — Expiry Calendar Overview

Describes the monthly event cadence that structures the entire study."""))

cells.append(code("""\
expiry_dates = ExpiryCalculator.generate_expiry_dates(STUDY_START, STUDY_END)
expiry_df = pd.DataFrame({'expiry_date': expiry_dates})
expiry_df['year']  = expiry_df['expiry_date'].dt.year
expiry_df['month'] = expiry_df['expiry_date'].dt.month
expiry_df['dow']   = expiry_df['expiry_date'].dt.day_name()

print(f"Total expiry dates in study window ({STUDY_START} → {STUDY_END}): {len(expiry_df)}")
print(f"Date of first expiry: {expiry_df['expiry_date'].iloc[0].date()}")
print(f"Date of last expiry:  {expiry_df['expiry_date'].iloc[-1].date()}")
print(f"\\nExpiry counts by year:")
print(expiry_df['year'].value_counts().sort_index().to_string())
print(f"\\nDay-of-week distribution (should all be Friday):")
print(expiry_df['dow'].value_counts().to_string())"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Expiry count per year
per_year = expiry_df['year'].value_counts().sort_index()
axes[0].bar(per_year.index.astype(str), per_year.values,
            color='steelblue', edgecolor='white')
axes[0].axhline(12, color='red', linewidth=1, linestyle='--', label='12 per year')
axes[0].set_title('Monthly Expiry Dates per Year')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Count')
axes[0].legend()
plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')

# Distribution by calendar month
per_month = expiry_df['month'].value_counts().sort_index()
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
axes[1].bar([month_names[m-1] for m in per_month.index],
            per_month.values, color='steelblue', edgecolor='white')
axes[1].set_title('Expiry Dates by Calendar Month')
axes[1].set_ylabel('Count')

plt.suptitle('§4 — Expiry Calendar Overview', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()"""))

# ── §5 VIX Regime ────────────────────────────────────────────────────────────
cells.append(md("""\
## §5 — VIX Regime Distribution (2015–2026)

Describes the macro-volatility environment over the study window and shows
the proportion of trading days in each VIX regime (Low / Medium / High).
This context is essential for interpreting the H₂ VIX-moderation hypothesis."""))

cells.append(code("""\
vix = pd.read_csv(os.path.join(CACHE_DIR, 'vix.csv'), parse_dates=['Date'], index_col='Date')
vix.index = pd.to_datetime(vix.index).tz_localize(None)
vix.columns = ['VIX']
vix_study = vix.loc[STUDY_START:STUDY_END].copy()

vix_study['Regime'] = pd.cut(
    vix_study['VIX'],
    bins=[-np.inf, VIX_LOW_THRESHOLD, VIX_HIGH_THRESHOLD, np.inf],
    labels=['Low (<15)', 'Medium (15–25)', 'High (>25)'],
)

regime_counts = vix_study['Regime'].value_counts()
print(f"VIX data coverage in study window: {len(vix_study)} trading days")
print(f"\\nVIX regime breakdown:")
for regime, count in regime_counts.items():
    pct = count / len(vix_study) * 100
    print(f"  {regime:20s}: {count:4d} days  ({pct:.1f}%)")

print(f"\\nVIX summary statistics (study window):")
print(vix_study['VIX'].describe().round(2).to_string())"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# 5a. VIX time series
axes[0].plot(vix_study.index, vix_study['VIX'], linewidth=0.8, color='steelblue')
axes[0].axhline(VIX_LOW_THRESHOLD, color='green', linewidth=1.2, linestyle='--',
                label=f'Low threshold ({VIX_LOW_THRESHOLD})')
axes[0].axhline(VIX_HIGH_THRESHOLD, color='red', linewidth=1.2, linestyle='--',
                label=f'High threshold ({VIX_HIGH_THRESHOLD})')
axes[0].set_title('VIX Daily Close (2015–2026)')
axes[0].set_ylabel('VIX')
axes[0].legend(fontsize=9)

# 5b. Regime pie chart
colors_pie = ['#2196F3', '#FF9800', '#F44336']
axes[1].pie(regime_counts.values, labels=regime_counts.index,
            colors=colors_pie, autopct='%1.1f%%', startangle=90)
axes[1].set_title('VIX Regime Distribution\n(trading days, 2015–2026)')

# 5c. VIX by year (boxplot)
vix_study['Year'] = vix_study.index.year
vix_by_year = [vix_study.loc[vix_study['Year']==yr, 'VIX'].values
               for yr in sorted(vix_study['Year'].unique())]
axes[2].boxplot(vix_by_year, labels=sorted(vix_study['Year'].unique()),
                patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.6))
axes[2].axhline(VIX_LOW_THRESHOLD, color='green', linewidth=1, linestyle='--')
axes[2].axhline(VIX_HIGH_THRESHOLD, color='red', linewidth=1, linestyle='--')
axes[2].set_title('VIX Distribution by Year')
axes[2].set_ylabel('VIX')
plt.setp(axes[2].get_xticklabels(), rotation=45, ha='right')

plt.suptitle('§5 — VIX Regime Distribution', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()"""))

# ── §6 FOMC & Earnings ───────────────────────────────────────────────────────
cells.append(md("""\
## §6 — FOMC & Earnings Event Density

Shows how often FOMC meetings and earnings releases fall near expiry dates,
which informs the interpretation of the `FOMC_Proximity` and `Earnings_Proximity`
binary control variables."""))

cells.append(code("""\
fomc = pd.read_csv(os.path.join(CACHE_DIR, 'fomc_dates.csv'), parse_dates=['Date'])
fomc['Date'] = pd.to_datetime(fomc['Date'])
fomc_study = fomc[(fomc['Date'] >= STUDY_START) & (fomc['Date'] <= STUDY_END)].copy()
fomc_study['Year'] = fomc_study['Date'].dt.year

earnings = pd.read_csv(os.path.join(CACHE_DIR, 'earnings_dates.csv'), parse_dates=['Date'])
earnings['Date'] = pd.to_datetime(earnings['Date'])
earn_study = earnings[(earnings['Date'] >= STUDY_START) &
                      (earnings['Date'] <= STUDY_END)].copy()
earn_study['Year'] = earn_study['Date'].dt.year

print(f"FOMC meetings in study window: {len(fomc_study)}")
print(f"  Per year (avg): {len(fomc_study) / fomc_study['Year'].nunique():.1f}")
print(f"\\nEarnings events in study window: {len(earn_study)}")
print(f"  Unique tickers with earnings: {earn_study['Ticker'].nunique()}")
print(f"  Per year (avg): {len(earn_study) / earn_study['Year'].nunique():.0f}")

# How many expiry dates have an FOMC meeting within ±5 days?
from modules.config.capstone_v4_params import FOMC_PROXIMITY_DAYS, EARNINGS_PROXIMITY_DAYS

fomc_dates_np = fomc_study['Date'].values.astype('datetime64[D]').astype(np.int64)
expiry_days   = np.array([np.datetime64(d, 'D').astype(np.int64) for d in expiry_dates])

fomc_near_expiry = sum(
    np.any(np.abs(ed - fomc_dates_np) <= FOMC_PROXIMITY_DAYS)
    for ed in expiry_days
)
print(f"\\nExpiry dates with FOMC meeting within ±{FOMC_PROXIMITY_DAYS} days: "
      f"{fomc_near_expiry}/{len(expiry_dates)} ({fomc_near_expiry/len(expiry_dates):.1%})")"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# 6a. FOMC meetings per year
fomc_per_year = fomc_study['Year'].value_counts().sort_index()
axes[0].bar(fomc_per_year.index.astype(str), fomc_per_year.values,
            color='steelblue', edgecolor='white')
axes[0].axhline(8, color='red', linewidth=1, linestyle='--', label='8/year (typical)')
axes[0].set_title('FOMC Meetings per Year')
axes[0].set_ylabel('Meetings')
axes[0].legend(fontsize=9)
plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')

# 6b. Earnings events per year
earn_per_year = earn_study['Year'].value_counts().sort_index()
axes[1].bar(earn_per_year.index.astype(str), earn_per_year.values,
            color='steelblue', edgecolor='white')
axes[1].set_title('Earnings Releases per Year')
axes[1].set_ylabel('Events')
plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')

# 6c. Expiry-proximity overlap: pie
fomc_overlap = fomc_near_expiry
fomc_no_overlap = len(expiry_dates) - fomc_overlap
axes[2].pie([fomc_overlap, fomc_no_overlap],
            labels=[f'FOMC within ±{FOMC_PROXIMITY_DAYS}d', 'No FOMC'],
            colors=['firebrick', 'steelblue'],
            autopct='%1.0f%%', startangle=90)
axes[2].set_title('Expiry Dates Near a FOMC Meeting')

plt.suptitle('§6 — FOMC & Earnings Event Density', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()"""))

# ── §7 Feature Distributions & Signal Frequency ─────────────────────────────
cells.append(md("""\
## §7 — Technical Feature Distributions & Signal Frequency

Computes all `TechnicalIndicators` columns for a representative 50-ticker sample,
then shows:
- Distribution of each key feature across the universe (2015–2026)
- How often the primary signal fires (RSI < 22 AND Consecutive_Count ≥ 3) per expiry date"""))

cells.append(code("""\
pipeline = FeaturePipeline([TechnicalIndicators()])

feature_rows = []
signal_rows  = []

print("Computing features for 50-ticker sample ...")
for tk in sample_tickers[:50]:
    try:
        df = loader.load_ticker(tk, validate=False)
        if df is None or df.empty:
            continue
        df = pipeline.transform(df)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df_study = df[(df.index >= STUDY_START) & (df.index <= STUDY_END)].copy()
        if df_study.empty:
            continue

        # Collect feature rows
        for col in ['RSI', 'Consecutive_Count', 'BB_Position',
                    'MACD_Hist', 'Volume_Ratio']:
            if col in df_study.columns:
                for val in df_study[col].dropna().values:
                    feature_rows.append({'feature': col, 'value': val})

        # Check signal on each expiry signal date
        for expiry_date in expiry_dates:
            norm = pd.Timestamp(expiry_date).tz_localize(None)
            cands = df_study.index[df_study.index <= norm]
            if len(cands) < 2:
                continue
            sig_idx = len(cands) - 2  # 1 day before expiry
            row_s = df_study.iloc[sig_idx]
            rsi_v  = row_s.get('RSI', np.nan)
            cc_v   = row_s.get('Consecutive_Count', np.nan)
            cc_d   = row_s.get('Consecutive_Direction', np.nan)
            if pd.isna(rsi_v) or pd.isna(cc_v) or pd.isna(cc_d):
                continue
            fired = int(rsi_v < RSI_THRESHOLD and cc_v >= MIN_CONSECUTIVE and cc_d == -1)
            signal_rows.append({'expiry_date': expiry_date, 'ticker': tk,
                                 'rsi': rsi_v, 'fired': fired})
    except Exception:
        pass

feat_df   = pd.DataFrame(feature_rows)
signal_df = pd.DataFrame(signal_rows)
print(f"Feature rows collected: {len(feat_df):,}")
print(f"Signal scan rows:       {len(signal_df):,}")
print(f"Signal fire rate:       {signal_df['fired'].mean():.1%}")"""))

cells.append(code("""\
FEATURE_LABELS = {
    'RSI':              ('RSI(14)', 'Value', None),
    'Consecutive_Count':('Consecutive Red Candles', 'Count', None),
    'BB_Position':      ('Bollinger Band Position', 'BB_Position', (0, 1)),
    'MACD_Hist':        ('MACD Histogram', 'Value', None),
    'Volume_Ratio':     ('Volume Ratio (Vol / Vol_SMA)', 'Ratio', (0, 8)),
}

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes_flat = axes.flatten()

for i, (feat, (title, xlabel, xlim)) in enumerate(FEATURE_LABELS.items()):
    ax = axes_flat[i]
    vals = feat_df.loc[feat_df['feature'] == feat, 'value'].dropna()
    if xlim:
        vals = vals[(vals >= xlim[0]) & (vals <= xlim[1])]
    ax.hist(vals, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    if feat == 'RSI':
        ax.axvline(RSI_THRESHOLD, color='red', linewidth=1.5,
                   linestyle='--', label=f'Threshold ({RSI_THRESHOLD})')
        ax.legend(fontsize=9)
    if feat == 'Consecutive_Count':
        ax.axvline(MIN_CONSECUTIVE, color='red', linewidth=1.5,
                   linestyle='--', label=f'Min ({MIN_CONSECUTIVE})')
        ax.legend(fontsize=9)

# 6th panel: signal fires per expiry date
if not signal_df.empty:
    fires_per_expiry = signal_df.groupby('expiry_date')['fired'].sum()
    axes_flat[5].hist(fires_per_expiry.values, bins=range(0, 25),
                      color='steelblue', edgecolor='white')
    axes_flat[5].axvline(fires_per_expiry.mean(), color='red', linewidth=1.5,
                          linestyle='--',
                          label=f'Mean = {fires_per_expiry.mean():.1f}')
    axes_flat[5].set_title('Signal Fires per Expiry Date\\n(50-ticker sample)')
    axes_flat[5].set_xlabel('Number of tickers firing signal')
    axes_flat[5].set_ylabel('Expiry date count')
    axes_flat[5].legend(fontsize=9)

plt.suptitle('§7 — Technical Feature Distributions & Signal Frequency', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

if not signal_df.empty:
    print(f"\\nSignal frequency summary (50-ticker sample):")
    print(f"  Total (ticker, expiry) pairs scanned: {len(signal_df):,}")
    print(f"  Signal fires (RSI<{RSI_THRESHOLD} AND CC≥{MIN_CONSECUTIVE}): "
          f"{signal_df['fired'].sum():,}  ({signal_df['fired'].mean():.2%})")
    print(f"  Mean fires per expiry date: {fires_per_expiry.mean():.1f}")
    print(f"  Max fires on a single expiry: {fires_per_expiry.max()}")"""))

# ── §8 Correlation + Return Baseline ────────────────────────────────────────
cells.append(md("""\
## §8 — Feature Correlation Matrix & Cross-Sectional Return Baseline

### 8a. Feature Correlation Matrix
Shows linear (Pearson) correlations between all technical features.
High inter-feature correlation would indicate multicollinearity relevant to
the ML models in `ml_validation.ipynb`.

### 8b. Cross-Sectional Return Baseline on Expiry Dates
Before the strategy is evaluated, this baseline answers: what is the typical
return distribution for S&P 500 stocks on expiry days, ignoring the signal?
This is the null-hypothesis distribution the strategy must beat."""))

cells.append(code("""\
# 8a. Correlation matrix on a wider feature set from the sample
CORR_FEATURES = ['RSI', 'Consecutive_Count', 'BB_Position',
                 'MACD_Hist', 'Volume_Ratio',
                 'MA_Distance_20', 'MA_Distance_50']

corr_rows = []
for tk in sample_tickers[:50]:
    try:
        df = loader.load_ticker(tk, validate=False)
        if df is None or df.empty:
            continue
        df = pipeline.transform(df)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df_study = df[(df.index >= STUDY_START) & (df.index <= STUDY_END)]
        avail = [c for c in CORR_FEATURES if c in df_study.columns]
        corr_rows.append(df_study[avail].dropna())
    except Exception:
        pass

if corr_rows:
    corr_data = pd.concat(corr_rows, ignore_index=True)
    corr_matrix = corr_data.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1,
                linewidths=0.5, ax=ax, mask=mask)
    ax.set_title('§8a — Feature Correlation Matrix (Pearson, 50-ticker sample)')
    plt.tight_layout()
    plt.show()

    print("\\nHighest absolute correlations (|r| > 0.3):")
    pairs = corr_matrix.unstack()
    pairs = pairs[pairs.index.get_level_values(0) < pairs.index.get_level_values(1)]
    high_corr = pairs[abs(pairs) > 0.3].sort_values(key=abs, ascending=False)
    print(high_corr.round(3).to_string())"""))

cells.append(code("""\
# 8b. Cross-sectional return baseline on expiry dates vs. non-expiry dates
# Use a subset of tickers for speed
baseline_rows = []
expiry_set = set(str(d.date()) for d in expiry_dates)

for tk in sample_tickers[:30]:
    try:
        df = loader.load_ticker(tk, validate=False)
        if df is None or df.empty:
            continue
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df_study = df[(df.index >= STUDY_START) & (df.index <= STUDY_END)].copy()
        if df_study.empty:
            continue
        df_study['intraday_ret'] = (df_study['Close'] - df_study['Open']) / df_study['Open']
        df_study['is_expiry']    = df_study.index.strftime('%Y-%m-%d').isin(expiry_set)
        baseline_rows.append(df_study[['intraday_ret', 'is_expiry']])
    except Exception:
        pass

if baseline_rows:
    baseline = pd.concat(baseline_rows)
    expiry_rets    = baseline.loc[baseline['is_expiry'],    'intraday_ret'].dropna() * 100
    nonexpiry_rets = baseline.loc[~baseline['is_expiry'], 'intraday_ret'].dropna() * 100

    tstat, pval = stats.ttest_ind(expiry_rets, nonexpiry_rets)

    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.linspace(-6, 6, 60)
    ax.hist(nonexpiry_rets.clip(-6, 6), bins=bins, alpha=0.55,
            color='darkorange', label=f'Non-expiry days (n={len(nonexpiry_rets):,})')
    ax.hist(expiry_rets.clip(-6, 6), bins=bins, alpha=0.65,
            color='steelblue', label=f'Expiry days (n={len(expiry_rets):,})')
    ax.axvline(expiry_rets.mean(), color='blue', linewidth=1.5, linestyle='--',
               label=f'Expiry mean = {expiry_rets.mean():.2f}%')
    ax.axvline(nonexpiry_rets.mean(), color='darkorange', linewidth=1.5, linestyle='--',
               label=f'Non-expiry mean = {nonexpiry_rets.mean():.2f}%')
    ax.set_title('§8b — Intraday Returns: Expiry vs. Non-Expiry Days (30-ticker sample)')
    ax.set_xlabel('Intraday return (%)')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=9)
    ax.set_xlim(-6, 6)
    plt.tight_layout()
    plt.show()

    print(f"\\nReturn comparison (expiry vs. non-expiry):")
    print(f"  Expiry days    — mean: {expiry_rets.mean():.3f}%  std: {expiry_rets.std():.3f}%")
    print(f"  Non-expiry days— mean: {nonexpiry_rets.mean():.3f}%  std: {nonexpiry_rets.std():.3f}%")
    print(f"  t-statistic: {tstat:.3f}   p-value: {pval:.4f}")
    sig = '✅ Significant difference' if pval < 0.05 else '⚠️  No significant difference'
    print(f"  {sig} in intraday returns between expiry and non-expiry days")"""))

# ── Summary ──────────────────────────────────────────────────────────────────
cells.append(md("## Summary"))
cells.append(code("""\
print("="*60)
print("EDA SUMMARY")
print("="*60)
print(f"  Tickers loaded:                {len(meta)}")
print(f"  Full-coverage tickers (2015–26):{full_cover}")
print(f"  Data quality — valid tickers:  {n_valid} ({n_valid/len(quality_df):.1%})")
print(f"  Expiry dates in study window:  {len(expiry_df)}")
print(f"  VIX study days:                {len(vix_study)}")
print(f"  FOMC meetings:                 {len(fomc_study)}")
print(f"  Earnings events:               {len(earn_study):,}")
if not signal_df.empty:
    print(f"  Signal fire rate (50-ticker):  {signal_df['fired'].mean():.2%}")
print("="*60)
print("All 8 EDA sections complete.")"""))

# ── Assemble ─────────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.12.0"},
    },
    "cells": cells,
}
with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"✅ Created {NB_PATH}  ({len(cells)} cells)")
