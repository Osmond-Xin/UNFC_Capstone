"""
One-time script: insert portfolio simulation, statistical validation,
and 6 core visualisation cells into spy_expiry_analysis_v2.ipynb.

Run from the project root:
    python tools/patch_notebook.py
"""

import json
import copy
import os
import uuid

NOTEBOOK_PATH = "notebooks/spy_expiry_analysis_v2.ipynb"


def new_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def new_markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "source": source,
    }


# ---------------------------------------------------------------------------
# Cell content
# ---------------------------------------------------------------------------

MD_SIM = """\
## 5. Portfolio Simulation — RSI Reversal Strategy v4.1

Implements the full capstone backtest:
- Signal date: 1 trading day before each monthly expiry (3rd Friday)
- Entry: Open price on expiry date (T+1 after signal)
- Selection: Top-3 tickers ranked by lowest RSI per expiry cycle
- Exit: Close price after `HOLD_DAYS` trading days
- Commission: 0.2% round-trip per trade
- Max concurrent positions: 15

**In-Sample (IS):** 2015-01-01 → 2025-06-30
**Out-of-Sample (OOS):** 2025-07-01 → latest cache date"""

CODE_SIM_IMPORTS = """\
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Ensure project root is on path
sys.path.insert(0, os.path.abspath('..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Capstone v4.1 parameters — single source of truth
from modules.config.capstone_v4_params import (
    RSI_THRESHOLD, MIN_CONSECUTIVE, HOLD_DAYS,
    IS_START, IS_END, OOS_START,
    COMMISSION,
)
from modules.evaluation.portfolio_simulator import run_simulation
from modules.evaluation.metrics import (
    calculate_composite_score,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
)

print(f"v4.1 parameters: RSI_THRESHOLD={RSI_THRESHOLD}, "
      f"MIN_CONSECUTIVE={MIN_CONSECUTIVE}, HOLD_DAYS={HOLD_DAYS}, "
      f"COMMISSION={COMMISSION}")

# Determine latest cache date dynamically
_cache_root = '../cache/constituent_data'
_sample_files = [f for f in os.listdir(_cache_root) if f.endswith('.csv')][:50]
_latest_dates = []
for _f in _sample_files:
    try:
        _df = pd.read_csv(os.path.join(_cache_root, _f), usecols=['Date'])
        _latest_dates.append(pd.to_datetime(_df['Date']).max())
    except Exception:
        pass
OOS_END = max(_latest_dates).strftime('%Y-%m-%d') if _latest_dates else '2026-04-01'
print(f"OOS end date (latest cache): {OOS_END}")"""

CODE_SIM_IS = """\
print(f"\\n{'='*60}")
print(f"Running IS simulation  ({IS_START} → {IS_END})")
print("Estimated: 3–8 minutes depending on hardware")
print(f"{'='*60}")

trades_df = run_simulation(
    rsi_threshold=RSI_THRESHOLD,
    min_consecutive=MIN_CONSECUTIVE,
    hold_days=HOLD_DAYS,
    start_date=IS_START,
    end_date=IS_END,
    cache_dir='../cache',
    verbose=True,
)
print(f"\\n✅ IS trades: {len(trades_df)}")
if not trades_df.empty:
    display(trades_df.head(10))
    print("\\nColumn dtypes:")
    print(trades_df.dtypes)"""

CODE_SIM_OOS = """\
print(f"\\n{'='*60}")
print(f"Running OOS simulation ({OOS_START} → {OOS_END})")
print(f"{'='*60}")

trades_df_oos = run_simulation(
    rsi_threshold=RSI_THRESHOLD,
    min_consecutive=MIN_CONSECUTIVE,
    hold_days=HOLD_DAYS,
    start_date=OOS_START,
    end_date=OOS_END,
    cache_dir='../cache',
    verbose=True,
)
print(f"\\n✅ OOS trades: {len(trades_df_oos)}")


def _portfolio_summary(df, label):
    \"\"\"Compute and print portfolio metrics, return as dict.\"\"\"
    if df is None or df.empty:
        print(f"{label}: no trades")
        return None
    rets = df['net_return'].values
    n = len(rets)
    wins = (rets > 0).sum()
    win_rate = wins / n
    gross_profit = rets[rets > 0].sum() if wins > 0 else 0.0
    gross_loss = abs(rets[rets < 0].sum()) if (rets < 0).any() else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)
    sharpe = float(calculate_sharpe_ratio(rets * 100))
    cum = (1 + rets).cumprod()
    run_max = np.maximum.accumulate(cum)
    mdd = float(abs(((cum - run_max) / run_max).min())) if n > 0 else 0.0
    composite = calculate_composite_score(pf, sharpe, win_rate, mdd)
    print(f"\\n{label}:")
    print(f"  Trades:        {n}")
    print(f"  Win Rate:      {win_rate:.1%}")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Sharpe Ratio:  {sharpe:.2f}")
    print(f"  Max Drawdown:  {mdd:.1%}")
    print(f"  Composite:     {composite:.4f}")
    return {'pf': pf, 'sharpe': sharpe, 'win_rate': win_rate, 'mdd': mdd, 'composite': composite}


is_metrics  = _portfolio_summary(trades_df,     f"IS  ({IS_START} → {IS_END})")
oos_metrics = _portfolio_summary(trades_df_oos, f"OOS ({OOS_START} → {OOS_END})")

# IS / OOS gap check (threshold: < 0.05)
if is_metrics and oos_metrics:
    gap = abs(is_metrics['composite'] - oos_metrics['composite'])
    status = '✅ PASS' if gap < 0.05 else '⚠️  REVIEW'
    print(f"\\n{'='*60}")
    print(f"IS  composite:  {is_metrics['composite']:.4f}")
    print(f"OOS composite:  {oos_metrics['composite']:.4f}")
    print(f"Gap:            {gap:.4f}  {status} (threshold: < 0.05)")
    print(f"{'='*60}")"""

# --- Statistical validation ---

MD_STATS = """\
## 8a. Statistical Robustness Validation

Three tests required by the capstone proposal (§7.6):

1. **Monte Carlo permutation test** — H₁: observed composite is not random (500 shuffles, seed=42)
2. **Walk-forward analysis** — 8 windows of 3-yr IS / 1-yr OOS, shows temporal stability
3. **H₂ two-proportion z-test** — win rate in High-VIX sub-sample vs Low-VIX sub-sample"""

CODE_MONTE_CARLO = """\
from modules.evaluation.robust_validation import MonteCarlo

print("Running Monte Carlo permutation test (500 iterations, seed=42) ...")
mc = MonteCarlo(trades_df)
mc_p_value, mc_null_dist = mc.run()

observed_composite = is_metrics['composite'] if is_metrics else 0.0
print(f"\\nObserved composite score:    {observed_composite:.4f}")
print(f"Null distribution mean:      {mc_null_dist.mean():.4f}")
print(f"Null distribution 95th pct:  {np.percentile(mc_null_dist, 95):.4f}")
print(f"Monte Carlo p-value:         {mc_p_value:.4f}")
print(f"  ({int(mc_p_value * 500)} of 500 permutations scored ≥ observed)")
if mc_p_value < 0.05:
    print("✅ H₁ supported: strategy performance is statistically significant (p < 0.05)")
elif mc_p_value < 0.10:
    print("⚠️  Marginal significance (0.05 ≤ p < 0.10) — interpret cautiously")
else:
    print("❌ H₁ not supported at 0.05 level — performance is consistent with random returns")"""

CODE_WALKFORWARD = """\
from modules.evaluation.robust_validation import WalkForward

print("Running walk-forward analysis (8 windows) ...")
print("Each window: 3-year IS period + 1-year OOS period")
print("Estimated: 20–60 minutes (8× run_simulation calls)")
print()

wf = WalkForward(cache_dir='../cache')
wf_results = wf.run(verbose=True)

print("\\nWalk-Forward Results:")
display(wf_results[['window', 'oos_start', 'oos_end',
                     'pf', 'sharpe', 'win_rate', 'max_drawdown', 'composite']]
        .round(4))

non_null = wf_results['composite'].dropna()
print(f"\\nOOS composite — mean: {non_null.mean():.4f}, "
      f"min: {non_null.min():.4f}, max: {non_null.max():.4f}")
print(f"Windows with composite > 0.5: {(non_null > 0.5).sum()} / {len(non_null)}")"""

CODE_H2_ZTEST = """\
# H₂ two-proportion z-test: High-VIX win rate vs Low-VIX win rate
# H₂: strategy performs better (higher win rate) in high-volatility regimes

high_trades = trades_df[trades_df['vix_regime_at_signal'] == 'High']
low_trades  = trades_df[trades_df['vix_regime_at_signal'] == 'Low']
med_trades  = trades_df[trades_df['vix_regime_at_signal'] == 'Medium']

print("VIX Regime breakdown of IS trades:")
for label, sub in [('Low', low_trades), ('Medium', med_trades), ('High', high_trades)]:
    if len(sub) > 0:
        wr = (sub['net_return'] > 0).mean()
        print(f"  {label:8s}: {len(sub):4d} trades, win rate = {wr:.1%}")

n_high = len(high_trades)
n_low  = len(low_trades)

if n_high >= 5 and n_low >= 5:
    wins_high = (high_trades['net_return'] > 0).sum()
    wins_low  = (low_trades['net_return'] > 0).sum()
    wr_high   = wins_high / n_high
    wr_low    = wins_low  / n_low

    # Pooled proportion
    p_pool = (wins_high + wins_low) / (n_high + n_low)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_high + 1 / n_low))
    z_stat = (wr_high - wr_low) / se if se > 0 else 0.0
    p_one_sided = 1 - stats.norm.cdf(z_stat)   # one-sided: High > Low

    print(f"\\nH₂ Two-proportion z-test (one-sided, High > Low):")
    print(f"  High-VIX: n={n_high}, wins={wins_high}, win rate={wr_high:.1%}")
    print(f"  Low-VIX:  n={n_low},  wins={wins_low},  win rate={wr_low:.1%}")
    print(f"  z-statistic:         {z_stat:.3f}")
    print(f"  p-value (one-sided): {p_one_sided:.4f}")
    if p_one_sided < 0.05:
        print("✅ H₂ supported: High-VIX win rate significantly higher than Low-VIX (p < 0.05)")
    else:
        print("⚠️  H₂ not supported at 0.05 — no significant VIX regime effect on win rate")
else:
    print(f"\\n⚠️  Insufficient samples for z-test (High: {n_high}, Low: {n_low})")
    print("H₂ test requires ≥5 trades in both High and Low VIX regimes.")"""

# --- Visualisations ---

MD_VIZ = """\
## 10. Core Visualisations (Six Required Charts)

All charts are sourced from `trades_df` (IS simulation output), `wf_results`
(walk-forward DataFrame), and `mc_null_dist` (Monte Carlo null distribution)."""

CODE_VIZ = """\
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd

plt.rcParams.update({'figure.dpi': 120, 'font.size': 11})
sns.set_style('darkgrid')

# --- Guard: check required objects exist ---
_missing = []
for _name in ['trades_df', 'wf_results', 'mc_null_dist', 'is_metrics', 'oos_metrics']:
    if _name not in dir() or eval(_name) is None:
        _missing.append(_name)
if _missing:
    print(f"⚠️  Cannot render charts — missing: {', '.join(_missing)}")
    print("Run the simulation cells above first.")
else:

 # -----------------------------------------------------------------------
 # 1. Equity Curve vs SPY Buy-and-Hold
 # -----------------------------------------------------------------------
 fig1, ax1 = plt.subplots(figsize=(12, 5))

 strategy_rets = trades_df.sort_values('entry_date')['net_return'].values
 strategy_cum  = (1 + strategy_rets).cumprod()
 strategy_dates = trades_df.sort_values('entry_date')['entry_date']

 ax1.plot(strategy_dates, (strategy_cum - 1) * 100,
          label='RSI Reversal v4.1', linewidth=2, color='steelblue')

 # Load SPY for comparison (if available)
 try:
     spy = pd.read_csv('../cache/constituent_data/SPY.csv', parse_dates=['Date'])
     spy = spy[(spy['Date'] >= strategy_dates.iloc[0]) &
               (spy['Date'] <= strategy_dates.iloc[-1])]
     spy = spy.sort_values('Date')
     spy_ret  = spy['Close'].pct_change().fillna(0)
     spy_cum  = (1 + spy_ret).cumprod()
     ax1.plot(spy['Date'], (spy_cum - 1) * 100,
              label='SPY Buy-and-Hold', linewidth=1.5,
              color='darkorange', linestyle='--')
 except Exception:
     pass

 ax1.axhline(0, color='black', linewidth=0.8, linestyle=':')
 ax1.set_title('Equity Curve — Strategy vs SPY Buy-and-Hold (IS Period)')
 ax1.set_xlabel('Date')
 ax1.set_ylabel('Cumulative Return (%)')
 ax1.legend()
 plt.tight_layout()
 plt.show()

 # -----------------------------------------------------------------------
 # 2. RSI Distribution at Signal Dates (histogram)
 # -----------------------------------------------------------------------
 fig2, ax2 = plt.subplots(figsize=(8, 4))
 rsi_vals = trades_df['rsi_at_signal'].dropna()
 ax2.hist(rsi_vals, bins=20, color='steelblue', edgecolor='white', alpha=0.85)
 ax2.axvline(rsi_vals.mean(), color='red', linewidth=1.5,
             linestyle='--', label=f'Mean RSI = {rsi_vals.mean():.1f}')
 ax2.axvline(RSI_THRESHOLD, color='orange', linewidth=1.5,
             linestyle='-', label=f'Threshold = {RSI_THRESHOLD}')
 ax2.set_title('RSI Distribution at Signal Dates')
 ax2.set_xlabel('RSI at Signal Date')
 ax2.set_ylabel('Trade Count')
 ax2.legend()
 plt.tight_layout()
 plt.show()

 # -----------------------------------------------------------------------
 # 3. Consecutive Candle Count Distribution (bar chart)
 # -----------------------------------------------------------------------
 fig3, ax3 = plt.subplots(figsize=(7, 4))
 consec_vals = trades_df['consecutive_at_signal'].dropna().astype(int)
 counts = consec_vals.value_counts().sort_index()
 bars = ax3.bar(counts.index, counts.values, color='steelblue', edgecolor='white')
 ax3.bar_label(bars, padding=2)
 ax3.set_title('Consecutive Red Candle Count at Signal Dates')
 ax3.set_xlabel('Consecutive Count')
 ax3.set_ylabel('Number of Trades')
 ax3.set_xticks(counts.index)
 plt.tight_layout()
 plt.show()

 # -----------------------------------------------------------------------
 # 4. Walk-Forward Heatmap (8 windows × 4 metrics)
 # -----------------------------------------------------------------------
 wf_heat = wf_results[['window', 'pf', 'sharpe', 'win_rate', 'max_drawdown', 'composite']].copy()
 wf_heat = wf_heat.set_index('window')
 wf_heat.index = wf_heat.index.astype(str)
 wf_heat.columns = ['Profit Factor', 'Sharpe', 'Win Rate', 'Max Drawdown', 'Composite']

 fig4, ax4 = plt.subplots(figsize=(10, 4))
 mask = wf_heat.isna()
 sns.heatmap(wf_heat.T, annot=True, fmt='.3f', cmap='RdYlGn',
             linewidths=0.5, ax=ax4, mask=mask.T,
             cbar_kws={'label': 'Score'})
 ax4.set_title('Walk-Forward OOS Performance (8 Windows)')
 ax4.set_xlabel('OOS Year')
 ax4.set_ylabel('Metric')
 plt.tight_layout()
 plt.show()

 # -----------------------------------------------------------------------
 # 5. Monte Carlo Histogram with Observed Composite Marked
 # -----------------------------------------------------------------------
 fig5, ax5 = plt.subplots(figsize=(9, 4))
 obs = is_metrics['composite']
 ax5.hist(mc_null_dist, bins=40, color='lightsteelblue', edgecolor='white',
          alpha=0.9, label='Null distribution (500 permutations)')
 ax5.axvline(obs, color='red', linewidth=2.5,
             label=f'Observed composite = {obs:.4f}  (p = {mc_p_value:.3f})')
 ax5.set_title('Monte Carlo Permutation Test — H₁ Null Distribution')
 ax5.set_xlabel('Composite Score')
 ax5.set_ylabel('Frequency')
 ax5.legend()
 plt.tight_layout()
 plt.show()

 # -----------------------------------------------------------------------
 # 6. Win Rate by VIX Regime Bar Chart
 # -----------------------------------------------------------------------
 fig6, ax6 = plt.subplots(figsize=(7, 4))
 regime_order = ['Low', 'Medium', 'High']
 wr_by_regime = (
     trades_df.groupby('vix_regime_at_signal')['net_return']
     .apply(lambda x: (x > 0).mean())
     .reindex(regime_order)
 )
 n_by_regime = trades_df['vix_regime_at_signal'].value_counts().reindex(regime_order)
 colors = ['#2196F3', '#FF9800', '#F44336']
 bars6 = ax6.bar(regime_order, wr_by_regime.values * 100,
                  color=colors, edgecolor='white', width=0.5)
 for bar, n in zip(bars6, n_by_regime.values):
     ax6.text(bar.get_x() + bar.get_width() / 2,
              bar.get_height() + 0.5,
              f'n={n}', ha='center', va='bottom', fontsize=10)
 ax6.axhline(50, color='black', linewidth=1, linestyle='--', label='50% baseline')
 ax6.set_title('Win Rate by VIX Regime at Signal Date')
 ax6.set_xlabel('VIX Regime')
 ax6.set_ylabel('Win Rate (%)')
 ax6.set_ylim(0, 100)
 ax6.legend()
 plt.tight_layout()
 plt.show()

 print("\\n✅ All 6 core visualisations rendered.")"""


# ---------------------------------------------------------------------------
# Patch the notebook
# ---------------------------------------------------------------------------

def patch():
    with open(NOTEBOOK_PATH) as f:
        nb = json.load(f)

    cells = nb["cells"]

    # Find insertion points by scanning cell content
    def find_cell(marker: str, start: int = 0) -> int:
        for i in range(start, len(cells)):
            src = "".join(cells[i]["source"])
            if marker in src:
                return i
        return -1

    # 1. Insert portfolio simulation AFTER the Feature Engineering code cell (index 9)
    feature_code_idx = find_cell("Initialize feature pipeline")
    assert feature_code_idx >= 0, "Feature pipeline cell not found"
    insert_after = feature_code_idx

    new_sim_cells = [
        new_markdown_cell(MD_SIM),
        new_code_cell(CODE_SIM_IMPORTS),
        new_code_cell(CODE_SIM_IS),
        new_code_cell(CODE_SIM_OOS),
    ]
    for j, cell in enumerate(new_sim_cells):
        cells.insert(insert_after + 1 + j, cell)

    # 2. Insert statistical validation BEFORE the §10 Visualizations markdown
    viz_md_idx = find_cell("## 10. Visualizations", insert_after)
    assert viz_md_idx >= 0, "§10 Visualizations markdown not found"

    new_stats_cells = [
        new_markdown_cell(MD_STATS),
        new_code_cell(CODE_MONTE_CARLO),
        new_code_cell(CODE_WALKFORWARD),
        new_code_cell(CODE_H2_ZTEST),
    ]
    for j, cell in enumerate(new_stats_cells):
        cells.insert(viz_md_idx + j, cell)

    # 3. Replace §10 Visualizations markdown + code with updated versions
    viz_md_idx2 = find_cell("## 10. Visualizations", viz_md_idx + len(new_stats_cells))
    assert viz_md_idx2 >= 0, "§10 Visualizations markdown (2nd pass) not found"

    cells[viz_md_idx2] = new_markdown_cell(MD_VIZ)
    # Replace the code cell immediately after
    if viz_md_idx2 + 1 < len(cells) and cells[viz_md_idx2 + 1]["cell_type"] == "code":
        cells[viz_md_idx2 + 1] = new_code_cell(CODE_VIZ)
    else:
        cells.insert(viz_md_idx2 + 1, new_code_cell(CODE_VIZ))

    nb["cells"] = cells

    # Write back
    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"✅ Patched {NOTEBOOK_PATH}")
    print(f"   Total cells now: {len(cells)}")
    new_sections = [
        "§5 Portfolio Simulation (IS + OOS runs)",
        "§8a Statistical Validation (Monte Carlo, WalkForward, H₂ z-test)",
        "§10 Core Visualisations (6 charts)",
    ]
    for s in new_sections:
        print(f"   + {s}")


if __name__ == "__main__":
    patch()
