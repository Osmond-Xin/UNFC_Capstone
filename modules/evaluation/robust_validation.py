"""
Robust Validation — Monte Carlo Permutation Test & Walk-Forward Analysis

Implements the two primary robustness tests required by the capstone
(architecture.md §10 P0) plus parameter sensitivity analysis (P1):

    MonteCarlo     — 500-iteration permutation test on an existing trades_df.
                     Does NOT re-run run_simulation(); it permutes net_return
                     values to break the signal-return link while preserving
                     the return distribution.

    WalkForward    — 8-window rolling IS/OOS analysis using run_simulation().
                     Window boundaries: 3-year IS / 1-year OOS, sliding 1 year
                     from OOS year 2018 through 2025.

    sensitivity    — standalone function; perturbs each of rsi_threshold,
                     min_consecutive, hold_days by ±10% and ±20% and reports
                     the composite score at each of 12 perturbation points.

Usage:
    from modules.evaluation.robust_validation import MonteCarlo, WalkForward
    from modules.evaluation.robust_validation import parameter_sensitivity

    mc = MonteCarlo(trades_df)
    p_value, null_dist = mc.run()

    wf = WalkForward()
    wf_results = wf.run()

    sensitivity_df = parameter_sensitivity(trades_df_full)
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from modules.config.capstone_v4_params import (
    COMPOSITE_WEIGHTS,
    HOLD_DAYS,
    MIN_CONSECUTIVE,
    MONTE_CARLO_ITERATIONS,
    RANDOM_SEED,
    RSI_THRESHOLD,
    WALKFORWARD_IS_YEARS,
    WALKFORWARD_OOS_YEARS,
)
from modules.evaluation.metrics import (
    calculate_composite_score,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
)


# ---------------------------------------------------------------------------
# Shared metric helpers
# ---------------------------------------------------------------------------

def _portfolio_metrics(net_returns: np.ndarray) -> dict:
    """Compute Sharpe, Win Rate, Profit Factor, Max Drawdown from net_returns."""
    if len(net_returns) == 0:
        return {"sharpe": np.nan, "win_rate": np.nan, "pf": np.nan, "max_dd": np.nan}

    wins = net_returns[net_returns > 0]
    losses = net_returns[net_returns < 0]
    win_rate = len(wins) / len(net_returns)
    gross_profit = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    # Sharpe: treat each trade return as one period (scaled to 252-period equivalent)
    sharpe = float(calculate_sharpe_ratio(net_returns * 100, periods_per_year=252))

    # Max drawdown: cumulative equity curve approach
    cumulative = (1 + net_returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_dd = float(abs(drawdowns.min())) if len(drawdowns) > 0 else 0.0

    return {"sharpe": sharpe, "win_rate": win_rate, "pf": pf, "max_dd": max_dd}


def _composite_from_trades(trades: pd.DataFrame) -> float:
    """Compute composite score from a trades_df slice."""
    if trades.empty:
        return 0.0
    returns = trades["net_return"].values
    m = _portfolio_metrics(returns)
    if any(np.isnan(v) for v in m.values()):
        return 0.0
    return calculate_composite_score(m["pf"], m["sharpe"], m["win_rate"], m["max_dd"])


# ---------------------------------------------------------------------------
# Monte Carlo permutation test
# ---------------------------------------------------------------------------

class MonteCarlo:
    """
    Permutation-based null hypothesis test for portfolio performance.

    The null hypothesis (H₀) is that the strategy's returns are no better
    than a random assignment of the same returns to the same trade dates.
    Each iteration shuffles the net_return column of trades_df (breaking the
    signal–return link) and recomputes the composite score.  The p-value is
    the fraction of null scores ≥ the observed composite.

    Note: seed is set ONCE before the first shuffle; subsequent iterations
    use the continuing RNG state to keep the test valid.
    """

    def __init__(self, trades_df: pd.DataFrame, n_iterations: int = MONTE_CARLO_ITERATIONS):
        """
        Args:
            trades_df:    Complete trades DataFrame produced by run_simulation().
            n_iterations: Number of permutations (default 500).
        """
        if "net_return" not in trades_df.columns:
            raise ValueError("trades_df must contain a 'net_return' column.")
        self.trades_df = trades_df.copy()
        self.n_iterations = n_iterations
        self._null_dist: Optional[np.ndarray] = None

    def run(self) -> Tuple[float, np.ndarray]:
        """
        Execute the Monte Carlo permutation test.

        Returns:
            p_value:   Fraction of null composite scores ≥ observed composite.
            null_dist: NumPy array of length n_iterations with null scores.
        """
        observed = _composite_from_trades(self.trades_df)
        returns = self.trades_df["net_return"].values.copy()

        rng = np.random.default_rng(RANDOM_SEED)  # seed set once here

        null_scores = np.empty(self.n_iterations)
        for i in range(self.n_iterations):
            permuted = rng.permutation(returns)
            m = _portfolio_metrics(permuted)
            if any(np.isnan(v) for v in m.values()):
                null_scores[i] = 0.0
            else:
                null_scores[i] = calculate_composite_score(
                    m["pf"], m["sharpe"], m["win_rate"], m["max_dd"]
                )

        p_value = float(np.sum(null_scores >= observed) / self.n_iterations)
        self._null_dist = null_scores
        return p_value, null_scores

    @property
    def null_distribution(self) -> Optional[np.ndarray]:
        """Null distribution array after run() has been called, else None."""
        return self._null_dist


# ---------------------------------------------------------------------------
# Walk-forward analysis
# ---------------------------------------------------------------------------

class WalkForward:
    """
    8-window rolling in-sample / out-of-sample validation.

    Each window has a 3-year IS period and a 1-year OOS period.  The OOS
    years run from 2018 through 2025.  run_simulation() is called once per
    window with the full date range; the composite score is then computed
    on the OOS portion of the returned trades_df.
    """

    def __init__(
        self,
        rsi_threshold: int = RSI_THRESHOLD,
        min_consecutive: int = MIN_CONSECUTIVE,
        hold_days: int = HOLD_DAYS,
        cache_dir: str = "cache",
    ):
        self.rsi_threshold = rsi_threshold
        self.min_consecutive = min_consecutive
        self.hold_days = hold_days
        self.cache_dir = cache_dir

    def run(self, verbose: bool = False) -> pd.DataFrame:
        """
        Run all 8 walk-forward windows.

        Returns:
            DataFrame with exactly 8 rows and columns:
                window, is_start, is_end, oos_start, oos_end,
                pf, sharpe, win_rate, max_drawdown, composite
        """
        from modules.evaluation.portfolio_simulator import run_simulation

        rows = []
        for oos_year in WALKFORWARD_OOS_YEARS:
            is_start = f"{oos_year - WALKFORWARD_IS_YEARS}-01-01"
            is_end = f"{oos_year - 1}-12-31"
            oos_start = f"{oos_year}-01-01"
            oos_end = f"{oos_year}-12-31"

            if verbose:
                print(f"Window OOS {oos_year}: IS {is_start}–{is_end}, OOS {oos_start}–{oos_end}")

            try:
                trades = run_simulation(
                    rsi_threshold=self.rsi_threshold,
                    min_consecutive=self.min_consecutive,
                    hold_days=self.hold_days,
                    start_date=is_start,
                    end_date=oos_end,
                    cache_dir=self.cache_dir,
                    verbose=False,
                )
            except Exception as exc:
                warnings.warn(f"Window OOS {oos_year} failed: {exc}")
                trades = pd.DataFrame(columns=["net_return", "entry_date"])

            # Score only the OOS slice
            oos_trades = trades[
                (trades["entry_date"] >= pd.Timestamp(oos_start))
                & (trades["entry_date"] <= pd.Timestamp(oos_end))
            ] if not trades.empty else trades

            if oos_trades.empty:
                rows.append(
                    {
                        "window": oos_year,
                        "is_start": is_start,
                        "is_end": is_end,
                        "oos_start": oos_start,
                        "oos_end": oos_end,
                        "pf": np.nan,
                        "sharpe": np.nan,
                        "win_rate": np.nan,
                        "max_drawdown": np.nan,
                        "composite": np.nan,
                    }
                )
                continue

            m = _portfolio_metrics(oos_trades["net_return"].values)
            composite = (
                calculate_composite_score(m["pf"], m["sharpe"], m["win_rate"], m["max_dd"])
                if not any(np.isnan(v) for v in m.values())
                else np.nan
            )
            rows.append(
                {
                    "window": oos_year,
                    "is_start": is_start,
                    "is_end": is_end,
                    "oos_start": oos_start,
                    "oos_end": oos_end,
                    "pf": m["pf"],
                    "sharpe": m["sharpe"],
                    "win_rate": m["win_rate"],
                    "max_drawdown": m["max_dd"],
                    "composite": composite,
                }
            )

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Parameter sensitivity analysis
# ---------------------------------------------------------------------------

def parameter_sensitivity(
    baseline_trades_df: pd.DataFrame,
    cache_dir: str = "cache",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Perturb each of rsi_threshold, min_consecutive, and hold_days by
    ±10% and ±20% and report the composite score at each of 12 points.

    Pass criterion: composite score degrades < 15% relative to baseline
    at every perturbation level.

    Args:
        baseline_trades_df: trades_df from the canonical run_simulation call
                            (used to compute the baseline composite score).
        cache_dir:          Root cache directory.
        verbose:            Print progress if True.

    Returns:
        DataFrame with columns:
            parameter, perturbation_pct, value,
            pf, sharpe, win_rate, max_drawdown, composite,
            pct_change_from_baseline, passes_threshold
    """
    from modules.evaluation.portfolio_simulator import run_simulation

    baseline_composite = _composite_from_trades(baseline_trades_df)

    perturbations = [-0.20, -0.10, 0.10, 0.20]
    base_params = {
        "rsi_threshold": RSI_THRESHOLD,
        "min_consecutive": MIN_CONSECUTIVE,
        "hold_days": HOLD_DAYS,
    }

    rows = []
    for param_name, base_val in base_params.items():
        for pct in perturbations:
            raw = base_val * (1 + pct)
            # RSI and consecutive values must be integers and ≥ 1
            if param_name in ("rsi_threshold", "min_consecutive", "hold_days"):
                value = max(1, round(raw))
            else:
                value = raw

            kwargs = {k: v for k, v in base_params.items()}
            kwargs[param_name] = value

            if verbose:
                print(f"  {param_name}={value} ({pct:+.0%} of {base_val})")

            try:
                trades = run_simulation(
                    rsi_threshold=kwargs["rsi_threshold"],
                    min_consecutive=kwargs["min_consecutive"],
                    hold_days=kwargs["hold_days"],
                    start_date="2015-01-01",
                    end_date="2025-06-30",
                    cache_dir=cache_dir,
                    verbose=False,
                )
            except Exception as exc:
                warnings.warn(f"Sensitivity run failed ({param_name}={value}): {exc}")
                trades = pd.DataFrame(columns=["net_return"])

            if trades.empty:
                composite = 0.0
                m = {"pf": np.nan, "sharpe": np.nan, "win_rate": np.nan, "max_dd": np.nan}
            else:
                m = _portfolio_metrics(trades["net_return"].values)
                composite = (
                    calculate_composite_score(m["pf"], m["sharpe"], m["win_rate"], m["max_dd"])
                    if not any(np.isnan(v) for v in m.values())
                    else 0.0
                )

            pct_change = (
                (composite - baseline_composite) / baseline_composite
                if baseline_composite != 0
                else np.nan
            )
            rows.append(
                {
                    "parameter": param_name,
                    "perturbation_pct": int(pct * 100),
                    "value": value,
                    "pf": m["pf"],
                    "sharpe": m["sharpe"],
                    "win_rate": m["win_rate"],
                    "max_drawdown": m["max_dd"],
                    "composite": composite,
                    "pct_change_from_baseline": pct_change,
                    "passes_threshold": (
                        bool(pct_change > -0.15) if not np.isnan(pct_change) else False
                    ),
                }
            )

    return pd.DataFrame(rows)
