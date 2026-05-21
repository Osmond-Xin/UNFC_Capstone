"""
Portfolio Simulator — SPY Expiry Signal Backtest

Implements the core capstone backtest described in project_purpose.md §7.3–7.4
and architecture.md §10 P0.

Entry rule:
    For each monthly expiry date, scan every S&P 500 ticker on the *signal date*
    (the trading day immediately before the expiry). Apply RSIReversalStrategy with
    the v4.1 parameters. Select up to MAX_ENTRIES_PER_EXPIRY tickers ranked by
    lowest RSI (most oversold). Enforce MAX_CONCURRENT_POSITIONS across all open
    trades at that moment.

Exit rule:
    Default — hold for `hold_days` trading days from entry, exit at close.
    Optional — apply exit_params (fixed_pct, trailing_stop, rsi_exit, hold_only).

Returns:
    trades_df — one row per completed trade with the columns required by the
    cross-stream interface contract:
        ticker, signal_date, entry_date, exit_date,
        entry_price, exit_price, gross_return, net_return,
        rsi_at_signal, consecutive_at_signal, vix_regime_at_signal

Usage:
    from modules.evaluation.portfolio_simulator import run_simulation

    trades_df = run_simulation(
        rsi_threshold=22,
        min_consecutive=3,
        hold_days=6,
        start_date='2015-01-01',
        end_date='2025-06-30',
    )
"""

import os
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from modules.config.capstone_v4_params import (
    COMMISSION,
    MAX_ENTRIES_PER_EXPIRY,
    MAX_CONCURRENT_POSITIONS,
)
from modules.data.data_loader import DataLoader
from modules.data.expiry_calculator import ExpiryCalculator
from modules.features.feature_pipeline import FeaturePipeline
from modules.features.technical_indicators import TechnicalIndicators
from modules.models.pattern_models import RSIReversalStrategy


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_simulation(
    rsi_threshold: int,
    min_consecutive: int,
    hold_days: int,
    start_date: str,
    end_date: str,
    exit_params: Optional[Dict] = None,
    cache_dir: str = "cache",
    verbose: bool = False,
    ticker_universe: Optional[list] = None,
) -> pd.DataFrame:
    """
    Run the full portfolio simulation and return trades_df.

    Args:
        rsi_threshold:    RSI must be strictly below this value to trigger a signal.
        min_consecutive:  Minimum consecutive red candles (close-to-close) required.
        hold_days:        Number of *trading days* to hold a position after entry.
        start_date:       ISO date string — include expiry dates on or after this.
        end_date:         ISO date string — include expiry dates on or before this.
        exit_params:      Optional dict controlling early exit logic.  Supported keys:
                              exit_type        — 'fixed_pct' | 'trailing_stop' |
                                                 'rsi_exit' | 'hold_only'
                              take_profit_pct  — exit if return exceeds this (0–1)
                              trailing_stop_pct — exit if drawdown from peak exceeds this
                              rsi_exit_threshold — exit when RSI rises above this
                          If None, falls back to fixed hold_days only.
        cache_dir:        Root cache directory (default 'cache').
        verbose:          Print per-expiry progress if True.
        ticker_universe:  Optional list of tickers to restrict the simulation to.
                          If None, uses the full sp500_list.csv universe.

    Returns:
        pd.DataFrame with columns:
            ticker, signal_date, entry_date, exit_date,
            entry_price, exit_price, gross_return, net_return,
            rsi_at_signal, consecutive_at_signal, vix_regime_at_signal
    """
    # ------------------------------------------------------------------
    # 1. Load & prepare data
    # ------------------------------------------------------------------
    loader = DataLoader(
        cache_dir=os.path.join(cache_dir, "constituent_data"),
        auto_update=False,
    )
    tickers = ticker_universe if ticker_universe is not None else loader.get_sp500_tickers()

    pipeline = FeaturePipeline([TechnicalIndicators()])

    all_data: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            df = loader.load_ticker(ticker)
            if df is None or df.empty:
                continue
            df = pipeline.transform(df)
            # Ensure DatetimeIndex is tz-naive and sorted
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.sort_index()
            all_data[ticker] = df
        except Exception:
            pass

    if verbose:
        print(f"Loaded {len(all_data)} tickers.")

    # ------------------------------------------------------------------
    # 2. Load VIX for regime lookup
    # ------------------------------------------------------------------
    vix_df = _load_vix(cache_dir)

    # ------------------------------------------------------------------
    # 3. Build expiry calendar
    # ------------------------------------------------------------------
    expiry_dates = ExpiryCalculator.generate_expiry_dates(start_date, end_date)

    # ------------------------------------------------------------------
    # 4. Strategy instance (stateless — one instance reused for all tickers)
    # ------------------------------------------------------------------
    strategy = RSIReversalStrategy(
        config={"rsi_threshold": rsi_threshold, "min_consecutive": min_consecutive}
    )

    # ------------------------------------------------------------------
    # 5. Main simulation loop
    # ------------------------------------------------------------------
    records: List[Dict] = []
    open_positions: List[Dict] = []  # tracks currently open trades

    for expiry_date in expiry_dates:
        # Close out any positions that exited on or before today
        open_positions = [p for p in open_positions if p["exit_date"] > expiry_date]
        current_open = len(open_positions)

        if current_open >= MAX_CONCURRENT_POSITIONS:
            if verbose:
                print(f"{expiry_date.date()} — positions full ({current_open}), skipping.")
            continue

        slots_available = MAX_CONCURRENT_POSITIONS - current_open

        # Scan each ticker at the signal date (1 trading day before expiry)
        signals: List[Dict] = []

        for ticker, df in all_data.items():
            try:
                sig = strategy.predict(df, expiry_date)
                if sig != 1:
                    continue

                # Retrieve the exact signal-date values used by the strategy
                # lookback_days=1 means the row 1 index-position before expiry
                expiry_loc = _find_loc(df, expiry_date)
                if expiry_loc is None or expiry_loc < 1:
                    continue
                signal_loc = expiry_loc - 1
                signal_date = df.index[signal_loc]

                rsi_val = df.iloc[signal_loc].get("RSI", np.nan)
                consec_val = df.iloc[signal_loc].get("Consecutive_Count", np.nan)

                if pd.isna(rsi_val):
                    continue

                signals.append(
                    {
                        "ticker": ticker,
                        "signal_date": signal_date,
                        "rsi_at_signal": rsi_val,
                        "consecutive_at_signal": consec_val,
                        "expiry_date": expiry_date,
                    }
                )
            except Exception:
                continue

        if not signals:
            continue

        # Select top-N by lowest RSI (most oversold), honouring capacity
        n_pick = min(MAX_ENTRIES_PER_EXPIRY, slots_available, len(signals))
        top_signals = sorted(signals, key=lambda x: x["rsi_at_signal"])[:n_pick]

        for sig in top_signals:
            ticker = sig["ticker"]
            df = all_data[ticker]
            entry_date, entry_price = _get_entry(df, sig["expiry_date"])
            if entry_date is None:
                continue

            exit_date, exit_price = _get_exit(
                df, entry_date, hold_days, exit_params
            )
            if exit_date is None:
                continue

            gross_return = (exit_price - entry_price) / entry_price
            net_return = gross_return - COMMISSION

            vix_regime = _lookup_vix_regime(vix_df, sig["signal_date"])

            record = {
                "ticker": ticker,
                "signal_date": sig["signal_date"],
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "net_return": net_return,
                "rsi_at_signal": sig["rsi_at_signal"],
                "consecutive_at_signal": sig["consecutive_at_signal"],
                "vix_regime_at_signal": vix_regime,
            }
            records.append(record)
            open_positions.append({"exit_date": exit_date})

        if verbose:
            print(
                f"{expiry_date.date()} — {len(top_signals)} signals, "
                f"{len(records)} total trades so far."
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "ticker", "signal_date", "entry_date", "exit_date",
                "entry_price", "exit_price", "gross_return", "net_return",
                "rsi_at_signal", "consecutive_at_signal", "vix_regime_at_signal",
            ]
        )

    trades_df = pd.DataFrame(records)
    trades_df["signal_date"] = pd.to_datetime(trades_df["signal_date"])
    trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
    trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])
    trades_df = trades_df.sort_values("entry_date").reset_index(drop=True)
    return trades_df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _find_loc(df: pd.DataFrame, date: pd.Timestamp) -> Optional[int]:
    """Return integer iloc for date, or the iloc of the nearest earlier date."""
    norm = pd.Timestamp(date).tz_localize(None)
    candidates = df.index[df.index <= norm]
    if len(candidates) == 0:
        return None
    return df.index.get_loc(candidates[-1])


def _get_entry(
    df: pd.DataFrame, expiry_date: pd.Timestamp
) -> tuple:
    """
    Entry is at the OPEN of the expiry_date (= T+1 relative to signal date).
    Returns (entry_date, entry_price) or (None, None).
    """
    loc = _find_loc(df, expiry_date)
    if loc is None:
        return None, None
    row = df.iloc[loc]
    entry_date = df.index[loc]
    # We need the open to be a valid number
    open_price = row.get("Open", np.nan)
    if pd.isna(open_price) or open_price <= 0:
        return None, None
    return entry_date, float(open_price)


def _get_exit(
    df: pd.DataFrame,
    entry_date: pd.Timestamp,
    hold_days: int,
    exit_params: Optional[Dict],
) -> tuple:
    """
    Determine exit date and price.

    With no exit_params (or exit_type='hold_only'), exit at close of
    entry_date + hold_days trading days.

    With exit_params, scan each day in the hold window and trigger early exit
    if the relevant condition is met.

    Returns (exit_date, exit_price) or (None, None).
    """
    entry_loc = _find_loc(df, entry_date)
    if entry_loc is None:
        return None, None

    entry_open = df.iloc[entry_loc].get("Open", np.nan)
    if pd.isna(entry_open) or entry_open <= 0:
        return None, None

    # Build the hold window (trading days starting from entry_loc)
    end_loc = min(entry_loc + hold_days, len(df) - 1)
    if end_loc <= entry_loc:
        return None, None

    exit_type = "hold_only"
    if exit_params:
        exit_type = exit_params.get("exit_type", "hold_only")

    if exit_type == "hold_only" or exit_params is None:
        row = df.iloc[end_loc]
        close_price = row.get("Close", np.nan)
        if pd.isna(close_price) or close_price <= 0:
            return None, None
        return df.index[end_loc], float(close_price)

    if exit_type == "fixed_pct":
        tp_pct = exit_params.get("take_profit_pct", 0.05)
        for i in range(entry_loc + 1, end_loc + 1):
            row = df.iloc[i]
            high = row.get("High", np.nan)
            close = row.get("Close", np.nan)
            if pd.isna(high) or pd.isna(close):
                continue
            if (high - entry_open) / entry_open >= tp_pct:
                return df.index[i], float(high)
        # Hold until end if take-profit never triggered
        close_price = df.iloc[end_loc].get("Close", np.nan)
        if pd.isna(close_price):
            return None, None
        return df.index[end_loc], float(close_price)

    if exit_type == "trailing_stop":
        ts_pct = exit_params.get("trailing_stop_pct", 0.03)
        peak = entry_open
        for i in range(entry_loc + 1, end_loc + 1):
            row = df.iloc[i]
            high = row.get("High", np.nan)
            low = row.get("Low", np.nan)
            close = row.get("Close", np.nan)
            if pd.isna(high) or pd.isna(low) or pd.isna(close):
                continue
            peak = max(peak, high)
            if (peak - low) / peak >= ts_pct:
                return df.index[i], float(low)
        close_price = df.iloc[end_loc].get("Close", np.nan)
        if pd.isna(close_price):
            return None, None
        return df.index[end_loc], float(close_price)

    if exit_type == "rsi_exit":
        rsi_thr = exit_params.get("rsi_exit_threshold", 50)
        for i in range(entry_loc + 1, end_loc + 1):
            row = df.iloc[i]
            rsi = row.get("RSI", np.nan)
            close = row.get("Close", np.nan)
            if pd.isna(rsi) or pd.isna(close):
                continue
            if rsi >= rsi_thr:
                return df.index[i], float(close)
        close_price = df.iloc[end_loc].get("Close", np.nan)
        if pd.isna(close_price):
            return None, None
        return df.index[end_loc], float(close_price)

    # Unknown exit type — fall back to hold
    close_price = df.iloc[end_loc].get("Close", np.nan)
    if pd.isna(close_price):
        return None, None
    return df.index[end_loc], float(close_price)


def _load_vix(cache_dir: str) -> pd.DataFrame:
    path = os.path.join(cache_dir, "vix.csv")
    vix = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    vix.index = pd.to_datetime(vix.index).tz_localize(None).normalize()
    vix.columns = ["VIX"]
    return vix


def _lookup_vix_regime(vix_df: pd.DataFrame, date: pd.Timestamp) -> Optional[str]:
    """Return VIX regime string for the given date, or None if not available."""
    from modules.config.capstone_v4_params import VIX_LOW_THRESHOLD, VIX_HIGH_THRESHOLD

    norm = pd.Timestamp(date).tz_localize(None).normalize()
    candidates = vix_df.index[vix_df.index <= norm]
    if len(candidates) == 0 or (norm - candidates[-1]).days > 5:
        return None
    vix_val = float(vix_df.loc[candidates[-1], "VIX"])
    if vix_val < VIX_LOW_THRESHOLD:
        return "Low"
    if vix_val <= VIX_HIGH_THRESHOLD:
        return "Medium"
    return "High"
