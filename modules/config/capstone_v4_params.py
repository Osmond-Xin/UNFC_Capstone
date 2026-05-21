"""
Capstone v4.1 — Frozen Parameter Set

Pre-registered parameters for the SPY Expiry Signal Pipeline capstone.
All notebooks and validation scripts must import from this module instead of
hardcoding values, so that the "pre-registration" claim in the proposal is
honoured in code.

Usage:
    from modules.config.capstone_v4_params import (
        RSI_THRESHOLD, MIN_CONSECUTIVE, HOLD_DAYS,
        COMMISSION, MAX_ENTRIES_PER_EXPIRY,
        MAX_CONCURRENT_POSITIONS, COMPOSITE_WEIGHTS,
        RANDOM_SEED
    )
"""

# --- Strategy signal parameters ---
RSI_THRESHOLD: int = 22          # RSI must be below this to trigger a long signal
MIN_CONSECUTIVE: int = 3         # Minimum consecutive red (close-to-close) candles
HOLD_DAYS: int = 6               # Calendar-day hold period after entry

# --- Portfolio simulation parameters ---
COMMISSION: float = 0.002        # Round-trip commission: 0.2% per trade
MAX_ENTRIES_PER_EXPIRY: int = 3  # Maximum new positions entered per expiry cycle
MAX_CONCURRENT_POSITIONS: int = 15  # Hard cap on open positions at any time

# --- Composite score weights (PF, Sharpe, WinRate, MaxDD) ---
# Formula: min(PF/2,1)×w0 + min(max(S,0)/2,1)×w1 + WR×w2 + (1−min(|MDD|/0.30,1))×w3
COMPOSITE_WEIGHTS: tuple = (0.35, 0.30, 0.20, 0.15)

# --- Reproducibility ---
RANDOM_SEED: int = 42

# --- VIX regime thresholds ---
VIX_LOW_THRESHOLD: float = 15.0
VIX_HIGH_THRESHOLD: float = 25.0

# --- Enrichment proximity windows (calendar days) ---
FOMC_PROXIMITY_DAYS: int = 5     # ±5 days around FOMC meeting date
EARNINGS_PROXIMITY_DAYS: int = 3  # ±3 days around earnings release date

# --- IS / OOS split dates ---
IS_START: str = "2015-01-01"
IS_END: str = "2025-06-30"
OOS_START: str = "2025-07-01"
# OOS_END is determined at runtime from the latest available cache date

# --- Walk-forward window definition ---
# 8 windows of 3-year IS / 1-year OOS (OOS year shown)
WALKFORWARD_OOS_YEARS: tuple = (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025)
WALKFORWARD_IS_YEARS: int = 3    # IS window length in years

# --- Monte Carlo ---
MONTE_CARLO_ITERATIONS: int = 500
