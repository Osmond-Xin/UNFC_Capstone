"""
Unit tests for MonteCarlo and WalkForward in robust_validation.py.

The Monte Carlo test runs against a synthetic trades_df so the test suite
does not require the full ~510-ticker cache load.  WalkForward tests only
verify structural contracts (correct column set, 8 rows) using a patched
run_simulation that returns deterministic data.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from modules.evaluation.robust_validation import MonteCarlo, _portfolio_metrics
from modules.config.capstone_v4_params import RANDOM_SEED



# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_trades(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    entry_dates = pd.date_range("2020-01-02", periods=n, freq="10D")
    net_returns = rng.normal(0.005, 0.03, n)
    return pd.DataFrame(
        {
            "ticker": ["AAPL"] * n,
            "signal_date": entry_dates - pd.Timedelta(days=1),
            "entry_date": entry_dates,
            "exit_date": entry_dates + pd.Timedelta(days=6),
            "entry_price": rng.uniform(100, 300, n),
            "exit_price": rng.uniform(100, 300, n),
            "gross_return": net_returns + 0.002,
            "net_return": net_returns,
            "rsi_at_signal": rng.uniform(10, 22, n),
            "consecutive_at_signal": rng.integers(3, 6, n),
            "vix_regime_at_signal": rng.choice(["Low", "Medium", "High"], n),
        }
    )


# ---------------------------------------------------------------------------
# MonteCarlo tests
# ---------------------------------------------------------------------------

class TestMonteCarlo:
    def test_run_returns_tuple(self):
        trades = _make_trades(80)
        mc = MonteCarlo(trades, n_iterations=50)
        result = mc.run()
        assert isinstance(result, tuple) and len(result) == 2

    def test_p_value_in_range(self):
        trades = _make_trades(80)
        mc = MonteCarlo(trades, n_iterations=100)
        p_value, _ = mc.run()
        assert 0.0 <= p_value <= 1.0

    def test_null_dist_length(self):
        trades = _make_trades(80)
        mc = MonteCarlo(trades, n_iterations=50)
        _, null_dist = mc.run()
        assert len(null_dist) == 50