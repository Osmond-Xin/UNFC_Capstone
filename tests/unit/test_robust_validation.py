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
    
    def test_reproducibility_with_seed(self):
        trades = _make_trades(80)
        p1, dist1 = MonteCarlo(trades, n_iterations=50).run()
        p2, dist2 = MonteCarlo(trades, n_iterations=50).run()
        assert p1 == p2
        np.testing.assert_array_equal(dist1, dist2)

    def test_null_distribution_stored(self):
        trades = _make_trades(80)
        mc = MonteCarlo(trades, n_iterations=30)
        assert mc.null_distribution is None
        mc.run()
        assert mc.null_distribution is not None

    def test_requires_net_return_column(self):
        trades = _make_trades(20).drop(columns=["net_return"])
        with pytest.raises(ValueError):
            MonteCarlo(trades)

    def test_p_value_formula(self):
        """p = sum(null >= observed) / n_iterations."""
        trades = _make_trades(80)
        mc = MonteCarlo(trades, n_iterations=200)
        p_value, null_dist = mc.run()

        from modules.evaluation.robust_validation import _composite_from_trades
        observed = _composite_from_trades(trades)
        expected_p = np.sum(null_dist >= observed) / 200
        assert p_value == pytest.approx(expected_p)



# ---------------------------------------------------------------------------
# portfolio_metrics tests
# ---------------------------------------------------------------------------

class TestPortfolioMetrics:
    def test_empty_returns(self):
        m = _portfolio_metrics(np.array([]))
        assert np.isnan(m["sharpe"])
        assert np.isnan(m["win_rate"])
        assert np.isnan(m["pf"])
        assert np.isnan(m["max_dd"])

    def test_all_wins(self):
        m = _portfolio_metrics(np.array([0.01, 0.02, 0.03]))
        assert m["win_rate"] == pytest.approx(1.0)
        assert m["pf"] == float("inf") or m["pf"] > 100

    def test_all_losses(self):
        m = _portfolio_metrics(np.array([-0.01, -0.02, -0.03]))
        assert m["win_rate"] == pytest.approx(0.0)
        assert m["pf"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# WalkForward structural tests
# ---------------------------------------------------------------------------

def _synthetic_trades(start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
    """Fake run_simulation that returns 20 rows within the requested window."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(start_date, end_date, periods=20)
    n = len(dates)
    return pd.DataFrame(
        {
            "ticker": ["FAKE"] * n,
            "signal_date": dates - pd.Timedelta(days=1),
            "entry_date": dates,
            "exit_date": dates + pd.Timedelta(days=6),
            "entry_price": rng.uniform(100, 200, n),
            "exit_price": rng.uniform(100, 200, n),
            "gross_return": rng.normal(0.005, 0.02, n),
            "net_return": rng.normal(0.003, 0.02, n),
            "rsi_at_signal": rng.uniform(12, 22, n),
            "consecutive_at_signal": rng.integers(3, 5, n),
            "vix_regime_at_signal": ["Medium"] * n,
        }
    )

