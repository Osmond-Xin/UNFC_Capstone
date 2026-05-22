"""
Unit tests for capstone_v4_params and calculate_composite_score.

Verifies that:
- All required parameter constants are present and correct.
- calculate_composite_score returns the expected value for the reference inputs.
- COMPOSITE_WEIGHTS sum to 1.0.
"""

import pytest
from modules.config.capstone_v4_params import (
    RSI_THRESHOLD,
    MIN_CONSECUTIVE,
    HOLD_DAYS,
    COMMISSION,
    MAX_ENTRIES_PER_EXPIRY,
    MAX_CONCURRENT_POSITIONS,
    COMPOSITE_WEIGHTS,
    RANDOM_SEED,
)
from modules.evaluation.metrics import calculate_composite_score


class TestCapstoneParams:
    def test_rsi_threshold(self):
        assert RSI_THRESHOLD == 22

    def test_min_consecutive(self):
        assert MIN_CONSECUTIVE == 3

    def test_hold_days(self):
        assert HOLD_DAYS == 6

    def test_commission(self):
        assert COMMISSION == pytest.approx(0.002)

    def test_max_entries_per_expiry(self):
        assert MAX_ENTRIES_PER_EXPIRY == 3

    def test_max_concurrent_positions(self):
        assert MAX_CONCURRENT_POSITIONS == 15

    def test_composite_weights_values(self):
        assert COMPOSITE_WEIGHTS == (0.35, 0.30, 0.20, 0.15)

    def test_composite_weights_sum(self):
        assert sum(COMPOSITE_WEIGHTS) == pytest.approx(1.0)

    def test_random_seed(self):
        assert RANDOM_SEED == 42


class TestCompositeScore:
    def test_reference_case(self):
        # From team_review_assignments.md success criteria:
        # calculate_composite_score(1.5, 1.0, 0.6, 0.1) → 0.6325
        # Verification: 0.75×0.35 + 0.50×0.30 + 0.60×0.20 + 0.667×0.15 ≈ 0.6325
        result = calculate_composite_score(1.5, 1.0, 0.6, 0.1)
        assert result == pytest.approx(0.6325, abs=1e-4)

    def test_perfect_score(self):
        # PF=inf (capped at 1.0), Sharpe≥2 (capped at 1.0), WR=1, MDD=0
        result = calculate_composite_score(100.0, 100.0, 1.0, 0.0)
        assert result == pytest.approx(1.0)

    def test_zero_score(self):
        # PF=0, Sharpe≤0, WR=0, MDD≥0.30 (capped at 1.0)
        result = calculate_composite_score(0.0, -5.0, 0.0, 0.30)
        assert result == pytest.approx(0.0)

    def test_negative_sharpe_clamped(self):
        # Sharpe < 0 → clamp to 0 → sharpe component = 0
        result_neg = calculate_composite_score(1.0, -2.0, 0.5, 0.1)
        result_zero = calculate_composite_score(1.0, 0.0, 0.5, 0.1)
        assert result_neg == pytest.approx(result_zero)

    def test_mdd_absolute_value(self):
        # max_drawdown can be supplied as negative or positive — result must be same
        score_pos = calculate_composite_score(1.5, 1.0, 0.6, 0.1)
        score_neg = calculate_composite_score(1.5, 1.0, 0.6, -0.1)
        assert score_pos == pytest.approx(score_neg)

    def test_output_range(self):
        import random
        rng = random.Random(0)
        for _ in range(100):
            pf = rng.uniform(0, 5)
            sharpe = rng.uniform(-3, 4)
            wr = rng.uniform(0, 1)
            mdd = rng.uniform(0, 0.5)
            score = calculate_composite_score(pf, sharpe, wr, mdd)
            assert 0.0 <= score <= 1.0
