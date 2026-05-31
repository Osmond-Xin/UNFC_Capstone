import unittest
import os
import pandas as pd
import numpy as np
from tools.edge_or_beta.engine import evaluate_rule
from tools.edge_or_beta.verdict import resolve_verdict
from tools.edge_or_beta.decompose import calculate_capm_decomposition

class TestEdgeOrBeta(unittest.TestCase):
    def setUp(self):
        self.cache_dir = "cache"
        self.rule_id = "capstone_expiry_rule"
        self.params = {"rsi_threshold": 30, "min_consecutive": 3}

    def test_evaluate_rule_runs_and_returns_valid_structure(self):
        """Test that evaluating a rule returns the correct ResultBundle fields."""
        # Use small n_null for fast test execution
        bundle = evaluate_rule(
            rule_id=self.rule_id,
            params=self.params,
            start="2020-01-01",
            end="2022-12-31",
            n_null=10,
            seed=42,
            cache_dir=self.cache_dir
        )
        
        self.assertEqual(bundle["schema_version"], "1.0")
        self.assertEqual(bundle["rule"]["id"], self.rule_id)
        self.assertIn("strategy", bundle)
        self.assertIn("benchmarks", bundle)
        self.assertIn("decomposition", bundle)
        self.assertIn("verdict", bundle)
        
        # Verify specific inner fields
        self.assertIn("cagr", bundle["strategy"])
        self.assertIn("sharpe", bundle["strategy"])
        self.assertIn("win_rate", bundle["strategy"])
        self.assertIn("max_drawdown", bundle["strategy"])
        self.assertIn("equity_curve", bundle["strategy"])
        
        # Verify benchmarks
        self.assertIn("random_stock_same_dates", bundle["benchmarks"])
        self.assertIn("random_etf_timing", bundle["benchmarks"])
        self.assertIn("buy_hold_spy", bundle["benchmarks"])
        
        self.assertEqual(len(bundle["benchmarks"]["random_stock_same_dates"]["distribution"]), 10)

    def test_reproducibility_with_seed(self):
        """Test that identical seeds yield identical Monte Carlo distributions."""
        bundle1 = evaluate_rule(
            rule_id=self.rule_id,
            params=self.params,
            start="2020-01-01",
            end="2022-12-31",
            n_null=10,
            seed=42,
            cache_dir=self.cache_dir
        )
        bundle2 = evaluate_rule(
            rule_id=self.rule_id,
            params=self.params,
            start="2020-01-01",
            end="2022-12-31",
            n_null=10,
            seed=42,
            cache_dir=self.cache_dir
        )
        
        dist1 = bundle1["benchmarks"]["random_stock_same_dates"]["distribution"]
        dist2 = bundle2["benchmarks"]["random_stock_same_dates"]["distribution"]
        self.assertEqual(dist1, dist2)

    def test_spy_coverage_guard(self):
        """Engine must fail loudly when SPY data does not cover the requested window."""
        # SPY cache starts 2018-01-02, so a 2015 start has no SPY baseline coverage.
        with self.assertRaises(ValueError):
            evaluate_rule(
                rule_id=self.rule_id,
                params=self.params,
                start="2015-01-01",
                end="2017-12-31",
                n_null=10,
                seed=42,
                cache_dir=self.cache_dir,
            )

    def test_verdict_resolution(self):
        """Test the verdict mapping logic for various metric inputs."""
        # Insufficient sample
        v1 = resolve_verdict(
            trade_count=5,
            p_select=0.01,
            p_timing=0.01,
            strategy_cagr=0.15,
            spy_cagr=0.10,
            alpha_t=3.0,
            alpha_annualized=0.05,
            beta_share=0.5
        )
        self.assertEqual(v1["status"], "insufficient_sample")

        # Candidate edge
        v2 = resolve_verdict(
            trade_count=20,
            p_select=0.01,
            p_timing=0.01,
            strategy_cagr=0.20,
            spy_cagr=0.10,
            alpha_t=3.0,
            alpha_annualized=0.08,
            beta_share=0.3
        )
        self.assertEqual(v2["status"], "candidate_edge_needs_validation")

        # Mostly beta
        v3 = resolve_verdict(
            trade_count=20,
            p_select=0.06,
            p_timing=0.07,
            strategy_cagr=0.11,
            spy_cagr=0.10,
            alpha_t=1.0,
            alpha_annualized=0.01,
            beta_share=0.85
        )
        self.assertEqual(v3["status"], "mostly_beta")

    def test_capm_regression(self):
        """Test CAPM decomposition OLS returns correct fields."""
        strat = [0.02, -0.01, 0.03, 0.01, -0.02, 0.04]
        spy = [0.01, -0.02, 0.02, 0.00, -0.01, 0.03]
        decomp = calculate_capm_decomposition(strat, spy, 0.12, 0.10, 6)
        
        self.assertIn("beta", decomp)
        self.assertIn("alpha_annualized", decomp)
        self.assertIn("alpha_t", decomp)
        self.assertIn("alpha_p", decomp)
        self.assertIn("beta_share", decomp)

if __name__ == "__main__":
    unittest.main()
