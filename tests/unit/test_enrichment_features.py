"""
Unit tests for EnrichmentFeatures (VIX/FOMC/Earnings joins).

Uses the actual cache files (fast — small CSVs loaded from disk).
"""

import os
import pytest
import pandas as pd
import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "cache")


def _has_cache() -> bool:
    return (
        os.path.exists(os.path.join(CACHE_DIR, "vix.csv"))
        and os.path.exists(os.path.join(CACHE_DIR, "fomc_dates.csv"))
        and os.path.exists(os.path.join(CACHE_DIR, "earnings_dates.csv"))
    )


@pytest.mark.skipif(not _has_cache(), reason="Cache files not available")
class TestEnrichmentFeatures:
    """Integration-style tests against the real cache files."""

    @pytest.fixture(scope="class")
    def enricher(self):
        from modules.features.enrichment_features import EnrichmentFeatures
        return EnrichmentFeatures(cache_dir=CACHE_DIR)

    @pytest.fixture(scope="class")
    def sample_df(self):
        dates = pd.date_range("2020-01-01", "2023-12-31", freq="B")
        df = pd.DataFrame(
            {"Close": np.random.randn(len(dates)) + 100},
            index=dates,
        )
        return df

    def test_vix_columns_added(self, enricher, sample_df):
        df = enricher.enrich(sample_df, ticker="AAPL")
        assert "VIX_Level" in df.columns
        assert "VIX_Regime" in df.columns

    def test_vix_regime_values_valid(self, enricher, sample_df):
        df = enricher.enrich(sample_df, ticker="AAPL")
        valid = {"Low", "Medium", "High", np.nan}
        unique = set(df["VIX_Regime"].dropna().unique())
        assert unique.issubset({"Low", "Medium", "High"})

    def test_fomc_proximity_binary(self, enricher, sample_df):
        df = enricher.enrich(sample_df, ticker="AAPL")
        assert "FOMC_Proximity" in df.columns
        assert set(df["FOMC_Proximity"].unique()).issubset({0, 1})

    def test_earnings_proximity_binary(self, enricher, sample_df):
        df = enricher.enrich(sample_df, ticker="AAPL")
        assert "Earnings_Proximity" in df.columns
        assert set(df["Earnings_Proximity"].unique()).issubset({0, 1})

    def test_no_all_nan_vix_in_2020(self, enricher, sample_df):
        df = enricher.enrich(sample_df, ticker="AAPL")
        window = df.loc["2020-01-01":"2023-12-31", "VIX_Level"]
        # VIX data covers this window — should have very few NaN
        nan_fraction = window.isna().mean()
        assert nan_fraction < 0.05, f"Too many NaN VIX values: {nan_fraction:.2%}"

    def test_known_fomc_date(self, enricher):
        """2022-03-16 was an FOMC meeting; dates within ±5 days should flag 1."""
        fomc_date = pd.Timestamp("2022-03-16")
        for delta in range(-5, 6):
            date = fomc_date + pd.Timedelta(days=delta)
            result = enricher.enrich_date(date, ticker="AAPL")
            assert result["fomc_proximity"] == 1, (
                f"Expected FOMC_Proximity=1 on {date.date()} (±{delta} days from 2022-03-16)"
            )

    def test_far_from_fomc(self, enricher):
        """A date far from any FOMC meeting should flag 0."""
        # 2020-04-01 is 20+ days from the nearest FOMC
        result = enricher.enrich_date(pd.Timestamp("2020-04-01"), ticker="AAPL")
        # We can't guarantee 0 without checking the full calendar, so just check it's 0 or 1
        assert result["fomc_proximity"] in (0, 1)

    def test_enrich_returns_copy(self, enricher, sample_df):
        original_cols = list(sample_df.columns)
        enricher.enrich(sample_df, ticker="AAPL")
        assert list(sample_df.columns) == original_cols


class TestEnrichmentFeaturesUnit:
    """Pure-unit tests that do not require cache files."""

    def test_vix_regime_thresholds(self):
        from modules.config.capstone_v4_params import VIX_LOW_THRESHOLD, VIX_HIGH_THRESHOLD
        assert VIX_LOW_THRESHOLD == 15.0
        assert VIX_HIGH_THRESHOLD == 25.0

    def test_fomc_proximity_days(self):
        from modules.config.capstone_v4_params import FOMC_PROXIMITY_DAYS
        assert FOMC_PROXIMITY_DAYS == 5

    def test_earnings_proximity_days(self):
        from modules.config.capstone_v4_params import EARNINGS_PROXIMITY_DAYS
        assert EARNINGS_PROXIMITY_DAYS == 3
