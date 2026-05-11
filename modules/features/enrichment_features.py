"""
Enrichment Feature Joins — VIX, FOMC, Earnings

Adds three contextual columns to any OHLCV+indicator DataFrame:

    VIX_Level         float   — raw VIX close value for that date
    VIX_Regime        str     — 'Low' (<15), 'Medium' (15–25), 'High' (>25)
    FOMC_Proximity    int     — 1 if FOMC meeting within ±FOMC_PROXIMITY_DAYS, else 0
    Earnings_Proximity int    — 1 if earnings release for that ticker within
                                ±EARNINGS_PROXIMITY_DAYS, else 0

All joins are left-joins keyed on the DataFrame's DatetimeIndex so that rows
with no match keep NaN / 0 rather than being dropped.

Usage:
    from modules.features.enrichment_features import EnrichmentFeatures

    enricher = EnrichmentFeatures(cache_dir='cache')
    df_enriched = enricher.enrich(df, ticker='AAPL')
"""

import os
import pandas as pd
import numpy as np
from typing import Optional

from modules.config.capstone_v4_params import (
    VIX_LOW_THRESHOLD,
    VIX_HIGH_THRESHOLD,
    FOMC_PROXIMITY_DAYS,
    EARNINGS_PROXIMITY_DAYS,
)


class EnrichmentFeatures:
    """
    Joins VIX, FOMC, and earnings proximity columns onto a feature DataFrame.

    The class loads the three cache CSVs once on first use and caches them in
    memory for subsequent calls, so it is efficient to create one instance and
    reuse it across all tickers in a simulation loop.
    """

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self._vix: Optional[pd.DataFrame] = None
        self._fomc: Optional[pd.Series] = None
        self._earnings: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def enrich(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Add VIX_Level, VIX_Regime, FOMC_Proximity, and Earnings_Proximity
        columns to df.

        Args:
            df:     DataFrame with a DatetimeIndex (trading-day index).
            ticker: Ticker symbol — used to filter the earnings calendar.

        Returns:
            A copy of df with the four new columns appended.
        """
        df = df.copy()
        df = self._join_vix(df)
        df = self._join_fomc(df)
        df = self._join_earnings(df, ticker)
        return df

    def enrich_date(self, date: pd.Timestamp, ticker: str) -> dict:
        """
        Return enrichment values for a single date and ticker as a dict.
        Useful inside simulation loops to avoid re-joining entire DataFrames.

        Returns:
            dict with keys: vix_level, vix_regime, fomc_proximity, earnings_proximity
        """
        self._load_all()
        return {
            "vix_level": self._vix_value(date),
            "vix_regime": self._vix_regime_value(date),
            "fomc_proximity": self._fomc_proximity_value(date),
            "earnings_proximity": self._earnings_proximity_value(date, ticker),
        }

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _load_all(self):
        self._load_vix()
        self._load_fomc()
        self._load_earnings()

    def _load_vix(self):
        if self._vix is not None:
            return
        path = os.path.join(self.cache_dir, "vix.csv")
        vix = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        vix.index = pd.to_datetime(vix.index).normalize()
        # The CSV has a single column; normalise its name to 'VIX'
        vix.columns = ["VIX"]
        self._vix = vix

    def _load_fomc(self):
        if self._fomc is not None:
            return
        path = os.path.join(self.cache_dir, "fomc_dates.csv")
        fomc = pd.read_csv(path, parse_dates=["Date"])
        self._fomc = pd.to_datetime(fomc["Date"]).dt.normalize()

    def _load_earnings(self):
        if self._earnings is not None:
            return
        path = os.path.join(self.cache_dir, "earnings_dates.csv")
        earnings = pd.read_csv(path, parse_dates=["Date"])
        earnings["Date"] = pd.to_datetime(earnings["Date"]).dt.normalize()
        self._earnings = earnings

    # ------------------------------------------------------------------
    # DataFrame-level join helpers
    # ------------------------------------------------------------------

    def _join_vix(self, df: pd.DataFrame) -> pd.DataFrame:
        self._load_vix()
        idx = df.index.normalize()
        vix_aligned = self._vix.reindex(idx)
        df["VIX_Level"] = vix_aligned["VIX"].values

        df["VIX_Regime"] = pd.cut(
            df["VIX_Level"],
            bins=[-np.inf, VIX_LOW_THRESHOLD, VIX_HIGH_THRESHOLD, np.inf],
            labels=["Low", "Medium", "High"],
            right=False,
        ).astype(str)
        # pd.cut returns 'nan' string for NaN inputs; convert back to np.nan
        df.loc[df["VIX_Level"].isna(), "VIX_Regime"] = np.nan

        return df

    def _join_fomc(self, df: pd.DataFrame) -> pd.DataFrame:
        self._load_fomc()
        # Convert both sides to integer day counts to avoid dtype promotion issues
        idx_days = df.index.normalize().values.astype("datetime64[D]").astype(np.int64)
        fomc_days = self._fomc.values.astype("datetime64[D]").astype(np.int64)
        # Shape: (n_dates, n_fomc) — broadcast subtraction
        diffs = np.abs(idx_days[:, None] - fomc_days[None, :])
        df["FOMC_Proximity"] = (diffs.min(axis=1) <= FOMC_PROXIMITY_DAYS).astype(int)
        return df

    def _join_earnings(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        self._load_earnings()
        ticker_dates = self._earnings.loc[
            self._earnings["Ticker"] == ticker, "Date"
        ]
        idx_days = df.index.normalize().values.astype("datetime64[D]").astype(np.int64)
        if len(ticker_dates) == 0:
            df["Earnings_Proximity"] = 0
        else:
            earn_days = ticker_dates.values.astype("datetime64[D]").astype(np.int64)
            diffs = np.abs(idx_days[:, None] - earn_days[None, :])
            df["Earnings_Proximity"] = (diffs.min(axis=1) <= EARNINGS_PROXIMITY_DAYS).astype(int)
        return df

    # ------------------------------------------------------------------
    # Scalar lookup helpers (used by enrich_date)
    # ------------------------------------------------------------------

    def _vix_value(self, date: pd.Timestamp) -> Optional[float]:
        self._load_vix()
        date = pd.Timestamp(date).normalize()
        if date in self._vix.index:
            return float(self._vix.loc[date, "VIX"])
        # Fallback: nearest past trading day within 5 days
        candidates = self._vix.index[self._vix.index <= date]
        if len(candidates) and (date - candidates[-1]).days <= 5:
            return float(self._vix.loc[candidates[-1], "VIX"])
        return None

    def _vix_regime_value(self, date: pd.Timestamp) -> Optional[str]:
        self._load_vix()
        vix = self._vix_value(date)
        if vix is None:
            return None
        if vix < VIX_LOW_THRESHOLD:
            return "Low"
        if vix <= VIX_HIGH_THRESHOLD:
            return "Medium"
        return "High"

    def _fomc_proximity_value(self, date: pd.Timestamp) -> int:
        self._load_fomc()
        d = np.int64(np.datetime64(pd.Timestamp(date).normalize(), "D").astype(np.int64))
        fomc_days = self._fomc.values.astype("datetime64[D]").astype(np.int64)
        return int(np.any(np.abs(d - fomc_days) <= FOMC_PROXIMITY_DAYS))

    def _earnings_proximity_value(self, date: pd.Timestamp, ticker: str) -> int:
        self._load_earnings()
        ticker_dates = self._earnings.loc[
            self._earnings["Ticker"] == ticker, "Date"
        ]
        if len(ticker_dates) == 0:
            return 0
        d = np.int64(np.datetime64(pd.Timestamp(date).normalize(), "D").astype(np.int64))
        earn_days = ticker_dates.values.astype("datetime64[D]").astype(np.int64)
        return int(np.any(np.abs(d - earn_days) <= EARNINGS_PROXIMITY_DAYS))
