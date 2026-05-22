"""
ADX Suitability Pre-Screening

Filters out stocks structurally unsuitable for trend-following short strategies.
Consumer Staples / Utilities (HSY, KO, PEP, KMB...) with ADX chronically < 20
are removed before the scanner runs.

Extracted from notebooks/ema5_slope_crossover_short.ipynb cell 13.

Usage:
    from modules.analysis.adx_prescreen import prescreen_universe

    filtered_data, meta = prescreen_universe(stock_data)
    # meta = {'total_screened': 506, 'suitable_count': 300, 'filtered_out_count': 206}
"""

import pandas as pd
from typing import Dict, Optional, Tuple

from modules.features.ema_indicators import EMAIndicators

PRESCREEN_DEFAULTS = {
    'adx_period': 14,
    'adx_threshold': 20,
    'min_trending_ratio': 0.40,   # >= 40% of days must have ADX >= threshold
    'min_atr_pct': 1.0,           # Minimum daily volatility (%)
    'max_atr_pct': 8.0,           # Maximum daily volatility (%)
    'lookback_days': 252,         # ~1 year of trading days
}


def compute_stock_suitability(df: pd.DataFrame, cfg: Optional[dict] = None) -> Optional[dict]:
    """
    Compute ADX-based suitability metrics for a single stock.

    A stock is 'suitable' when:
      1. It trends often enough (ADX >= threshold for >= min_trending_ratio of days).
      2. Its daily volatility (ATR%) falls within [min_atr_pct, max_atr_pct].

    Args:
        df:  OHLCV DataFrame with DatetimeIndex
        cfg: pre-screening config dict (defaults to PRESCREEN_DEFAULTS)

    Returns:
        dict with keys: trending_ratio, avg_adx, atr_pct, suitable
        None if insufficient data
    """
    if cfg is None:
        cfg = PRESCREEN_DEFAULTS
    period = cfg['adx_period']
    threshold = cfg['adx_threshold']
    lookback = cfg['lookback_days']

    if df is None or len(df) < max(lookback // 2, period * 3):
        return None

    recent = df.tail(lookback).copy()  # .copy() required: calculate() mutates in-place

    # Use public API to compute ADX via EMAIndicators
    ema_calc = EMAIndicators(config={'ema_period': 5, 'adx_period': period})
    recent = ema_calc.calculate(recent)
    adx_col = f'ADX_{period}'
    adx_valid = recent[adx_col].dropna()

    if len(adx_valid) < period:
        return None

    trending_ratio = float((adx_valid >= threshold).sum()) / len(adx_valid)
    avg_adx = float(adx_valid.mean())

    # ATR% = mean(ATR_14 / Close) * 100
    tr = pd.concat([
        recent['High'] - recent['Low'],
        (recent['High'] - recent['Close'].shift(1)).abs(),
        (recent['Low'] - recent['Close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_pct = float(
        (tr.rolling(period).mean() / recent['Close']).dropna().mean() * 100
    )

    suitable = (
        trending_ratio >= cfg['min_trending_ratio']
        and cfg['min_atr_pct'] <= atr_pct <= cfg['max_atr_pct']
    )

    return {
        'trending_ratio': round(trending_ratio, 3),
        'avg_adx': round(avg_adx, 1),
        'atr_pct': round(atr_pct, 2),
        'suitable': suitable,
    }


def prescreen_universe(
    stock_data: Dict[str, pd.DataFrame],
    cfg: Optional[dict] = None,
) -> Tuple[Dict[str, pd.DataFrame], dict]:
    """
    Filter the stock universe by ADX suitability.

    Args:
        stock_data: {ticker: OHLCV DataFrame}
        cfg: pre-screening config (defaults to PRESCREEN_DEFAULTS)

    Returns:
        (filtered_stock_data, meta)
        meta = {'total_screened': int, 'suitable_count': int, 'filtered_out_count': int}
    """
    filtered = {}
    total = 0

    for ticker, df in stock_data.items():
        result = compute_stock_suitability(df, cfg)
        if result is None:
            continue
        total += 1
        if result['suitable']:
            filtered[ticker] = df

    meta = {
        'total_screened': total,
        'suitable_count': len(filtered),
        'filtered_out_count': total - len(filtered),
    }
    return filtered, meta
