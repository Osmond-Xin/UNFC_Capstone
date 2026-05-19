"""
Analysis Layer for SPY Expiry Analysis

This module provides filtering and analysis utilities.

Main classes:
- VolatilityFilter: Filter stocks based on historical volatility on monthly expiry dates
- WeeklyVolatilityFilter: Filter stocks based on historical volatility on weekly expiry dates
- AIAnalyzer: AI-powered stock analysis via Claude CLI or Gemini CLI

Functions:
- compute_stock_suitability: ADX-based suitability check for a single stock
- prescreen_universe: Filter stock universe by ADX suitability
"""

from .volatility_filter import VolatilityFilter
from .weekly_volatility_filter import WeeklyVolatilityFilter
from .ai_analyzer import AIAnalyzer
from .adx_prescreen import compute_stock_suitability, prescreen_universe, PRESCREEN_DEFAULTS

__all__ = [
    'VolatilityFilter',
    'WeeklyVolatilityFilter',
    'AIAnalyzer',
    'compute_stock_suitability',
    'prescreen_universe',
    'PRESCREEN_DEFAULTS',
]
