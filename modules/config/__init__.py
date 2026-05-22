"""
Configuration module for capstone and shared strategy parameters.
"""

from .ema_params import (
    EMA_SIGNAL_CONFIG,
    EMA_SIGNAL_THRESHOLDS,
    EMA_BUY_WEIGHTS,
    calculate_buy_confidence,
    is_kill_candle,
)
from .high_point_params import (
    HEIGHT_SCORE_WEIGHTS,
    HEIGHT_SCORE_THRESHOLDS,
    RESISTANCE_CONFIG,
    calculate_height_score,
)
from .capstone_v4_params import (
    RSI_THRESHOLD,
    MIN_CONSECUTIVE,
    HOLD_DAYS,
    COMMISSION,
    MAX_ENTRIES_PER_EXPIRY,
    MAX_CONCURRENT_POSITIONS,
    COMPOSITE_WEIGHTS,
    RANDOM_SEED,
)

__all__ = [
    'RSI_THRESHOLD',
    'MIN_CONSECUTIVE',
    'HOLD_DAYS',
    'COMMISSION',
    'MAX_ENTRIES_PER_EXPIRY',
    'MAX_CONCURRENT_POSITIONS',
    'COMPOSITE_WEIGHTS',
    'RANDOM_SEED',
    'EMA_SIGNAL_CONFIG',
    'EMA_SIGNAL_THRESHOLDS',
    'EMA_BUY_WEIGHTS',
    'calculate_buy_confidence',
    'is_kill_candle',
    'HEIGHT_SCORE_WEIGHTS',
    'HEIGHT_SCORE_THRESHOLDS',
    'RESISTANCE_CONFIG',
    'calculate_height_score',
]
