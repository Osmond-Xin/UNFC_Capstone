"""
Configuration module for analysis parameters
"""

from .weekly_params import (
    WEEKLY_DEFAULT_CONFIG,
    WEEKLY_SIGNAL_THRESHOLDS,
    WEEKLY_SIGNAL_WEIGHTS,
)
from .ema_params import (
    EMA_SIGNAL_CONFIG,
    EMA_SIGNAL_THRESHOLDS,
    EMA_BUY_WEIGHTS,
    calculate_buy_confidence,
    is_kill_candle,
)
from .mtf_params import (
    MTF_WEEKLY_THRESHOLDS,
    MTF_WEEKLY_WEIGHTS,
    MTF_DAILY_THRESHOLDS,
    MTF_DAILY_WEIGHTS,
    MTF_COMBINED_WEIGHTS,
    calculate_weekly_oversold_score,
    calculate_daily_reversal_score,
    calculate_combined_score,
)
from .high_point_params import (
    HEIGHT_SCORE_WEIGHTS,
    HEIGHT_SCORE_THRESHOLDS,
    RESISTANCE_CONFIG,
    calculate_height_score,
)
from .breakout_params import (
    BREAKOUT_WEIGHTS,
    BREAKOUT_THRESHOLDS,
    calculate_breakout_strength,
)
from .divergence_params import (
    DIVERGENCE_CONFIG,
    DIVERGENCE_WEIGHTS,
)
from .meta_params import (
    META_SCANNER_WEIGHTS,
    META_AGREEMENT_BONUS,
    META_HIGHPOINT_PENALTY,
    META_BACKTEST_CONFIG,
    calculate_meta_score,
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
from .active_list_params import (
    ACTIVE_LIST_PATTERNS,
    LOOKBACK_YEARS as ACTIVE_LOOKBACK_YEARS,
    TOP_N as ACTIVE_TOP_N,
    BOTTOM_N as ACTIVE_BOTTOM_N,
    EMA_PERIOD as ACTIVE_EMA_PERIOD,
    EMA_SLOPE_LOOKBACK as ACTIVE_EMA_SLOPE_LOOKBACK,
    RETURN_TYPE as ACTIVE_RETURN_TYPE,
    MIN_APPEARANCES as ACTIVE_MIN_APPEARANCES,
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
    'WEEKLY_DEFAULT_CONFIG',
    'WEEKLY_SIGNAL_THRESHOLDS',
    'WEEKLY_SIGNAL_WEIGHTS',
    'EMA_SIGNAL_CONFIG',
    'EMA_SIGNAL_THRESHOLDS',
    'EMA_BUY_WEIGHTS',
    'calculate_buy_confidence',
    'is_kill_candle',
    'MTF_WEEKLY_THRESHOLDS',
    'MTF_WEEKLY_WEIGHTS',
    'MTF_DAILY_THRESHOLDS',
    'MTF_DAILY_WEIGHTS',
    'MTF_COMBINED_WEIGHTS',
    'calculate_weekly_oversold_score',
    'calculate_daily_reversal_score',
    'calculate_combined_score',
    'HEIGHT_SCORE_WEIGHTS',
    'HEIGHT_SCORE_THRESHOLDS',
    'RESISTANCE_CONFIG',
    'calculate_height_score',
    'BREAKOUT_WEIGHTS',
    'BREAKOUT_THRESHOLDS',
    'calculate_breakout_strength',
    'DIVERGENCE_CONFIG',
    'DIVERGENCE_WEIGHTS',
    'META_SCANNER_WEIGHTS',
    'META_AGREEMENT_BONUS',
    'META_HIGHPOINT_PENALTY',
    'META_BACKTEST_CONFIG',
    'calculate_meta_score',
    'ACTIVE_LIST_PATTERNS',
    'ACTIVE_LOOKBACK_YEARS',
    'ACTIVE_TOP_N',
    'ACTIVE_BOTTOM_N',
    'ACTIVE_EMA_PERIOD',
    'ACTIVE_EMA_SLOPE_LOOKBACK',
    'ACTIVE_RETURN_TYPE',
    'ACTIVE_MIN_APPEARANCES',
]
