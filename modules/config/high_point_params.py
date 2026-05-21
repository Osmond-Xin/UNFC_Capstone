"""
High Point Detection Parameters

This module contains configuration parameters for detecting whether
a stock is at a technical high point. The strategy combines:
1. Overbought sub-scores (RSI, BB, MA distance, MACD)
2. Resistance level detection (pivot highs, round numbers)

Combined HEIGHT_SCORE is a weighted composite (0-1) indicating
how "elevated" the stock is from a technical perspective.

Usage:
    from modules.config import HEIGHT_SCORE_WEIGHTS, HEIGHT_SCORE_THRESHOLDS
    from modules.config import RESISTANCE_CONFIG, calculate_height_score
"""

# =============================================================================
# Height Score Weights (sum = 1.0)
# =============================================================================
HEIGHT_SCORE_WEIGHTS = {
    'rsi_overbought': 0.30,         # RSI overbought contribution
    'bb_position': 0.25,            # Bollinger Band position
    'ma_distance': 0.25,            # Distance above MA(20)
    'macd_momentum': 0.20,          # MACD histogram momentum
}

# =============================================================================
# Height Score Thresholds
# =============================================================================
HEIGHT_SCORE_THRESHOLDS = {
    # RSI overbought scoring
    'rsi_ob_start': 50,             # RSI at which scoring begins
    'rsi_ob_full': 70,              # RSI at which score reaches 1.0

    # BB position scoring
    'bb_ob_start': 0.5,             # BB_Position at which scoring begins

    # MA distance scoring (percentage above MA20)
    'ma_dist_full': 5.0,            # % above MA20 for full score

    # MACD histogram normalization
    'macd_hist_lookback': 20,       # Lookback period for normalizing MACD_Hist
}

# =============================================================================
# Resistance / Support Configuration
# =============================================================================
RESISTANCE_CONFIG = {
    'pivot_lookback': 20,           # Bars on each side for pivot detection
    'min_prominence_pct': 2.0,      # Min prominence as % above surrounding troughs
    'max_age_bars': 252,            # Max age of pivot in trading days (1 year)

    # Round number resistance rules
    'round_levels': [
        # (price_threshold, round_to)
        (50, 1),                    # Price < $50: round to next $1
        (200, 5),                   # $50 <= Price < $200: round to next $5
        (float('inf'), 10),         # Price >= $200: round to next $10
    ],
}


def calculate_height_score(indicators: dict) -> float:
    """
    Calculate height score (0-1) from overbought sub-scores.

    Args:
        indicators: Dictionary of sub-score values:
            - RSI_OB_SCORE: RSI overbought score (0-1)
            - BB_OB_SCORE: BB position score (0-1)
            - MA_OB_SCORE: MA distance score (0-1)
            - MACD_OB_SCORE: MACD histogram score (0-1)

    Returns:
        float: Weighted composite height score between 0 and 1
    """
    score = 0.0

    rsi_ob = indicators.get('RSI_OB_SCORE', 0.0)
    score += HEIGHT_SCORE_WEIGHTS['rsi_overbought'] * max(0.0, min(1.0, rsi_ob))

    bb_ob = indicators.get('BB_OB_SCORE', 0.0)
    score += HEIGHT_SCORE_WEIGHTS['bb_position'] * max(0.0, min(1.0, bb_ob))

    ma_ob = indicators.get('MA_OB_SCORE', 0.0)
    score += HEIGHT_SCORE_WEIGHTS['ma_distance'] * max(0.0, min(1.0, ma_ob))

    macd_ob = indicators.get('MACD_OB_SCORE', 0.0)
    score += HEIGHT_SCORE_WEIGHTS['macd_momentum'] * max(0.0, min(1.0, macd_ob))

    return min(score, 1.0)
