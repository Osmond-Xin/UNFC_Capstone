"""
EMA Signal Strategy Parameters

This module contains configuration parameters for the EMA signal
trading strategy. The strategy identifies buy signals when price
bounces off an upward-trending EMA, and sell signals when a
"kill candle" appears.

Usage:
    from modules.config import EMA_SIGNAL_CONFIG, EMA_SIGNAL_THRESHOLDS
"""

# Default configuration for EMA signal strategy
EMA_SIGNAL_CONFIG = {
    # EMA calculation parameters
    'ema_period': 20,               # EMA period (20-day)
    'ema_slope_lookback': 5,        # Lookback for slope calculation

    # Buy signal parameters
    'buy_ema_proximity': 0.02,      # Max distance from EMA (2%)
    'min_ema_slope': 0.0,           # Minimum EMA slope (0 = flat or rising)

    # Sell signal (Kill candle) parameters
    'kill_candle_threshold': 0.03,  # Min drop for kill candle (3%)
    'kill_body_multiplier': 2.0,    # Kill candle must be 2x avg body

    # Position management
    'stop_loss_pct': 0.05,          # Stop loss at 5% below entry
    'take_profit_pct': 0.10,        # Take profit at 10% above entry
}

# Signal thresholds for pattern matching
EMA_SIGNAL_THRESHOLDS = {
    # EMA proximity thresholds (percentage)
    'ema_proximity_tight': 0.01,    # Within 1% of EMA
    'ema_proximity_normal': 0.02,   # Within 2% of EMA
    'ema_proximity_loose': 0.03,    # Within 3% of EMA

    # EMA slope thresholds (percentage per lookback period)
    'slope_strong_up': 1.0,         # Strong uptrend: >1% rise
    'slope_moderate_up': 0.5,       # Moderate uptrend: >0.5% rise
    'slope_flat': 0.0,              # Flat or uptrend
    'slope_down': -0.5,             # Downtrend: <-0.5%

    # Kill candle thresholds
    'kill_drop_severe': 0.05,       # Severe drop: 5%+
    'kill_drop_moderate': 0.03,     # Moderate drop: 3%+
    'kill_body_large': 3.0,         # Very large: 3x avg body
    'kill_body_moderate': 2.0,      # Large: 2x avg body
}

# Confidence weights for buy signal scoring
EMA_BUY_WEIGHTS = {
    'ema_uptrend': 0.30,            # EMA is in uptrend
    'price_near_ema': 0.25,         # Price close to EMA
    'candle_touches_ema': 0.20,     # Candle body touches/crosses EMA
    'close_above_ema': 0.15,        # Close is above EMA
    'volume_confirmation': 0.10,    # Volume is above average
}


def calculate_buy_confidence(indicators: dict) -> float:
    """
    Calculate confidence score for buy signal (0-1)

    Args:
        indicators: Dictionary of indicator values containing:
            - ema_slope: EMA slope percentage
            - ema_dist_pct: Distance from price to EMA (%)
            - ema_cross: EMA cross signal (-1, 0, 1)
            - close: Current close price
            - ema: Current EMA value
            - volume_ratio: Current volume / avg volume

    Returns:
        float: Confidence score between 0 and 1
    """
    score = 0.0

    # EMA uptrend (slope > 0)
    ema_slope = indicators.get('ema_slope', -1)
    if ema_slope > EMA_SIGNAL_THRESHOLDS['slope_strong_up']:
        score += EMA_BUY_WEIGHTS['ema_uptrend']
    elif ema_slope > EMA_SIGNAL_THRESHOLDS['slope_moderate_up']:
        score += EMA_BUY_WEIGHTS['ema_uptrend'] * 0.7
    elif ema_slope > EMA_SIGNAL_THRESHOLDS['slope_flat']:
        score += EMA_BUY_WEIGHTS['ema_uptrend'] * 0.4

    # Price near EMA
    ema_dist = abs(indicators.get('ema_dist_pct', 10))
    if ema_dist <= EMA_SIGNAL_THRESHOLDS['ema_proximity_tight'] * 100:
        score += EMA_BUY_WEIGHTS['price_near_ema']
    elif ema_dist <= EMA_SIGNAL_THRESHOLDS['ema_proximity_normal'] * 100:
        score += EMA_BUY_WEIGHTS['price_near_ema'] * 0.7
    elif ema_dist <= EMA_SIGNAL_THRESHOLDS['ema_proximity_loose'] * 100:
        score += EMA_BUY_WEIGHTS['price_near_ema'] * 0.4

    # Candle touches/crosses EMA
    ema_cross = indicators.get('ema_cross', 0)
    if ema_cross == 1:  # Bullish touch
        score += EMA_BUY_WEIGHTS['candle_touches_ema']
    elif ema_cross == -1:  # Bearish touch (partial credit)
        score += EMA_BUY_WEIGHTS['candle_touches_ema'] * 0.3

    # Close above EMA
    close = indicators.get('close', 0)
    ema = indicators.get('ema', close + 1)
    if close > ema:
        score += EMA_BUY_WEIGHTS['close_above_ema']

    # Volume confirmation
    volume_ratio = indicators.get('volume_ratio', 0)
    if volume_ratio > 1.5:
        score += EMA_BUY_WEIGHTS['volume_confirmation']
    elif volume_ratio > 1.0:
        score += EMA_BUY_WEIGHTS['volume_confirmation'] * 0.5

    return min(score, 1.0)


def is_kill_candle(
    candle_body_pct: float,
    avg_body: float,
    kill_threshold: float = None,
    body_multiplier: float = None
) -> bool:
    """
    Check if a candle qualifies as a "kill candle"

    A kill candle is a large bearish candle that signals potential
    trend reversal or momentum exhaustion.

    Criteria:
    1. Candle is bearish (body_pct < 0)
    2. Drop exceeds kill threshold (default 3%)
    3. Body size is larger than multiplier * avg body

    Args:
        candle_body_pct: Candle body percentage (negative for bearish)
        avg_body: Average absolute body size
        kill_threshold: Minimum drop threshold (default from config)
        body_multiplier: Body size multiplier (default from config)

    Returns:
        bool: True if candle is a kill candle
    """
    if kill_threshold is None:
        kill_threshold = EMA_SIGNAL_CONFIG['kill_candle_threshold']
    if body_multiplier is None:
        body_multiplier = EMA_SIGNAL_CONFIG['kill_body_multiplier']

    # Must be bearish
    if candle_body_pct >= 0:
        return False

    # Check drop threshold
    drop = abs(candle_body_pct) / 100  # Convert to decimal
    if drop < kill_threshold:
        return False

    # Check body size relative to average
    if avg_body > 0:
        body_ratio = abs(candle_body_pct) / avg_body
        if body_ratio < body_multiplier:
            return False

    return True
