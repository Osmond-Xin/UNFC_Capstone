"""
Model Layer for SPY Analysis

Capstone strategies: rule-based monthly expiry signals.
"""

from .base import BaseStrategy
from .pattern_models import (
    RSIReversalStrategy,
    ConsecutiveCandleStrategy,
    MADistanceStrategy,
    VolumeMACD_ComboStrategy,
)
from .model_registry import ModelRegistry

__all__ = [
    'BaseStrategy',
    'RSIReversalStrategy',
    'ConsecutiveCandleStrategy',
    'MADistanceStrategy',
    'VolumeMACD_ComboStrategy',
    'ModelRegistry',
]
