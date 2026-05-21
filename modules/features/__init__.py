"""
Feature Engineering Layer for SPY Analysis

This module provides feature calculation and engineering capabilities.

Main classes:
- BaseFeature: Abstract interface for feature calculators
- TechnicalIndicators: Calculate technical analysis indicators (monthly)
- WeeklyIndicators: Calculate indicators for weekly expiry analysis
- EMAIndicators: Calculate EMA-based indicators for signal strategy
- FeaturePipeline: Orchestrate multiple feature calculators
"""

from .base import BaseFeature
from .technical_indicators import TechnicalIndicators
from .feature_pipeline import FeaturePipeline
from .weekly_indicators import WeeklyIndicators, WeeklyIndicatorsLite
from .ema_indicators import EMAIndicators
from .weekly_resampler import WeeklyResampler
from .high_point_indicators import HighPointIndicators
from .breakout_indicators import BreakoutIndicators
from .divergence_indicators import DivergenceIndicators
from .enrichment_features import EnrichmentFeatures

__all__ = [
    'EnrichmentFeatures',
    'BaseFeature',
    'TechnicalIndicators',
    'WeeklyIndicators',
    'WeeklyIndicatorsLite',
    'EMAIndicators',
    'WeeklyResampler',
    'HighPointIndicators',
    'BreakoutIndicators',
    'DivergenceIndicators',
    'FeaturePipeline'
]
