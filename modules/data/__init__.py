"""
Data Layer for SPY Analysis

This module provides data acquisition, caching, validation, and expiry date calculation.

Main classes:
- CacheManager: Smart cache update manager
- DataLoader: Unified data loading interface
- DataValidator: Data quality validation
- ExpiryCalculator: SPY options expiry date calculations
"""

from .cache_manager import CacheManager
from .data_loader import DataLoader
from .data_validator import DataValidator
from .expiry_calculator import ExpiryCalculator
from .news_fetcher import NewsFetcher

__all__ = [
    'CacheManager',
    'DataLoader',
    'DataValidator',
    'ExpiryCalculator',
    'NewsFetcher',
]
