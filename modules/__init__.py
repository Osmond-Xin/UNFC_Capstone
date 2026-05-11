"""
SPY Analysis Modules

Modular architecture for SPY expiry date pattern analysis.

Layers:
- data: Data acquisition, caching, validation
- features: Feature engineering and technical indicators
- models: Trading strategies and pattern models
- evaluation: Performance analysis and metrics
"""

from . import data
from . import features
from . import models
from . import evaluation

__all__ = ['data', 'features', 'models', 'evaluation']
