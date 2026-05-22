"""
Model Registry

Central registry for capstone trading strategies.
"""

from typing import Dict, List, Type

from .base import BaseStrategy
from .pattern_models import (
    RSIReversalStrategy,
    ConsecutiveCandleStrategy,
    MADistanceStrategy,
    VolumeMACD_ComboStrategy,
)


class ModelRegistry:
    """Factory for capstone strategy instances."""

    _registry: Dict[str, Type[BaseStrategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_class: Type[BaseStrategy]) -> None:
        if not issubclass(strategy_class, BaseStrategy):
            raise ValueError(
                f"Strategy class must inherit from BaseStrategy, "
                f"got {strategy_class} instead"
            )
        if name in cls._registry:
            raise ValueError(f"Strategy '{name}' is already registered")
        cls._registry[name] = strategy_class

    @classmethod
    def create(cls, name: str, config: dict = None) -> BaseStrategy:
        if name not in cls._registry:
            raise ValueError(
                f"Strategy '{name}' not found. "
                f"Available strategies: {list(cls._registry.keys())}"
            )
        return cls._registry[name](config=config)

    @classmethod
    def list_models(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def get_model_info(cls, name: str) -> Dict:
        if name not in cls._registry:
            raise ValueError(f"Strategy '{name}' not found")
        strategy_class = cls._registry[name]
        temp_instance = strategy_class()
        metadata = temp_instance.get_metadata()
        return {
            'name': name,
            'class': strategy_class.__name__,
            'metadata': metadata,
            'default_config': temp_instance.config,
        }

    @classmethod
    def _register_builtin_models(cls) -> None:
        cls.register('rsi_reversal', RSIReversalStrategy)
        cls.register('consecutive_candle', ConsecutiveCandleStrategy)
        cls.register('ma_distance', MADistanceStrategy)
        cls.register('volume_macd_combo', VolumeMACD_ComboStrategy)

    @classmethod
    def list_monthly_models(cls) -> List[str]:
        return list(cls._registry.keys())


ModelRegistry._register_builtin_models()
