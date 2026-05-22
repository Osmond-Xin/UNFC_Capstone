"""
Base Strategy Interface

This module defines the abstract base class for all trading strategies.
All strategy models must implement this interface.

Usage:
    from modules.models import BaseStrategy

    class MyStrategy(BaseStrategy):
        def predict(self, df, expiry_date):
            # Strategy logic
            return signal  # 1 (long), -1 (short), 0 (neutral)

        def get_metadata(self):
            return {'name': 'MyStrategy', 'version': '1.0'}
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies

    All strategies must inherit from this class and implement:
    - predict(df, expiry_date): Generate trading signal
    - get_metadata(): Return strategy metadata

    Signal values:
    - 1: Long (expect price to go up)
    - -1: Short (expect price to go down)
    - 0: Neutral (no clear signal or insufficient data)

    Contract:
    - Must be stateless (predict can be called multiple times)
    - Must handle missing data gracefully (return 0 if insufficient data)
    - Config should be stored in self.config dict
    - Should not raise exceptions (catch and return neutral signal)
    """

    def __init__(self, config: dict = None):
        """
        Initialize the strategy

        Args:
            config (dict, optional): Configuration parameters for the strategy
                                   Will be merged with default config
        """
        # Start with default config
        self.config = self.get_default_config()

        # Merge with user-provided config
        if config:
            self.config.update(config)

        # Set name attribute for easy access
        self.name = self.__class__.__name__

    @abstractmethod
    def predict(self, df: pd.DataFrame, expiry_date: pd.Timestamp) -> int:
        """
        Generate trading signal for a stock on expiry date

        Args:
            df (pd.DataFrame): Stock data with OHLCV and technical indicators
            expiry_date (pd.Timestamp): Expiry date to generate signal for

        Returns:
            int: Signal value
                - 1: Long (buy)
                - -1: Short (sell)
                - 0: Neutral (no signal)

        Note:
            - df must have all required features pre-calculated
            - Must handle case where expiry_date is not in df.index
            - Should not raise exceptions (return 0 on error)
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict:
        """
        Get strategy metadata for tracking and reporting

        Returns:
            dict: Strategy metadata including:
                - name (str): Strategy name
                - version (str): Strategy version
                - description (str): Brief description
                - config (dict): Current configuration
        """
        pass

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Train/calibrate the strategy (optional)

        This method is optional and can be used for ML-based strategies
        that need to learn from historical data.

        Args:
            train_data (pd.DataFrame): Historical training data

        Note:
            Most pattern-based strategies don't need training.
            Override this method only if your strategy requires it.
        """
        pass

    def get_default_config(self) -> Dict:
        """
        Get default configuration for this strategy

        Returns:
            dict: Default configuration parameters

        Note:
            Subclasses should override this to provide sensible defaults
        """
        return {}

    def get_required_features(self) -> list:
        """
        Get list of required feature columns

        Returns:
            list: List of required feature names

        Note:
            Subclasses should override this to specify their requirements
        """
        return []

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate that DataFrame has required features

        Args:
            df (pd.DataFrame): DataFrame to validate

        Returns:
            bool: True if valid, False otherwise
        """
        required_features = self.get_required_features()
        missing_features = [f for f in required_features if f not in df.columns]

        if missing_features:
            return False

        return True

    def _get_value_at_date(
        self,
        df: pd.DataFrame,
        column: str,
        date: pd.Timestamp,
        lookback: int = 0
    ) -> Optional[float]:
        """
        Helper method to get column value at a specific date

        Args:
            df (pd.DataFrame): DataFrame with DatetimeIndex
            column (str): Column name
            date (pd.Timestamp): Target date
            lookback (int): Number of days to look back from date

        Returns:
            float: Column value, or None if not available
        """
        try:
            if date not in df.index:
                # Find closest date
                closest_date = df.index[df.index <= date]
                if len(closest_date) == 0:
                    return None
                date = closest_date[-1]

            # Get index position
            idx = df.index.get_loc(date)

            # Apply lookback
            target_idx = max(0, idx - lookback)

            # Get value
            value = df.iloc[target_idx][column]

            return value if not pd.isna(value) else None

        except Exception:
            return None

    def __repr__(self):
        """String representation of the strategy"""
        metadata = self.get_metadata()
        return f"{metadata.get('name', self.__class__.__name__)}(config={self.config})"
