"""
Base Feature Interface

This module defines the abstract base class that all feature calculators must implement.
This ensures a consistent interface across all feature engineering modules.

Usage:
    from modules.features import BaseFeature

    class MyFeature(BaseFeature):
        def calculate(self, df):
            # Feature calculation logic
            return df

        def get_feature_names(self):
            return ['my_feature_1', 'my_feature_2']
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import List


class BaseFeature(ABC):
    """
    Abstract base class for all feature calculators

    All feature calculators must inherit from this class and implement:
    - calculate(df): Add feature columns to input DataFrame
    - get_feature_names(): Return list of feature column names

    Contract:
    - Input DataFrame must have: Open, High, Low, Close, Volume columns
    - Must not modify original columns (only add new ones)
    - Must handle NaN gracefully (especially in rolling calculations)
    - Feature names must be unique across all calculators
    """

    def __init__(self, config: dict = None):
        """
        Initialize the feature calculator

        Args:
            config (dict, optional): Configuration parameters for feature calculation
                                   Will be merged with default config
        """
        # Start with default config
        self.config = self.get_default_config()

        # Merge with user-provided config
        if config:
            self.config.update(config)

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features and add them as new columns to the DataFrame

        Args:
            df (pd.DataFrame): Input DataFrame with OHLCV columns

        Returns:
            pd.DataFrame: DataFrame with added feature columns

        Raises:
            ValueError: If required columns are missing
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature column names this calculator adds

        Returns:
            list: List of feature column names
        """
        pass

    def get_default_config(self) -> dict:
        """
        Get default configuration for this feature calculator

        Returns:
            dict: Default configuration parameters

        Note:
            Subclasses should override this to provide sensible defaults
        """
        return {}

    def validate_input(self, df: pd.DataFrame) -> None:
        """
        Validate that input DataFrame has required columns

        Args:
            df (pd.DataFrame): DataFrame to validate

        Raises:
            ValueError: If required columns are missing
        """
        required_columns = self.get_required_columns()
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(
                f"{self.__class__.__name__} requires columns {required_columns}, "
                f"but {missing_columns} are missing"
            )

    def get_required_columns(self) -> List[str]:
        """
        Get list of required input columns

        Returns:
            list: List of required column names

        Note:
            Subclasses can override to specify custom requirements
            Default: OHLCV columns
        """
        return ['Open', 'High', 'Low', 'Close', 'Volume']

    def __repr__(self):
        """String representation of the feature calculator"""
        return f"{self.__class__.__name__}(config={self.config})"
