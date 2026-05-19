"""
Feature Pipeline

This module orchestrates multiple feature calculators in sequence.
It provides a clean interface to apply all feature engineering steps at once.

Usage:
    from modules.features import FeaturePipeline, TechnicalIndicators

    pipeline = FeaturePipeline([
        TechnicalIndicators(config={'rsi_period': 14}),
        # Add more feature calculators here
    ])

    df_with_features = pipeline.transform(df)
"""

import pandas as pd
from typing import List
from .base import BaseFeature


class FeaturePipeline:
    """
    Pipeline for orchestrating multiple feature calculators

    This class chains multiple BaseFeature instances together,
    applying them sequentially to the input DataFrame.

    Features:
    - Sequential application of feature calculators
    - Automatic feature name collection
    - Error handling for individual calculators
    - Easy to extend with new features
    """

    def __init__(self, calculators: List[BaseFeature]):
        """
        Initialize the feature pipeline

        Args:
            calculators (list): List of BaseFeature instances to apply
        """
        self.calculators = calculators

        # Validate that all are BaseFeature instances
        for calc in calculators:
            if not isinstance(calc, BaseFeature):
                raise TypeError(
                    f"All calculators must be BaseFeature instances. "
                    f"Got {type(calc)} instead."
                )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature calculators sequentially

        Args:
            df (pd.DataFrame): Input DataFrame with OHLCV columns

        Returns:
            pd.DataFrame: DataFrame with all features added

        Note:
            If a calculator fails, it logs a warning but continues with others
        """
        # Make a copy to avoid modifying original
        df = df.copy()

        # Apply each calculator
        for calculator in self.calculators:
            try:
                df = calculator.calculate(df)
            except Exception as e:
                # Log error but continue with other calculators
                print(f"⚠ Warning: {calculator.__class__.__name__} failed: {e}")
                continue

        return df

    def get_all_feature_names(self) -> List[str]:
        """
        Get combined list of all feature names from all calculators

        Returns:
            list: List of all feature column names
        """
        all_features = []

        for calculator in self.calculators:
            features = calculator.get_feature_names()
            all_features.extend(features)

        return all_features

    def add_calculator(self, calculator: BaseFeature) -> None:
        """
        Add a new calculator to the pipeline

        Args:
            calculator (BaseFeature): Feature calculator to add

        Raises:
            TypeError: If calculator is not a BaseFeature instance
        """
        if not isinstance(calculator, BaseFeature):
            raise TypeError(
                f"Calculator must be a BaseFeature instance. "
                f"Got {type(calculator)} instead."
            )

        self.calculators.append(calculator)

    def remove_calculator(self, calculator_class: type) -> bool:
        """
        Remove a calculator by class type

        Args:
            calculator_class (type): Class type to remove (e.g., TechnicalIndicators)

        Returns:
            bool: True if removed, False if not found
        """
        initial_length = len(self.calculators)

        self.calculators = [
            calc for calc in self.calculators
            if not isinstance(calc, calculator_class)
        ]

        return len(self.calculators) < initial_length

    def get_calculator(self, calculator_class: type) -> BaseFeature:
        """
        Get a calculator instance by class type

        Args:
            calculator_class (type): Class type to find

        Returns:
            BaseFeature: Calculator instance, or None if not found
        """
        for calc in self.calculators:
            if isinstance(calc, calculator_class):
                return calc

        return None

    def __len__(self):
        """Return number of calculators in the pipeline"""
        return len(self.calculators)

    def __repr__(self):
        """String representation of the pipeline"""
        calc_names = [calc.__class__.__name__ for calc in self.calculators]
        return f"FeaturePipeline(calculators={calc_names})"
