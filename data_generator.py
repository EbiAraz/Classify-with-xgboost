"""
data_generator.py - Synthetic dataset generator for classification tasks.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class DataGenerator:
    """Generates synthetic classification datasets for training and evaluation."""

    def __init__(
        self,
        n_samples: int = 1000,
        n_features: int = 20,
        n_informative: int = 10,
        n_classes: int = 2,
        random_state: int = 42,
    ):
        """
        Initialize the DataGenerator.

        Args:
            n_samples: Total number of samples to generate.
            n_features: Total number of features.
            n_informative: Number of informative features.
            n_classes: Number of target classes (2 for binary, >2 for multi-class).
            random_state: Random seed for reproducibility.
        """
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2.")
        if n_informative > n_features:
            raise ValueError("n_informative cannot exceed n_features.")

        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative = n_informative
        self.n_classes = n_classes
        self.random_state = random_state

    def generate(self, test_size: float = 0.2):
        """
        Generate a synthetic dataset and split it into train/test sets.

        Args:
            test_size: Fraction of samples reserved for the test set.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=self.n_informative,
            n_redundant=max(0, self.n_features - self.n_informative - 2),
            n_classes=self.n_classes,
            random_state=self.random_state,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        print(
            f"Dataset generated: {self.n_samples} samples, "
            f"{self.n_features} features, {self.n_classes} classes."
        )
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        return X_train, X_test, y_train, y_test
