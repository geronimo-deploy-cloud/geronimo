"""Geronimo Features Module.

The features module provides the abstractions for defining feature transformations
and pipelines. It is inspired by scikit-learn's fit/transform paradigm but
optimized for production systems where consistency between training and serving is critical.

Key components:
- FeatureSet: A logical grouping of features (e.g., user features, item features).
- Feature: A specific transformation logic (e.g., OneHotEncoding, Normalization).

This module ensures that the exact same feature engineering logic is applied during
batch training and real-time inference preventing training-serving skew.
"""

from geronimo.features.base import FeatureSet
from geronimo.features.feature import Feature

__all__ = ["FeatureSet", "Feature"]

__docformat__ = "google"
