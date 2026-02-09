"""Test model implementations using Geronimo constructs.

Provides minimal model implementations for integration testing
that follow Geronimo's Model and FeatureSet patterns.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd

from geronimo.artifacts import ArtifactStore
from geronimo.features import Feature, FeatureSet
from geronimo.models import Model, HyperParams


class TestFeatures(FeatureSet):
    """Minimal feature set for testing.
    
    Uses the Iris dataset feature columns as a simple,
    well-known test case.
    """
    
    sepal_length = Feature(dtype="numeric")
    sepal_width = Feature(dtype="numeric")
    petal_length = Feature(dtype="numeric")
    petal_width = Feature(dtype="numeric")


class TestModel(Model):
    """Minimal model for integration testing.
    
    Uses Geronimo's Model base class with sklearn LogisticRegression.
    Follows the same pattern as IrisModel in examples/iris-realtime.
    
    This model can train, save, load, and predict using the
    Iris dataset, making it ideal for end-to-end testing.
    
    Example:
        >>> from geronimo.testing import TestModel
        >>> from geronimo.artifacts import ArtifactStore
        >>> 
        >>> model = TestModel()
        >>> metrics = model.train()
        >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
        >>> 
        >>> store = ArtifactStore(project="test", version="1.0.0")
        >>> model.save(store)
    """
    
    name = "test-model"
    version = "1.0.0"
    
    # Class labels for Iris dataset
    SPECIES = ["setosa", "versicolor", "virginica"]
    
    def __init__(self) -> None:
        super().__init__()
        self.estimator: Optional[Any] = None
        self.features: Optional[TestFeatures] = None
        self._is_fitted = False
    
    def train(self) -> dict:
        """Train on the Iris dataset.
        
        Uses sklearn's built-in Iris dataset for simplicity.
        
        Returns:
            Training metrics dict with accuracy, sample count, etc.
        """
        try:
            from sklearn.datasets import load_iris
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            raise ImportError(
                "scikit-learn is required for TestModel. "
                "Install with: pip install geronimo[testing]"
            )
        
        # Load Iris dataset
        iris = load_iris()
        df = pd.DataFrame(
            iris.data,
            columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        )
        y = iris.target
        
        # Initialize and fit features
        self.features = TestFeatures()
        X_transformed = self.features.fit_transform(df)
        
        # Train a simple LogisticRegression
        params = HyperParams(max_iter=200, random_state=42)
        self.estimator = LogisticRegression(**params.to_dict())
        self.estimator.fit(X_transformed, y)
        self._is_fitted = True
        
        # Calculate training accuracy
        train_accuracy = self.estimator.score(X_transformed, y)
        
        return {
            "accuracy": train_accuracy,
            "n_samples": len(y),
            "n_features": X_transformed.shape[1],
        }
    
    def predict(self, X: Any) -> np.ndarray:
        """Predict species for input features.
        
        Args:
            X: Feature array or DataFrame of shape (n_samples, 4)
            
        Returns:
            Predicted class labels
        """
        if not self._is_fitted:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(
                X,
                columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
            )
        else:
            df = X
        
        X_transformed = self.features.transform(df)
        return self.estimator.predict(X_transformed)
    
    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature array or DataFrame of shape (n_samples, 4)
            
        Returns:
            Probability array of shape (n_samples, 3)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(
                X,
                columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
            )
        else:
            df = X
        
        X_transformed = self.features.transform(df)
        return self.estimator.predict_proba(X_transformed)
    
    def save(self, store: ArtifactStore) -> list[str]:
        """Save trained model and features to ArtifactStore.
        
        Args:
            store: ArtifactStore instance for saving artifacts
            
        Returns:
            List of saved artifact URIs
        """
        if not self._is_fitted:
            raise RuntimeError("Model not trained. Nothing to save.")
        
        paths = []
        
        # Save the trained estimator
        path = store.save(
            "estimator",
            self.estimator,
            artifact_type="LogisticRegression",
            tags={"model": self.name, "version": self.version},
        )
        paths.append(path)
        
        # Save the fitted features
        path = store.save(
            "features",
            self.features,
            artifact_type="TestFeatures",
            tags={"model": self.name, "version": self.version},
        )
        paths.append(path)
        
        return paths
    
    def load(self, store: ArtifactStore) -> None:
        """Load trained model and features from ArtifactStore.
        
        Args:
            store: ArtifactStore instance for loading artifacts
        """
        self.estimator = store.get("estimator")
        self.features = store.get("features")
        self._is_fitted = True
    
    @property
    def is_fitted(self) -> bool:
        """Check if model is trained and ready for predictions."""
        return self._is_fitted
