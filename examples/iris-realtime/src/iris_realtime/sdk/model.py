"""Model definition for Iris classification."""

from typing import Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path
import pandas as pd

from geronimo.models import Model, HyperParams
from .features import IrisFeatures
from .data_sources import load_iris_data


class IrisModel(Model):
    """Random Forest classifier for Iris species prediction.
    
    Uses the declarative IrisFeatures for feature transformation.
    
    Predicts one of three Iris species:
    - setosa (0)
    - versicolor (1)  
    - virginica (2)
    """

    name = "iris-realtime"
    version = "1.0.0"
    features = IrisFeatures()
    
    # Class labels
    SPECIES = ["setosa", "versicolor", "virginica"]
    
    def __init__(self):
        super().__init__()
        self.estimator: Optional[RandomForestClassifier] = None
        self._is_fitted = False

    def train(self, X=None, y=None, params: Optional[HyperParams] = None) -> dict:
        """Train the Iris classifier.
        
        Uses the declarative IrisFeatures for preprocessing.
        If X and y are not provided, loads from sklearn's iris dataset.
        
        Returns:
            Training metrics dict
        """
        # Load data if not provided
        if X is None or y is None:
            df = load_iris_data()
            y = df["species"].values
            
            # Use the declarative features for transformation
            X_transformed = self.features.fit_transform(df)
        else:
            # X is already a DataFrame - transform it
            if isinstance(X, pd.DataFrame):
                X_transformed = self.features.fit_transform(X)
            else:
                # X is numpy array - wrap in DataFrame first
                df = pd.DataFrame(X, columns=self.features.feature_names)
                X_transformed = self.features.fit_transform(df)
        
        # Default hyperparameters
        if params is None:
            params = HyperParams(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        
        # Train model on transformed features
        self.estimator = RandomForestClassifier(**params.to_dict())
        self.estimator.fit(X_transformed, y)
        self._is_fitted = True
        
        # Calculate training accuracy
        train_accuracy = self.estimator.score(X_transformed, y)
        
        return {
            "accuracy": train_accuracy,
            "n_samples": len(y),
            "n_features": X_transformed.shape[1],
        }

    def predict(self, X) -> np.ndarray:
        """Predict species for input features.
        
        Uses the declarative IrisFeatures for preprocessing.
        
        Args:
            X: Feature array or DataFrame of shape (n_samples, 4)
            
        Returns:
            Predicted class labels
        """
        if not self._is_fitted:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        
        # Transform using declarative features
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=self.features.feature_names)
        else:
            df = X
        
        X_transformed = self.features.transform(df)
        return self.estimator.predict(X_transformed)
    
    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities.
        
        Uses the declarative IrisFeatures for preprocessing.
        
        Args:
            X: Feature array or DataFrame of shape (n_samples, 4)
            
        Returns:
            Probability array of shape (n_samples, 3)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        
        # Transform using declarative features
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=self.features.feature_names)
        else:
            df = X
        
        X_transformed = self.features.transform(df)
        return self.estimator.predict_proba(X_transformed)
    
    def save(self, path: str = "models") -> str:
        """Save trained model and features to disk.
        
        Args:
            path: Directory to save model artifacts
            
        Returns:
            Path to saved model
        """
        if not self._is_fitted:
            raise RuntimeError("Model not trained. Nothing to save.")
        
        model_dir = Path(path)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / f"{self.name}_v{self.version}.joblib"
        
        # Save both the estimator AND the fitted features
        joblib.dump({
            "estimator": self.estimator,
            "features": self.features,  # Includes fitted transformers
            "version": self.version,
        }, model_path)
        
        return str(model_path)
    
    def load(self, path: str = "models") -> None:
        """Load trained model and features from disk.
        
        Args:
            path: Directory containing model artifacts
        """
        model_dir = Path(path)
        model_path = model_dir / f"{self.name}_v{self.version}.joblib"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        data = joblib.load(model_path)
        self.estimator = data["estimator"]
        self.features = data["features"]  # Load fitted features
        self._is_fitted = True
    
    @property
    def is_fitted(self) -> bool:
        """Check if model is trained and ready for predictions."""
        return self._is_fitted
