"""Model definition for Iris classification."""

from typing import Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import pandas as pd

from geronimo.models import Model, HyperParams
from geronimo.artifacts import ArtifactStore
from .features import IrisFeatures
from .data_sources import training_data


class IrisModel(Model):
    """Random Forest classifier for Iris species prediction.
    
    Uses the declarative IrisFeatures for feature transformation
    and ArtifactStore for persisting trained artifacts.
    
    Predicts one of three Iris species:
    - setosa (0)
    - versicolor (1)  
    - virginica (2)
    """

    name = "iris-realtime"
    version = "1.0.0"
    
    # Class labels
    SPECIES = ["setosa", "versicolor", "virginica"]
    
    def __init__(self):
        super().__init__()
        self.estimator: Optional[RandomForestClassifier] = None
        self.features: Optional[IrisFeatures] = None
        self._is_fitted = False

    def train(self, X=None, y=None, params: Optional[HyperParams] = None) -> dict:
        """Train the Iris classifier.
        
        Uses the declarative IrisFeatures for preprocessing.
        If X and y are not provided, loads data using the training_data DataSource.
        
        Returns:
            Training metrics dict
        """
        # Initialize fresh features for training
        self.features = IrisFeatures()
        
        # Load data from DataSource if not provided
        if X is None or y is None:
            df = training_data.load()
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
        
        Uses the fitted IrisFeatures for preprocessing.
        
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
        
        Uses the fitted IrisFeatures for preprocessing.
        
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
    
    def save(self, store: ArtifactStore) -> list[str]:
        """Save trained model and features to ArtifactStore.
        
        Saves:
        - estimator: The fitted RandomForest classifier
        - features: The fitted IrisFeatures (includes scalers)
        
        Args:
            store: ArtifactStore instance for saving artifacts
            
        Returns:
            List of saved artifact paths
        """
        if not self._is_fitted:
            raise RuntimeError("Model not trained. Nothing to save.")
        
        paths = []
        
        # Save the trained estimator
        path = store.save(
            "estimator", 
            self.estimator, 
            artifact_type="RandomForestClassifier",
            tags={"model": self.name, "version": self.version}
        )
        paths.append(path)
        
        # Save the fitted features (includes transformers/scalers)
        path = store.save(
            "features",
            self.features,
            artifact_type="IrisFeatures",
            tags={"model": self.name, "version": self.version}
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
