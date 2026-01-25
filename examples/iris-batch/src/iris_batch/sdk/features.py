"""Feature definitions for Iris classification."""

from sklearn.preprocessing import StandardScaler
from geronimo.features import Feature, FeatureSet


class IrisFeatures(FeatureSet):
    """Declarative feature definition for Iris dataset.
    
    Defines the schema and transformations for the input features.
    The FeatureSet handles fitting transformers (during training)
    and applying them (during inference) automatically.
    """
    
    # Numeric features - normalized using StandardScaler
    sepal_length = Feature(dtype="numeric", transformer=StandardScaler())
    sepal_width = Feature(dtype="numeric", transformer=StandardScaler())
    petal_length = Feature(dtype="numeric", transformer=StandardScaler())
    petal_width = Feature(dtype="numeric", transformer=StandardScaler())
