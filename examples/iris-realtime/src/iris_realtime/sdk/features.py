"""Feature definitions for Iris classification."""

from geronimo.features import FeatureSet, Feature
from sklearn.preprocessing import StandardScaler


class IrisFeatures(FeatureSet):
    """Feature set for Iris flower classification.
    
    All 4 measurements are numeric and standardized for optimal
    classifier performance.
    """
    
    sepal_length = Feature(
        dtype="numeric",
        transformer=StandardScaler(),
        description="Sepal length in cm"
    )
    sepal_width = Feature(
        dtype="numeric", 
        transformer=StandardScaler(),
        description="Sepal width in cm"
    )
    petal_length = Feature(
        dtype="numeric",
        transformer=StandardScaler(),
        description="Petal length in cm"
    )
    petal_width = Feature(
        dtype="numeric",
        transformer=StandardScaler(),
        description="Petal width in cm"
    )
    
    @property
    def feature_names(self) -> list[str]:
        """Return ordered list of feature column names."""
        return ["sepal_length", "sepal_width", "petal_length", "petal_width"]
