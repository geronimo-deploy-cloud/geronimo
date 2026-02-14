"""Feature definitions for iris-realtime.

DEVELOPMENT WORKFLOW:
1. Review training and production data sources for column consistency
2. Perform exploratory data analysis (EDA) to identify:
   - Missing values → consider imputation strategies
   - Outliers → consider clipping or winsorization  
   - Skewed distributions → consider log/power transforms
   - Categorical cardinality → consider encoding strategies
3. Define Features with appropriate transformers

Each Feature describes a column with its type, transformer, and encoder.
The FeatureSet handles fit_transform (training) and transform (inference).
"""

from geronimo.features import FeatureSet, Feature
from sklearn.preprocessing import StandardScaler


class IrisRealtimeFeatures(FeatureSet):
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
    