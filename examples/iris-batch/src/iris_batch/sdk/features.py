"""Feature definitions for iris-batch.

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


class IrisBatchFeatures(FeatureSet):
    """Feature engineering for iris-batch.
    
    Define your features here. Each Feature describes a column transformation.
    """
    sepal_length = Feature(dtype="numeric", transformer=StandardScaler())
    sepal_width = Feature(dtype="numeric", transformer=StandardScaler())
    petal_length = Feature(dtype="numeric", transformer=StandardScaler())
    petal_width = Feature(dtype="numeric", transformer=StandardScaler())
