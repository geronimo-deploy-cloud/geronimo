"""Feature definitions - define your feature engineering here."""

from geronimo.features import FeatureSet, Feature
# from sklearn.preprocessing import StandardScaler, OneHotEncoder


class ProjectFeatures(FeatureSet):
    """Define your features here.
    
    Each Feature describes a column with its type, transformer, and encoder.
    
    Example:
        age = Feature(dtype='numeric', transformer=StandardScaler())
        income = Feature(dtype='numeric', transformer=StandardScaler())
        category = Feature(dtype='categorical', encoder=OneHotEncoder())
        
        # Derived feature from multiple columns
        age_income_ratio = Feature(
            dtype='numeric',
            derived_feature_fn=lambda df: df['age'] / df['income']
        )
    """
    
    # TODO: Define your features
    # feature_1 = Feature(dtype='numeric')
    # feature_2 = Feature(dtype='categorical')
    pass
