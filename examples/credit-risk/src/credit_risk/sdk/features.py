"""Feature definitions for credit risk model."""

from geronimo.features import FeatureSet, Feature
# from sklearn.preprocessing import StandardScaler, OneHotEncoder


class ProjectFeatures(FeatureSet):
    """Credit risk features.
    
    Define features for credit risk prediction including:
    - Numeric: income, debt_ratio, credit_score
    - Categorical: employment_type, loan_purpose
    """
    
    # Numeric features (would use StandardScaler in production)
    # income = Feature(dtype='numeric', transformer=StandardScaler())
    # debt_ratio = Feature(dtype='numeric', transformer=StandardScaler())
    # credit_score = Feature(dtype='numeric', transformer=StandardScaler())
    # loan_amount = Feature(dtype='numeric', transformer=StandardScaler())
    
    # Categorical features (would use OneHotEncoder in production)
    # employment_type = Feature(dtype='categorical', encoder=OneHotEncoder())
    # loan_purpose = Feature(dtype='categorical', encoder=OneHotEncoder())
    
    pass
