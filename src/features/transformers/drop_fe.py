from sklearn.base import BaseEstimator, TransformerMixin
from src.features.new_features import NewFeature


class DropColumnsFe(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_drop = X.drop(columns=self.cols)
        return X_drop
