from sklearn.base import BaseEstimator, TransformerMixin
from src.features.new_features import NewFeature


class CreateNewfeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = X.copy()
        for f in self.features:
            X_new = NewFeature(f).apply(X_new)
        return X_new
