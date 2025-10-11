from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureEngineering(ABC):
    @abstractmethod
    def apply_transform(self, df):
        pass


class OneHotEncoding(FeatureEngineering):
    def __init__(self, features):
        self.features = features

    def apply_transform(self, df):
        encoder = OneHotEncoder(handle_unknown='ignore')
        df_transformed = encoder.fit_transform(df[self.features])
        return df_transformed


class StandardScaling(FeatureEngineering):
    def __init__(self, features):
        self.features = features

    def apply_transform(self, df):
        scaler = StandardScaler()
        df_transformed = scaler.fit_transform(df[self.features])
        return df_transformed


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, strategies=None):

        self.strategies = strategies

    def fit(self, X, y=None):
        for strat in self.strategies:
            if hasattr(strat, 'encoder') or hasattr(strat, 'scaler'):
                # Algumas estratégias possuem fit embutido
                strat.apply_transform(X)  # aqui fit será chamado
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for strat in self.strategies:
            X_transformed = strat.apply_transform(X_transformed)
        return X_transformed
