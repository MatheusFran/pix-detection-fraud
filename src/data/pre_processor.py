from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class Preprocessor(ABC):
    @abstractmethod
    def pre_process(self, df):
        return df

class CategoricalPreprocessor(Preprocessor):
    def pre_process(self, df):
        categorical = df.select_dtypes(exclude=[np.number])
        return df.select_dtypes(include=[np.number])

class NumericalPreprocessor(Preprocessor):
    def pre_process(self, df):
        return df.select_dtypes(include=[np.number])


class ClassPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, strategy: Preprocessor):
        self._strategy = strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self._strategy.pre_process(X)

    # def set_strategy(self, strategy: Preprocessor):
    #     self._strategy = strategy