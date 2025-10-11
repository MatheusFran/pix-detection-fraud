from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, smoothing=1.0):
        self.columns = columns
        self.smoothing = smoothing
        self.target_means_ = {}
        self.global_mean_ = None

    def fit(self, X, y):
        self.global_mean_ = y.mean()
        X = X.copy()
        y = y.copy()

        for col in self.columns:
            stats = pd.DataFrame({'count': X.groupby(col).size(),
                                  'mean': X.groupby(col).apply(lambda x: y.loc[x.index].mean())})
            stats['encoding'] = (stats['mean'] * stats['count'] + self.global_mean_ * self.smoothing) / (
                        stats['count'] + self.smoothing)
            self.target_means_[col] = stats['encoding'].to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].map(self.target_means_[col]).fillna(self.global_mean_)
        return X
