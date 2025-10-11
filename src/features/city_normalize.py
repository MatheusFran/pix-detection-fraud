from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer


class CityNormalize(BaseEstimator, TransformerMixin):
    def __init__(self, city):
        self.city = city

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        X['city_embed'] = X['city'].apply(lambda x: model.encode(x))
        return X[self.city]

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
