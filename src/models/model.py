from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class ModelSelection(ABC):
    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass


class RegressionLogisticModel(ModelSelection):
    def __init__(self, **kwargs):
        self.model = LogisticRegression(**kwargs,class_weight='balanced')

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)


class RandomForestClassifierModel(ModelSelection):

    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs,class_weight='balanced')

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)


class NewModel:
    def __init__(self, strategy: ModelSelection):
        self.strategy = strategy

    def fit(self, X_train, y_train):
        self.strategy.fit(X_train, y_train)

    def predict(self, X_test):
        return self.strategy.predict(X_test)
