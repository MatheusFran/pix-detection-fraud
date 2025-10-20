from abc import ABC, abstractmethod
import pandas as pd


class FeatureStrategy(ABC):
    @abstractmethod
    def apply(self, X):
        pass


class DateFeatureStrategy(FeatureStrategy):
    def apply(self, X):
        date = pd.to_datetime(X['timestamp'])
        X['day_date'] = date.dt.day
        X['month_date'] = date.dt.month
        X['year_date'] = date.dt.year
        return X


class TimeFeatureStrategy(FeatureStrategy):
    def apply(self, X):
        time = pd.to_datetime(X['timestamp'])
        X['hour_date'] = time.dt.hour
        X['minute_date'] = time.dt.minute
        return X


class AgeCategoryFeatureStrategy(FeatureStrategy):
    def apply(self, X):
        age = pd.to_numeric(X['age'])
        X['age_category'] = pd.cut(
            age, bins=[0, 30, 50, 200], labels=['jovem', 'adulto', 'idoso']
        )
        return X


class HourCategoryFeatureStrategy(FeatureStrategy):
    def apply(self, X):
        hour = pd.to_datetime(X['timestamp']).dt.hour
        X['hour_category'] = pd.cut(hour, bins=[-1, 4, 11, 17, 21, 23],
                                    labels=['madrugada', 'manha', 'tarde', 'noite', 'madrugada'],
                                    include_lowest=True,
                                    ordered=False)

        return X


class NewFeature:
    def __init__(self, strategy: FeatureStrategy):
        self.strategy = strategy

    def apply(self, X):
        return self.strategy.apply(X)
