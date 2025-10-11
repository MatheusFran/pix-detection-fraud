from abc import ABC, abstractmethod
import pandas as pd


class NewFeatures(ABC):
    @abstractmethod
    def apply(self):
        return self


class DateFeature(NewFeatures):
    def apply(self, X):
        date = pd.to_datetime(X['timestamp'])
        date.day = date.day
        date.month = date.month
        date.year = date.year

        X['day_date'] = date.day
        X['month_date'] = date.month
        X['year_date'] = date.year

        return X


class TimeFeature(NewFeatures):
    def apply(self, X):
        time = pd.to_datetime(X['timestamp'])

        time.hour = time.hour
        time.minute = time.minute

        X['hour_date'] = time.hour
        X['minute_date'] = time.minute

        return X


class AgeCategoryFeature(NewFeatures):
    def apply(self, X):
        age = pd.to_numeric(X['age'])

        if age <= 30:
            X['age_category'] = 'jovem'
        elif 30 < age < 50:
            X['age_category'] = 'adulto'
        else:
            X['age_category'] = 'idoso'

        return X


class HourCategoryFeature(NewFeatures):
    def apply(self, X):
        time = pd.to_numeric(X['timestamp'])
        hour = time.hour

        if 5 < hour < 12:
            X['hour_category'] = 'manha'
        elif 12 < hour < 18:
            X['hour_category'] = 'tarde'
        elif 18 < hour < 22:
            X['hour_category'] = 'noite'
        elif 22 < hour < 4:
            X['hour_category'] = 'madrugada'

        return X


class NewFeature:
    @staticmethod
    def create_feature(type):
        if type == 'data':
            return DateFeature()
        elif type == 'time':
            return TimeFeature()
        elif type == 'age':
            return AgeCategoryFeature()
        elif type == 'hour':
            return HourCategoryFeature()
        else:
            return None
