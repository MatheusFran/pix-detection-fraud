from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.city_normalize import CityNormalize


def preprocessing(df,list_cat,list_num):
    categorical = df[list_cat]
    numerical = df[list_num]

    categorical = ColumnTransformer([
        ('enconding',OneHotEncoder(handle_unknown='ignore'),categorical['device_type','gender','account_type']),
        ('city', CityNormalize(), categorical['city']),
    ])

    numerical = ColumnTransformer([
        ('scale', StandardScaler(), numerical),
    ])

    return categorical,numerical

