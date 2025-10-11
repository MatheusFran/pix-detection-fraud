from zenml import step

from src.features.feature_engineering import FeatureEngineer, OneHotEncoding, StandardScaling


@step
def preprocessing(dict):
    fe_pipeline = FeatureEngineer(strategies=[
        OneHotEncoding(features=dict['onehotenconder']),
        StandardScaling(features=dict['standardscaler']),
    ])

    return fe_pipeline
