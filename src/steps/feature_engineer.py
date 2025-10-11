from zenml import step

from src.features.new_features import NewFeature, AgeCategoryFeatureStrategy, HourCategoryFeatureStrategy, \
    TimeFeatureStrategy


@step
def builder_features(df):
    features = [
        TimeFeatureStrategy(),
        AgeCategoryFeatureStrategy(),
        HourCategoryFeatureStrategy()
    ]

    df_feature = map(lambda f: NewFeature(f).apply((df)), features)

    return df_feature
