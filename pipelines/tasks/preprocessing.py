from prefect import task

from src.features.build_pipe_fe import build_pipe_fe
from src.features.new_features import TimeFeatureStrategy, AgeCategoryFeatureStrategy, HourCategoryFeatureStrategy, \
    NewFeature


@task
def preprocessing(X, y, cat_cols, num_cols, columns_drop, features):
    pipe_fe = build_pipe_fe(
        num_cols=num_cols,
        cat_cols=cat_cols,
        features=features,
        columns_drop=columns_drop)

    X_resampled, y_resampled = pipe_fe.fit_resample(X, y)

    return X_resampled, y_resampled
