from zenml import step

from src.features.feature_engineering import FeatureEngineer, OneHotEncoding, StandardScaling


def preprocessing(X):
    df = X.drop(
        columns=[
            'transaction_id', 'timestamp', 'sender_id', 'receiver_id',
            'customer_id', 'cpf', 'pix_key', 'hour_date', 'minute_date'
        ]
    )

    features_preprocessing = {
        'onehotencoder': ['hour_category', 'age_category', 'account_type', 'gender', 'device_type'],
        'standardscaler': ['amount', 'age'],
    }

    fe_pipeline = FeatureEngineer(strategies=[
        OneHotEncoding(features=features_preprocessing['onehotencoder']),
        StandardScaling(features=features_preprocessing['standardscaler']),
    ])

    X_processed = fe_pipeline.fit_transform(df)

    return X_processed
