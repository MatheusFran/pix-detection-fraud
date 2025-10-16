from prefect import flow
import mlflow

from pipelines.tasks.evaluate import evaluator
from pipelines.tasks.load_data import load_data
from pipelines.tasks.preprocessing import preprocessing
from pipelines.tasks.select_model import select_model
from pipelines.tasks.split_data import split_data
from src.features.new_features import TimeFeatureStrategy, AgeCategoryFeatureStrategy, HourCategoryFeatureStrategy


@flow
def train_pipeline(cat_cols, num_cols, columns_drop):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("pix-detection-fraud-training")

    features = [
        TimeFeatureStrategy(),
        AgeCategoryFeatureStrategy(),
        HourCategoryFeatureStrategy()
    ]

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df, 'fraud')
    X_train_fe, y_train_fe = preprocessing(
        X=X_train,
        y=y_train,
        cat_cols=cat_cols,
        num_cols=num_cols,
        columns_drop=columns_drop,
        features=features
    )

    result = select_model(X_train_fe, y_train_fe)
    metrics = evaluator(result, X_test, y_test)

    return metrics


if __name__ == '__main__':
    train_pipeline(
        cat_cols=['hour_category', 'age_category', 'account_type', 'gender', 'device_type'],
        num_cols=['amount', 'age'],
        columns_drop=[
            'transaction_id', 'timestamp', 'sender_id', 'receiver_id',
            'customer_id', 'cpf', 'pix_key', 'hour_date', 'minute_date'
        ]
    )
