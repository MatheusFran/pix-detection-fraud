from prefect import task, flow
from sklearn.ensemble import RandomForestClassifier
import mlflow
import joblib

from src.data.split_data import split_data
from src.features.build_pipe_fe import build_pipe_fe
from src.models.evaluator import evaluator
from src.models.predict import predict
from src.models.train import train_model


@task
def get_data():
    X_train, y_train, X_test, y_test = split_data()
    return X_train, y_train, X_test, y_test


@task
def builder_pipe_train(X_train, y_train):
    best_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42
    )
    full_pipe = build_pipe_fe()

    final_pipe_train = train_model(best_model, full_pipe, X_train, y_train)

    return final_pipe_train


@task
def predict_model(pipe, X_test, y_test):
    y_pred = predict(pipe, X_test)
    metrics = evaluator(y_pred, y_test)
    return metrics


@flow
def ml_pipe_train():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("train_pix_fraud_v1")

    with mlflow.start_run():
        X_train, y_train, X_test, y_test = get_data()
        pipe = builder_pipe_train(X_train, y_train)
        metrics = predict_model(pipe, X_test, y_test)

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        mlflow.log_params({
            "model": "RandomForestClassifier",
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_split": 2,
            "random_state": 42
        })

        path_pipeline = "../artifacts/pipeline/model_pipe.pkl"
        joblib.dump(pipe, path_pipeline)
        mlflow.log_artifact(path_pipeline)

        return metrics
