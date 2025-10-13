from zenml import Model, pipeline
import mlflow

from src.models.model_selection import model_selection
from src.steps.evaluate import evaluator
from src.steps.feature_engineer import builder_features
from src.steps.load_data import load_data
from src.steps.preprocessing import preprocessing
from src.steps.smote_data import smote_data
from src.steps.split_data import split_data
from src.steps.train_model import train_model


@pipeline(
    model=Model(name="fraud_detection_pix")
)
def ml_pipeline():
    mlflow.set_experiment("fraud_detection_pix")
    mlflow.set_tracking_uri("http://localhost:5000")

    df = load_data()
    df = builder_features(df)
    X_train, X_test, y_train, y_test = split_data(df, 'fraud')
    X_train, y_train = smote_data(X_train, y_train)
    X_train_processing = preprocessing(X_train)
    grid_selection = model_selection(X_train_processing, y_train)

    model_pipeline = train_model(
        X_train_processing,
        y_train,
        preprocessing,
        grid_selection
    )

    metrics_dict = evaluator(model_pipeline, X_test, y_test)

    return metrics_dict


if __name__ == "__main__":
    pipe = ml_pipeline()
    pipe.run()
