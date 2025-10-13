from sklearn.pipeline import Pipeline
from zenml import step
import mlflow.sklearn


@step
def train_model(X_train, y_train, fe_pipeline, model_selection, grid_selection):
    best_model = grid_selection["best_model"]

    model_pipeline = Pipeline([
        ('features', fe_pipeline),
        ('classifier', model_selection())
    ])

    model_pipeline.fit(X_train, y_train)

    mlflow.sklearn.log_model(model_pipeline, artifact_path="fraud_model")

    return model_pipeline
