from zenml import step

from src.models.evaluate import evaluator


@step
def evaluate_model():
    evaluator()