from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from prefect import task


@task
def evaluator(model_result, X_test, y_test):
    best_model = model_result["best_model"]
    best_name = model_result["best_name"]
    best_params = model_result["best_params"]

    preds = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    metrics_dict = {
        "model_name": best_name,
        "best_params": best_params,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": report
    }

    return metrics_dict
