from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import mlflow
from prefect import task


@task
def evaluator(model_pipeline, X_test, y_test):
    preds = model_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": report
    }

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, preds))

    mlflow.log_metric("accuracy_test", accuracy)
    mlflow.log_metric("precision_test", precision)
    mlflow.log_metric("recall_test", recall)
    mlflow.log_metric("f1_test", f1)

    return metrics_dict
