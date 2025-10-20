from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def evaluator(pred, y_test):
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    report = classification_report(y_test, pred, output_dict=True)

    metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": report
    }

    return metrics_dict
