from prefect import task

from src.models.model import RegressionLogisticModel, RandomForestClassifierModel
from src.models.model_selection import ModelSelection


@task
def select_model(X, y):
    models = [
        RegressionLogisticModel(),
        RandomForestClassifierModel()
    ]

    param_grids = [
        {  # LogisticRegression
            'classifier__C': [0.1, 1, 10],
            'classifier__penalty': ['l2'],
            'classifier__solver': ['lbfgs']
        },
        {  # RandomForest
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20]
        }
    ]

    selection = ModelSelection(
        models=models,
        param_grids=param_grids,
        X=X,
        y=y,
    )

    return selection
