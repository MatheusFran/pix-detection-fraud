from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV
import mlflow

from src.models.model import RegressionLogisticModel, RandomForestClassifierModel

mlflow.autolog()


def model_selection(X, y):
    best_score = -1

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

    for model, param_grid in zip(models, param_grids):
        pipeline_model = ImbPipeline([
            ('classifier', model.model)
        ])

        grid_search = GridSearchCV(
            estimator=pipeline_model,
            param_grid=param_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1
        )

        grid_search.fit(X, y)

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_info = {
                "model": model.__class__.__name__,
                "best_params": grid_search.best_params_,
                "best_score": grid_search.best_score_,
                "best_model": grid_search.best_estimator_
            }

    return best_info
