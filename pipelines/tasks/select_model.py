from prefect import task
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV


@task
def select_model(X, y):
    best_score = -1
    best_model = None
    best_name = None
    best_params = None

    models_params = [
        {
            'name': 'LogisticRegression',
            'model': LogisticRegressionCV(max_iter=1000, class_weight='balanced'),
            'param_grid': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            }
        },
        {
            'name': 'RandomForest',
            'model': RandomForestClassifier(class_weight='balanced'),
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5]
            }
        }
    ]

    for mp in models_params:
        grid = GridSearchCV(
            estimator=mp['model'],
            param_grid=mp['param_grid'],
            cv=3,
            scoring='f1'
        )
        grid.fit(X, y)

        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
            best_name = mp['name']
            best_params = grid.best_params_

    return {
        'best_model': best_model,
        'best_name': best_name,
        'best_params': best_params
    }
