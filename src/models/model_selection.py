from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV
import yaml


class ModelSelection:
    def __init__(self, models, param_grids, X, y, config_path="../../config_train.yml"):
        self.models = models
        self.param_grids = param_grids
        self.config_path = config_path
        self.X = X
        self.y = y

    def _grid_search(self, scoring):
        grid_search = None
        for model, param_grid in zip(self.models, self.param_grids):
            pipeline_model = ImbPipeline([
                ('classifier', model.model)
            ])

            grid_search = GridSearchCV(
                estimator=pipeline_model,
                param_grid=param_grid,
                cv=3,
                scoring=scoring,
                n_jobs=-1
            )

            grid_search.fit(self.X, self.y)

        return grid_search

    def _best_info(self, grid_search):
        best_score = -1
        best_info = None

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_info = {
                "best_params": grid_search.best_params_,
                "best_score": grid_search.best_score_,
                "best_model": grid_search.best_estimator_
            }

        return best_info, best_score

    def save_info_yml(self, info):
        best_info, best_score = info
        config = {
            "model_selection": {
                "best_params": best_info["best_params"],
                "best_model": best_info["best_model"]
            },
        }

        with open(self.config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)

    def save_mlflow(self):
        pass

    def __call__(self):
        model = self._grid_search(scoring="f1")
        self.save_info_yml(self._best_info(model))
        return model
