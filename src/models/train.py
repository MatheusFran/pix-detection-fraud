from sklearn.pipeline import Pipeline


def train_model(model, feature_engineer, X_train, y_train):
    full_pipe = Pipeline([
        ('features', feature_engineer),
        ('model', model)
    ])

    full_pipe.fit(X_train, y_train)

    return full_pipe
