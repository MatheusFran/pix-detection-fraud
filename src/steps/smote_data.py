from imblearn.over_sampling import SMOTE
from zenml import step


@step
def smote_data(X_train, y_train):
    smote = SMOTE(
        sampling_strategy='auto',
        random_state=42
    )
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    return X_resampled, y_resampled
