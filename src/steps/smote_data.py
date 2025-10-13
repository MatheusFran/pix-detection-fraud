from imblearn.over_sampling import SMOTE
from zenml import step
from typing import Tuple
import pandas as pd


def smote_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    smote = SMOTE(
        sampling_strategy='auto',
        random_state=42
    )
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return X_resampled, y_resampled
