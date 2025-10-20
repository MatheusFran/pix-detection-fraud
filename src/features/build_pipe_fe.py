from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline

from src.features.new_features import TimeFeatureStrategy, AgeCategoryFeatureStrategy, HourCategoryFeatureStrategy
from src.features.transformers.create_fe import CreateNewfeatures
from src.features.transformers.drop_fe import DropColumnsFe
from src.features.transformers.select_cols import DataFrameSelector


def build_pipe_fe():
    features = [
        TimeFeatureStrategy(),
        AgeCategoryFeatureStrategy(),
        HourCategoryFeatureStrategy()
    ]

    cat_cols = ['hour_category', 'age_category', 'account_type', 'gender', 'device_type'],
    num_cols = ['amount', 'age'],
    columns_drop = [
        'transaction_id', 'timestamp', 'sender_id', 'receiver_id',
        'customer_id', 'cpf', 'pix_key', 'hour_date', 'minute_date'
    ]

    num_fe = Pipeline([
        ('selector', DataFrameSelector(num_cols)),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_fe = Pipeline([
        ('selector', DataFrameSelector(cat_cols)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    pipe_transform = FeatureUnion([
        ('num_fe', num_fe),
        ('cat_fe', cat_fe)
    ])

    steps = [
        ('new_feature', CreateNewfeatures(features)),
        ('drop_cols', DropColumnsFe(columns_drop)),
        ('preprocessing', pipe_transform),
        ('smote', SMOTE(random_state=42))
    ]

    full_pipe = ImbPipeline(steps)

    return full_pipe
