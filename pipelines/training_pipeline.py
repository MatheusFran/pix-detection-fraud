from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from zenml import step, pipeline
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline

from src.features.new_features import NewFeature, TimeFeatureStrategy, AgeCategoryFeatureStrategy, \
    HourCategoryFeatureStrategy
from src.models.model import RegressionLogisticModel, NewModel, RandomForestClassifierModel


@step
def load_data():
    df_transactions = pd.read_csv('../data/raw/transactions.csv', encoding='utf-8')
    df_customers = pd.read_csv('../data/raw/customers.csv', encoding='utf-8')
    df = pd.merge(
        df_transactions,
        df_customers,
        left_on='sender_id',
        right_on='customer_id',
        how='left'
    )
    return df


@step
def split_data(data):
    X = data.drop(columns=['fraud'])
    y = data['fraud']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


@step
def new_features(X_train, X_test, y_train, y_test):
    features = [
        TimeFeatureStrategy(),
        AgeCategoryFeatureStrategy(),
        HourCategoryFeatureStrategy()
    ]

    for f in features:
        processor = NewFeature(f)
        train = processor.apply(train)
        test = processor.apply(test)


@step
def preprocessor_data(train, test):

    list_onehot = ['hour_category', 'age_category', 'account_type', 'gender', 'device_type']
    list_num = ['amount', 'age']

    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    num_pipeline = Pipeline([
        ('num', StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, list_num),
        ('cat', cat_pipeline, list_onehot),
    ])

    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE()),
    ])

    return pipeline


@step
def train_model():

    log_model = RegressionLogisticModel()
    model = NewModel(strategy=log_model)
    y_pred = model.apply(X_train, y_train, X_test)

    rf_model = RandomForestClassifierModel(n_estimators=100)
    model = NewModel(strategy=rf_model)
    y_pred_rf = model.apply(X_train, y_train, X_test)

    # GridSearchCV


@step
def evaluate_model():
    pass


@pipeline
def train_pipeline():
    df = load_data()
    train, test = split_data(df)
    X_train, y_train, X_test, y_test = preprocessor_data(train, test)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    pass
