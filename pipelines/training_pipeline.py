from sklearn.preprocessing import StandardScaler
from zenml import Model, pipeline, step

from src.features.feature_engineering import FeatureEngineer, OneHotEncoding, StandardScaling
from src.steps.drop_columns import drop_columns
from src.steps.feature_engineer import builder_features
from src.steps.load_data import load_data
from src.steps.preprocessing import preprocessing
from src.steps.smote_data import smote_data
from src.steps.split_data import split_data


@pipeline(
    model=Model(name="fraud_detection_pix")
)
def ml_pipeline():
    df = load_data()

    df = builder_features(df)

    list_drop = [
        'transaction_id',
        'timestamp',
        'sender_id',
        'receiver_id',
        'customer_id',
        'cpf',
        'pix_key',
        'hour_date',
        'minute_date'
    ]
    df = drop_columns(df, list_drop)

    X_train, X_test, y_train, y_test = split_data(df, 'fraud')
    X_train, y_train = smote_data(X_train, y_train)

    features_preprocessing = {
        'onehotenconder': ['hour_category', 'age_category', 'account_type', 'gender', 'device_type'],
        'standardscaler': ['amount', 'age'],
    }
    fe_pipeline = preprocessing(features_preprocessing)

    # Seleção de modelos

    # avalia o melhor modelo


if __name__ == "__main__":
    ml_pipeline()
