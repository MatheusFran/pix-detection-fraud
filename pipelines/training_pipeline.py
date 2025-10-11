from zenml import step, pipeline
import pandas as pd


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
def split_data():
    pass


@step
def preprocessor_data():
    pass


@step
def train_model():
    pass


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
