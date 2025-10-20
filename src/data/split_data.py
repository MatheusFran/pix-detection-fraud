import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(target='fraud'):
    df_transactions = pd.read_csv('../data/raw/transactions.csv', encoding='utf-8')
    df_customers = pd.read_csv('../data/raw/customers.csv', encoding='utf-8')
    df = pd.merge(
        df_transactions,
        df_customers,
        left_on='sender_id',
        right_on='customer_id',
        how='left'
    )

    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
