import pandas as pd
from prefect import task


@task(name="load_data")
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
