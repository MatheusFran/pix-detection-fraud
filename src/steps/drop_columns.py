from zenml import step


@step
def drop_columns(df, list):
    df = df.drop(
        columns=[list])
    return df
