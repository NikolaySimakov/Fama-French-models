import pandas as pd


def stock_df_format(df: pd.DataFrame) -> bool:
    expected_columns = ['Date', 'Symbol', 'Adj Close',
                        'Close', 'High', 'Low', 'Open', 'Volume']
    df_columns = df.columns.tolist()
    return set(expected_columns).issubset(set(df_columns))
