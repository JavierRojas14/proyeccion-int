import pandas as pd


def create_features_datetime_index(df):
    """
    Create datetime features based on datetime index
    """
    df = df.copy()
    df["dayofweek"] = df.index.dayofweek
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["dayofyear"] = df.index.dayofyear

    return df


def add_lag_features(df):
    df = df.copy()

    target_map = df["n_egresos"].to_dict()
    df["lag_1_anio"] = (df.index - pd.Timedelta("364 days")).map(target_map)
    df["lag_2_anios"] = (df.index - pd.Timedelta("728 days")).map(target_map)
    df["lag_3_anios"] = (df.index - pd.Timedelta("1092 days")).map(target_map)

    return df
