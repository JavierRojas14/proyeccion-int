import pandas as pd
from holidays import country_holidays

FERIADOS_CHILE = country_holidays("CL")


def create_features_datetime_index(df):
    """
    Create datetime features based on datetime index
    """
    df = df.copy()
    df["dayofweek"] = df.index.dayofweek
    df["quarter"] = df.index.quarter
    df["day"] = df.index.day
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["dayofyear"] = df.index.dayofyear
    df["weekend"] = (df["dayofweek"] > 4).astype(int)
    df["holidays"] = df.index.to_series().apply(es_feriado)

    return df


def add_lag_features(df):
    df = df.copy()

    target_map = df["n_egresos"].to_dict()
    df["lag_1_anio"] = (df.index - pd.Timedelta("364 days")).map(target_map)
    df["lag_2_anios"] = (df.index - pd.Timedelta("728 days")).map(target_map)
    df["lag_3_anios"] = (df.index - pd.Timedelta("1092 days")).map(target_map)

    return df


def es_feriado(fecha):
    if FERIADOS_CHILE.get(fecha):
        return 1

    else:
        return 0
