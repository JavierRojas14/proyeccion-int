import datetime
import calendar

import pandas as pd
from holidays import country_holidays

FERIADOS_CHILE = country_holidays("CL")


def days_in_year(year=datetime.datetime.now().year):
    return 365 + calendar.isleap(year)


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


def add_lag_features(df, var_a_traer_valor):
    df = df.copy()

    target_map = df[var_a_traer_valor].to_dict()
    df["lag_1_anio"] = (df.index - pd.Timedelta("364 days")).map(target_map)
    df["lag_2_anios"] = (df.index - pd.Timedelta("728 days")).map(target_map)
    df["lag_3_anios"] = (df.index - pd.Timedelta("1092 days")).map(target_map)

    return df


def create_lag_features(df, column_name, lag_values, fill_value=None):
    """
    Create lag features in a DataFrame for a specific column.

    Parameters:
    - df: DataFrame
        The input DataFrame.
    - column_name: str
        The name of the column for which lag features will be created.
    - lag_values: list of ints
        A list of lag values.
    - fill_value: int or None, default=None
        The value to fill NaNs in lag features. If None, NaNs in lag features are left as NaNs.

    Returns:
    - new_df: DataFrame
        A new DataFrame with lag features and filled NaNs in lag features.
    """
    new_df = df.copy()

    for lag in lag_values:
        new_column = f"{column_name}_lag_{lag}"
        new_df[new_column] = new_df[column_name].shift(lag)
        if fill_value is not None:
            new_df[new_column] = new_df[new_column].fillna(fill_value)

    return new_df


def es_feriado(fecha):
    if FERIADOS_CHILE.get(fecha):
        return 1

    else:
        return 0


def obtener_tabla_resumen_egresos(
    tabla_dinamica_egresos_int,
    tabla_dinamica_egresos_pais,
    poblacion_teorica_hospitalizados,
    brecha_pais,
):
    tabla_dinamica_egresos_int_ppt = tabla_dinamica_egresos_int.astype(str)
    tabla_dinamica_egresos_int_ppt[[i for i in range(2021, 2036)]] = "0"

    tabla_dinamica_egresos_pais_ppt = tabla_dinamica_egresos_pais.astype(str)
    tabla_dinamica_egresos_pais_ppt[[i for i in range(2021, 2036)]] = "0"

    teorica_hospitalizados_ppt = (
        poblacion_teorica_hospitalizados.round(0).fillna(0).astype(int).astype(str)
    )

    brecha_pais_ppt = brecha_pais.round(2).astype(str).replace("nan", "-")

    tabla_resumen_ppt = (
        tabla_dinamica_egresos_int_ppt
        + "; "
        + tabla_dinamica_egresos_pais_ppt
        + "; "
        + teorica_hospitalizados_ppt
        + "; ("
        + brecha_pais_ppt
        + ")"
    )

    return tabla_resumen_ppt
