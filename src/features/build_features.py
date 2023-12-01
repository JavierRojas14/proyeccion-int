import datetime
import calendar

import pandas as pd
from holidays import country_holidays

FERIADOS_CHILE = country_holidays("CL")


def days_in_year(year=datetime.datetime.now().year):
    return 365 + calendar.isleap(year)


def obtener_aumento_de_procedimiento_especifico(proceds_aumentados, glosa_procedimiento_a_incluir):
    resumen_aumento = (
        proceds_aumentados[
            proceds_aumentados["Descripción_x"].str.contains(glosa_procedimiento_a_incluir)
        ]
        .groupby("ANIO_EGRESO")[["cantidad_procedimientos", "cantidad_procedimientos_aumentados"]]
        .sum()
    )

    return resumen_aumento


def obtener_cantidad_de_dias_laborales_por_anio(fecha_inicio, fecha_termino):
    dias_laborales = pd.DataFrame(
        {
            "fecha": pd.date_range(
                start=fecha_inicio,
                end=fecha_termino,
                freq="B",
            )
        }
    )

    dias_laborales["es_feriado"] = dias_laborales["fecha"].apply(es_feriado)
    dias_laborales = dias_laborales.query("es_feriado == 0")
    dias_laborales = dias_laborales.groupby(dias_laborales["fecha"].dt.year).size()
    dias_laborales.index = pd.to_datetime(dias_laborales.index, format="%Y")

    return dias_laborales


def create_features_datetime_index(df):
    """
    Create datetime features based on datetime index
    """
    df = df.copy()

    # Extract basic datetime features
    df["dayofweek"] = df.index.dayofweek
    df["quarter"] = df.index.quarter
    df["day"] = df.index.day
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["dayofyear"] = df.index.dayofyear
    df["weekend"] = (df["dayofweek"] > 4).astype(int)
    df["holidays"] = df.index.to_series().apply(es_feriado)

    # Add season information based on Chile's temporal data
    df["season"] = (df.index.month % 12) // 3 + 1

    # Additional time series features
    df["is_leap_year"] = df.index.is_leap_year.astype(int)
    df["is_month_start"] = df.index.is_month_start.astype(int)
    df["is_month_end"] = df.index.is_month_end.astype(int)
    df["is_quarter_start"] = df.index.is_quarter_start.astype(int)
    df["is_quarter_end"] = df.index.is_quarter_end.astype(int)

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
