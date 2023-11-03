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


def add_lag_features(df, var_a_traer_valor):
    df = df.copy()

    target_map = df[var_a_traer_valor].to_dict()
    df["lag_1_anio"] = (df.index - pd.Timedelta("364 days")).map(target_map)
    df["lag_2_anios"] = (df.index - pd.Timedelta("728 days")).map(target_map)
    df["lag_3_anios"] = (df.index - pd.Timedelta("1092 days")).map(target_map)

    return df


def es_feriado(fecha):
    if FERIADOS_CHILE.get(fecha):
        return 1

    else:
        return 0


def obtener_tabla_resumen_egresos(
    poblacion_teorica_hospitalizados, tabla_dinamica_egresos_pais, brecha_pais
):
    teorica_hospitalizados_ppt = (
        poblacion_teorica_hospitalizados.round(0).fillna(0).astype(int).astype(str)
    )
    tabla_dinamica_egresos_pais_ppt = tabla_dinamica_egresos_pais.copy().astype(str)
    tabla_dinamica_egresos_pais_ppt[[i for i in range(2021, 2036)]] = "0"

    brecha_pais_ppt = brecha_pais.copy()
    brecha_pais_ppt = brecha_pais_ppt.round(2).astype(str).replace("nan", "-")

    tabla_resumen_ppt = (
        tabla_dinamica_egresos_pais_ppt
        + "; "
        + teorica_hospitalizados_ppt
        + "; ("
        + brecha_pais_ppt
        + ")"
    )

    return tabla_resumen_ppt
