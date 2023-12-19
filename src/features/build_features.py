import datetime
import calendar

import pandas as pd
import numpy as np
from holidays import country_holidays

FERIADOS_CHILE = country_holidays("CL")


def preprocesar_egresos_multivariado(df):
    tmp = df.copy()

    # Genera variable y
    # Corresponde a los egresos del proximo mes
    tmp["n_egresos_proximo_mes"] = tmp.groupby("DIAG1")["n_egresos"].shift(-1)
    tmp = tmp.dropna()

    # Genera variables X (Lag, Diff, Rolling Mean y Variables de Fechas)
    tmp["lag_1"] = tmp.groupby("DIAG1")["n_egresos"].shift(1)
    tmp["diff_1"] = tmp.groupby("DIAG1")["n_egresos"].diff(1)
    tmp["mean_4"] = (
        tmp.groupby("DIAG1")["n_egresos"].rolling(4).mean().reset_index(level=0, drop=True)
    )

    # Pone la fecha como indice y agrega variables relacionadas a la fecha
    tmp = tmp.set_index("FECHA_EGRESO")
    tmp = add_time_series_columns_by_month(tmp)

    return tmp


def days_in_year(year=datetime.datetime.now().year):
    return 365 + calendar.isleap(year)


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


def add_time_series_columns_by_month(df):
    """
    Add time series related columns to a DataFrame with a DateTimeIndex aggregated by month.

    Parameters:
    - df: pandas DataFrame with a single continuous column and a DateTimeIndex aggregated by month.

    Returns:
    - pandas DataFrame with added time series related columns.
    """
    df = df.copy()
    # Extract date-related information
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["season"] = (df["month"] % 12 + 3) // 3  # Calculate season based on month
    df["is_leap_year"] = df["is_leap_year"] = df.index.is_leap_year.astype(int)

    # Check if the month is part of the start/end of the quarter
    df["is_quarter_start"] = df.index.is_quarter_start.astype(int)
    df["is_quarter_end"] = df.index.is_quarter_end.astype(int)

    # Calculate days per month
    df["days_in_month"] = df.index.days_in_month

    anio_inicio = df.index[0].year
    anio_termino = df.index[-1].year

    # Calculate holidays per month
    df["holidays_per_month"] = calcular_feriados_por_mes(anio_inicio, anio_termino, "CL")

    # Calculate weekends per month
    df["weekends_per_month"] = calcular_fin_de_semana_por_mes(anio_inicio, anio_termino)

    # Calculate business days per month
    df["business_days_per_month"] = calcular_dias_laborales_por_mes(anio_inicio, anio_termino)

    return df


def calcular_feriados_por_mes(ano_inicio, ano_termino, pais):
    # Obtiene feriados en el periodo
    feriados_periodo = country_holidays(pais, years=[i for i in range(ano_inicio, ano_termino + 1)])
    # Transforma los feriados a DataFrame y agrega conteo por mes
    feriados_periodo = pd.DataFrame(feriados_periodo.values(), index=feriados_periodo.keys())
    feriados_periodo.index = pd.to_datetime(feriados_periodo.index)
    feriados_periodo = feriados_periodo.resample("M").count()[0]

    return feriados_periodo


def calcular_fin_de_semana_por_mes(ano_inicio, ano_termino):
    """Funcion que permite obtener la cantidad de fin de semanas por mes"""
    fin_de_semana = pd.DataFrame(
        index=pd.bdate_range(
            start=f"{ano_inicio}-01-01",
            end=f"{ano_termino}-12-31",
            freq="C",
            weekmask="Sat Sun",
        )
    )
    fin_de_semana["n_fin_de_semanas"] = 1
    fin_de_semana = fin_de_semana.resample("M").sum()["n_fin_de_semanas"]

    return fin_de_semana


def calcular_dias_laborales_por_mes(ano_inicio, ano_termino):
    """Funcion que permite obtener la cantidad de dias laborales por mes. Aquí se incluyen los
    días feriados."""

    dias_laborales = pd.DataFrame(
        index=pd.bdate_range(start=f"{ano_inicio}-01-01", end=f"{ano_termino}-12-31", freq="B")
    )

    dias_laborales["n_dias_laborales"] = 1
    dias_laborales = dias_laborales.resample("M").sum()["n_dias_laborales"]

    return dias_laborales


def to_sequences(dataset, seq_size=1):
    x = []
    y = []

    for i in range(len(dataset) - seq_size):
        window = dataset[i : (i + seq_size), 0]
        x.append(window)
        y.append(dataset[i + seq_size, 0])

    return np.array(x), np.array(y)
