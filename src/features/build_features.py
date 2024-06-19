import datetime
import calendar

import pandas as pd
import numpy as np
from holidays import country_holidays

FERIADOS_CHILE = country_holidays("CL")


def filter_dataframes(dataframes, query_string):
    """
    Filter a list of DataFrames using a query string.

    Parameters:
        dataframes (dict): Dictionary of pandas DataFrames.
        query_string (str): Query string to filter the DataFrames.

    Returns:
        dict: Dictionary of filtered DataFrames.
    """
    filtered_dataframes = {key: df.query(query_string).copy() for key, df in dataframes.items()}
    return filtered_dataframes


def calculate_sum_columns_INE(dataframes_dict, columns):
    """
    Calculate the sum of specified columns for each DataFrame in a dictionary.

    Parameters:
        dataframes_dict (dict): Dictionary of pandas DataFrames.
        columns (list): List of column names for which sum needs to be calculated.

    Returns:
        pandas DataFrame: DataFrame containing the sums of specified columns for each DataFrame,
                          with DataFrame keys as index.
    """
    # Dictionary to store sums for each DataFrame
    sums = {}

    # Calculate sum for each DataFrame
    for key, df in dataframes_dict.items():
        # Select specified columns and calculate sum
        sums[key] = df[columns].sum()

    # Concatenate sums into a DataFrame
    result_df = pd.concat(sums, axis=1).T

    return result_df


def calculate_sum_columns_FONASA(dataframes_dict, columns):
    # Dictionary to store sums for each DataFrame
    sums = {}

    # Calculate sum for each DataFrame
    for key, df in dataframes_dict.items():
        # Select specified columns and calculate sum
        sums[key] = df.groupby("ANO_INFORMACION")[columns].sum()

    # Concatenate sums into a DataFrame
    result_df = pd.DataFrame(sums).transpose()

    return result_df


def iterate_queries(dataframes, query_strings, columns_to_sum, tipo_poblacion="INE"):
    """
    Iterate over a dictionary of query strings, filter DataFrames, and calculate the sum of
    specified columns.

    Parameters:
        dataframes (dict): Dictionary of pandas DataFrames.
        query_strings (dict): Dictionary of query strings.
        columns_to_sum (list): List of column names for which sum needs to be calculated.

    Returns:
        dict: Dictionary containing filtered DataFrames and their sums for each query string.
    """
    result = {}

    # Iterate over each query string
    for query_name, query_string in query_strings.items():
        # Filter DataFrames

        # If the query isn't null
        if query_string:
            filtered_dfs = filter_dataframes(dataframes, query_string)
        # If the query is null
        else:
            filtered_dfs = dataframes

        # If we want to know the INE demographic
        if tipo_poblacion == "INE":
            # Calculate sum of columns for filtered DataFrames
            sums_dfs = calculate_sum_columns_INE(filtered_dfs, columns_to_sum)

        # If we want to know the FONASA demographic
        elif tipo_poblacion == "FONASA":
            # Calculate sum of columns for FONASA filtered DataFrames
            sums_dfs = calculate_sum_columns_FONASA(filtered_dfs, columns_to_sum)

        # Store filtered DataFrames and their sums
        result[query_name] = sums_dfs

    return result


def multiple_dfs(df_dict, file_name, spaces):
    with pd.ExcelWriter(file_name) as writer:
        row = 0
        for key, df in df_dict.items():
            # Creates table header and writes it
            df_name = pd.Series()
            df_name.name = key
            df_name.to_excel(writer, startrow=row)
            row += 1

            # Saves DataFrame and updates row
            df.to_excel(writer, startrow=row)
            row += len(df.index) + spaces + 1


def assign_diagnosis(df, diag_code, new_diag1, new_diag2):
    """
    Creates two new DataFrames with the specified diagnosis assigned to DIAG1 and DIAG2 respectively,
    and returns the original DataFrame without modifications.

    Args:
        df (pandas.DataFrame): The original DataFrame.
        diag_code (str): The diagnosis code to be reassigned.
        new_diag1 (str): The new diagnosis code to be assigned to DIAG1.
        new_diag2 (str): The new diagnosis code to be assigned to DIAG2.

    Returns:
        pandas.DataFrame: The original DataFrame without modifications.
        pandas.DataFrame: The DataFrame with the first diagnosis assigned to DIAG1.
        pandas.DataFrame: The DataFrame with the second diagnosis assigned to DIAG2.
    """

    # Filter rows with the specified diagnosis code
    filtered_df = df.query(f"DIAG1 == '{diag_code}'")

    # Create new DataFrames with the specified diagnoses assigned to DIAG1 and DIAG2
    df_diag1 = filtered_df.copy()
    df_diag1["DIAG1"] = new_diag1

    df_diag2 = filtered_df.copy()
    df_diag2["DIAG1"] = new_diag2

    # Filters the specified diagnosis code out of the original DataFrame
    filtered_df_without_original_diagnosis = df.query(f"DIAG1 != '{diag_code}'")

    # Combine the modified DataFrames with the original DataFrame
    modified_df = pd.concat([filtered_df_without_original_diagnosis, df_diag1, df_diag2])

    # Return the original DataFrame and the modified DataFrames
    return modified_df


def preprocesar_egresos_multivariado(df):
    tmp = df.copy()

    # Genera variable y
    # Corresponde a los egresos del proximo mes
    tmp["n_egresos_proximo_mes"] = tmp.groupby("DIAG1")["n_egresos"].shift(-1)
    tmp = tmp.dropna()

    # Genera variables X (Lag, Diff, Rolling Mean y Variables de Fechas)
    tmp = create_grouped_lag_features(tmp, "n_egresos", "DIAG1", [1, 2, 3, 11, 12, 24])
    tmp["diff_1"] = tmp.groupby("DIAG1")["n_egresos"].diff(1)
    tmp["mean_4"] = (
        tmp.groupby("DIAG1")["n_egresos"].rolling(4).mean().reset_index(level=0, drop=True)
    )

    # Pone la fecha como indice y agrega variables relacionadas a la fecha
    tmp = tmp.set_index("FECHA_EGRESO")
    tmp = add_time_series_columns_by_month(tmp)

    return tmp


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


def create_grouped_lag_features(df, column_name, grouping_column, lag_values, fill_value=None):
    new_df = df.copy()

    for lag in lag_values:
        new_column = f"{column_name}_lag_{lag}"
        new_df[new_column] = new_df.groupby(grouping_column)[column_name].shift(lag)
        if fill_value is not None:
            new_df[new_column] = new_df[new_column].fillna(fill_value)

    return new_df


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
    # tabla_dinamica_egresos_int_ppt = tabla_dinamica_egresos_int.astype(str)
    tabla_dinamica_egresos_pais_ppt = tabla_dinamica_egresos_pais.astype(str)

    teorica_hospitalizados_ppt = (
        poblacion_teorica_hospitalizados.round(0).fillna(0).astype(int).astype(str)
    )

    brecha_pais_ppt = brecha_pais.round(2).astype(str).replace("nan", "-")

    tabla_resumen_ppt = (
        # tabla_dinamica_egresos_int_ppt
        # + "; "
        "Pais: "
        + tabla_dinamica_egresos_pais_ppt
        + "\n Teorica:"
        + teorica_hospitalizados_ppt
        + "\n Brecha:"
        + brecha_pais_ppt
    )

    return tabla_resumen_ppt


def to_sequences(dataset, seq_size=1):
    x = []
    y = []

    for i in range(len(dataset) - seq_size):
        window = dataset[i : (i + seq_size), 0]
        x.append(window)
        y.append(dataset[i + seq_size, 0])

    return np.array(x), np.array(y)


def calculate_discharges_metrics(df):
    metricas = pd.pivot_table(
        df,
        index="DIAG1",
        columns="ANO_EGRESO",
        values=["n_pacientes_distintos", "n_egresos", "dias_estada_totales"],
        aggfunc="sum",
        fill_value=0,
    ).sort_index()

    # Obtiene dias de estada promedio
    dias_estada_promedio = metricas["dias_estada_totales"] / metricas["n_egresos"]
    dias_estada_promedio.columns = [
        ("dias_estada_promedio", i) for i in dias_estada_promedio.columns
    ]

    # Obtiene los dias de estada promedio de todos los anios acumulados
    dias_estada_promedio_agrupado_en_anios = (
        metricas["dias_estada_totales"].sum(axis=1) / metricas["n_egresos"].sum(axis=1)
    ).to_frame()
    dias_estada_promedio_agrupado_en_anios.columns = [
        ("dias_estada_promedio_agrupado", "2017-2020")
    ]

    # Obtiene egresos por paciente
    egresos_por_paciente = metricas["n_egresos"] / metricas["n_pacientes_distintos"]
    egresos_por_paciente.columns = [
        ("egresos_por_paciente", i) for i in egresos_por_paciente.columns
    ]

    # Obtiene la cantidad de egresos por paciente de todos los anios acumulados
    egresos_por_paciente_agrupado_en_anios = (
        metricas["n_egresos"].sum(axis=1) / metricas["n_pacientes_distintos"].sum(axis=1)
    ).to_frame()

    egresos_por_paciente_agrupado_en_anios.columns = [
        ("egresos_por_paciente_agrupado", "2017-2020")
    ]

    metricas = pd.concat([metricas, dias_estada_promedio], axis=1)
    metricas = pd.concat([metricas, dias_estada_promedio_agrupado_en_anios], axis=1)
    metricas = pd.concat([metricas, egresos_por_paciente], axis=1)
    metricas = pd.concat([metricas, egresos_por_paciente_agrupado_en_anios], axis=1)

    return metricas
