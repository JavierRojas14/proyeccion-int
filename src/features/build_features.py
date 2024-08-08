import datetime
import calendar
import math

import pandas as pd
import polars as pl
import numpy as np
from holidays import country_holidays

FERIADOS_CHILE = country_holidays("CL")


def procesar_resultados_por_estrato_y_grupos_etarios(
    dataframes, consultas, columnas, tipo_poblacion
):
    """
    Procesa los resultados por estratos y grupos etarios para un tipo de población dado.

    Parámetros:
        dataframes (dict): Diccionario de DataFrames de pandas. Contiene los distintos estratos
        a calcular ("Pais", "Region", "SSMO", etc)
        consultas (dict): Diccionario de consultas. Filtra por Grupos etarios
        (">15 anios", "entre 20 y 25", etc)
        columnas (list): Lista de nombres de columnas para calcular la suma. Puede ser
        "CUENTA BENEFICIARIO" o los anios a sumar en INE
        tipo_poblacion (str): Tipo de población ("INE" o "FONASA").

    Retorna:
        DataFrame: DataFrame concatenado con los resultados procesados y el índice restablecido.
    """
    # Obtener resultados por estratos y grupos etarios
    poblaciones_estratos_calculados = iterar_consultas(
        dataframes, consultas, columnas, tipo_poblacion
    )

    # Unir todos los resultados en un único DataFrame
    poblaciones_estratos_calculados = pd.concat(poblaciones_estratos_calculados)
    poblaciones_estratos_calculados = poblaciones_estratos_calculados.reset_index(
        names=["Edad Incidencia", "Estrato"]
    )

    return poblaciones_estratos_calculados


def iterar_consultas(dataframes, consultas, columnas_a_sumar, tipo_poblacion="INE"):
    """
    Iterar sobre un diccionario de consultas, filtrar DataFrames y calcular la suma de columnas
    especificadas.

    Parámetros:
        dataframes (dict): Diccionario de DataFrames de pandas.
        consultas (dict): Diccionario de consultas.
        columnas_a_sumar (list): Lista de nombres de columnas para calcular la suma.

    Retorna:
        dict: Diccionario que contiene DataFrames filtrados y sus sumas para cada consulta.
    """
    resultado = {}

    for nombre_consulta, consulta in consultas.items():
        # Filtrar DataFrames
        if consulta:
            dfs_filtrados = filtrar_dataframes(dataframes, consulta)
        else:
            dfs_filtrados = dataframes

        # Calcular suma de columnas según el tipo de población
        if tipo_poblacion == "INE":
            suma_dfs = calcular_suma_columnas(dfs_filtrados, columnas_a_sumar)
        elif tipo_poblacion == "FONASA":
            suma_dfs = calcular_suma_columnas_fonasa(dfs_filtrados, columnas_a_sumar)

        resultado[nombre_consulta] = suma_dfs

    return resultado


def calcular_suma_columnas(dataframes_dict, columnas):
    """
    Calcular la suma de columnas especificadas para cada DataFrame en un diccionario.

    Parámetros:
        dataframes_dict (dict): Diccionario de DataFrames de pandas.
        columnas (list): Lista de nombres de columnas para calcular la suma.

    Retorna:
        DataFrame: DataFrame que contiene las sumas de las columnas especificadas para
        cada DataFrame, con claves del DataFrame como índice.
    """
    sumas = {key: df[columnas].sum() for key, df in dataframes_dict.items()}
    return pd.DataFrame(sumas).T


def calcular_suma_columnas_fonasa(dataframes_dict, columnas):
    """
    Calcular la suma de columnas especificadas para cada DataFrame en un diccionario agrupado
    por 'ANO_INFORMACION'.

    Parámetros:
        dataframes_dict (dict): Diccionario de DataFrames de pandas.
        columnas (list): Lista de nombres de columnas para calcular la suma.

    Retorna:
        DataFrame: DataFrame que contiene las sumas de las columnas especificadas para
        cada DataFrame, con claves del DataFrame como índice.
    """

    sumas = {
        key: df.groupby("ANO_INFORMACION")[columnas].sum() for key, df in dataframes_dict.items()
    }
    return pd.DataFrame(sumas).T


def filtrar_dataframes(dataframes, consulta):
    """
    Filtrar una lista de DataFrames usando una consulta.

    Parámetros:
        dataframes (dict): Diccionario de DataFrames de pandas.
        consulta (str): Consulta para filtrar los DataFrames.

    Retorna:
        dict: Diccionario de DataFrames filtrados.
    """
    return {key: df.query(consulta).copy() for key, df in dataframes.items()}


def obtener_resumen_poblacion_pais(dataframe, anios_para_porcentaje):
    # Obtiene los estratos que se quieren comparar para calcular el % FONASA
    poblacion_a_comparar = dataframe.query(f"`Edad Incidencia` == 'todos'").set_index("Estrato")[
        anios_para_porcentaje
    ]

    # Obtiene la suma totales por estrato
    poblacion_a_comparar["suma_poblacion"] = poblacion_a_comparar.sum(axis=1)

    return poblacion_a_comparar


def obtener_porcentaje_de_fonasa_pais(poblaciones_ine, poblaciones_fonasa, anios_para_porcentaje):
    # Obtiene poblacion pais INE
    resumen_poblacion_pais_INE = obtener_resumen_poblacion_pais(
        poblaciones_ine, anios_para_porcentaje
    )

    # Obtiene poblacion pais FONASA
    resumen_poblacion_pais_FONASA = obtener_resumen_poblacion_pais(
        poblaciones_fonasa, anios_para_porcentaje
    )

    # Obtiene el porcentaje de FONASA por cada estrato a nivel pais
    porcentaje_fonasa = resumen_poblacion_pais_FONASA / resumen_poblacion_pais_INE
    porcentaje_fonasa = porcentaje_fonasa.rename(
        columns={"suma_poblacion": "porcentaje_fonasa_acumulado"}
    )

    # Obtiene el resumen total del % de FONASA a nivel pais
    resumen_porcentaje_fonasa = pd.concat(
        [resumen_poblacion_pais_INE, resumen_poblacion_pais_FONASA, porcentaje_fonasa], axis=1
    )

    return resumen_porcentaje_fonasa


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


def obtener_metricas_egresos(df, agrupacion):
    resumen = (
        df.group_by(agrupacion).agg(
            [
                pl.col("DIAG1").count().alias("n_egresos"),
                pl.col("ID_PACIENTE").n_unique().alias("n_pacientes_distintos"),
                pl.col("DIAS_ESTADA").sum().alias("dias_estada_totales"),
            ]
        )
    ).with_columns(
        [
            (pl.col("n_egresos") / pl.col("n_pacientes_distintos")).alias("egresos_por_paciente"),
            (pl.col("dias_estada_totales") / pl.col("n_egresos")).alias("dias_estada_promedio"),
        ]
    )

    return resumen


def calcular_resumen_metricas_desagregadas_y_agrupadas_en_anios(df, ano_inicio, ano_termino):
    df_filtrada = df.filter(
        (pl.col("ANO_EGRESO") >= ano_inicio) & (pl.col("ANO_EGRESO") <= ano_termino)
    )

    # Obtiene las metricas desagregadas por anio
    metricas_desagregadas = obtener_metricas_egresos(df_filtrada, ["ANO_EGRESO", "DIAG1"])
    metricas_desagregadas = metricas_desagregadas.sort(["ANO_EGRESO", "n_egresos"], descending=True)

    # Obtiene las metricas agrupadas en el periodo determinado
    metricas_agrupadas = obtener_metricas_egresos(df_filtrada, ["DIAG1"])
    metricas_agrupadas = metricas_agrupadas.sort(["DIAG1"], descending=True)

    # Convierte los resumenes a pandas
    metricas_desagregadas = metricas_desagregadas.to_pandas()
    metricas_agrupadas = metricas_agrupadas.to_pandas().set_index("DIAG1")

    # Pivota la tabla desagregada
    metricas_desagregadas = pd.pivot_table(
        metricas_desagregadas,
        index="DIAG1",
        columns="ANO_EGRESO",
        values=["n_egresos", "n_pacientes_distintos", "dias_estada_totales"],
        aggfunc="sum",
        fill_value=0,
    )

    # Cambia las columnas de la tabla agrupada en el periodo
    columnas_agrupado = [
        (f"agrupado_entre_{ano_inicio}_{ano_termino}", col) for col in metricas_agrupadas.columns
    ]
    metricas_agrupadas.columns = pd.MultiIndex.from_tuples(columnas_agrupado)

    # Concatena las metircs desagregadas y agregadas
    resumen = pd.concat([metricas_desagregadas, metricas_agrupadas], axis=1)

    return resumen


def obtener_camas(camas_al_2035, fraccion_uci, fraccion_uti, fraccion_medias):
    camas_uci = camas_al_2035 * fraccion_uci
    camas_uti = camas_al_2035 * fraccion_uti
    camas_medias = camas_al_2035 * fraccion_medias
    return camas_al_2035, camas_uci, camas_uti, camas_medias


def leer_casos_area_de_influencia(columnas_poblacion_interes):
    # Obtiene los casos de area de influencia
    casos_area_de_influencia = pd.read_excel(
        "../data/interim/casos_teoricos_diagnosticos.xlsx",
        sheet_name="casos_area_de_influencia_INT",
    )

    # Preprocesa el diagnostico
    casos_area_de_influencia["Diagnostico"] = (
        casos_area_de_influencia["Diagnostico"].str.split(" - ").str[0]
    )

    # Preprocesa los diagnosticos agrupados
    casos_area_de_influencia["Diagnosticos Contenidos"] = casos_area_de_influencia[
        "Diagnosticos Contenidos"
    ].str.split(", ")

    # Renombra columnas de la poblacion
    casos_area_de_influencia = casos_area_de_influencia.rename(columns=columnas_poblacion_interes)

    # Pone como indice el diagnostico
    casos_area_de_influencia = casos_area_de_influencia.set_index("Diagnostico")

    return casos_area_de_influencia
