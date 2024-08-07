# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


import pandas as pd
import numpy as np

import polars as pl

import glob


TRANSFORMACION_ESTRATOS_FONASA = TRANSFORMACION_ESTRATOS_FONASA = {
    "DE ANTOFAGASTA": "Antofagasta",
    "DE ARICA Y PARINACOTA": "Arica y Parinacota",
    "DE ATACAMA": "Atacama",
    "DE AYSÉN DEL GRAL. C. IBÁÑEZ DEL CAMPO": "Aysén del General Carlos Ibáñez del Campo",
    "DE COQUIMBO": "Coquimbo",
    "DE LA ARAUCANÍA": "La Araucanía",
    "DE LOS LAGOS": "Los Lagos",
    "DE LOS RÍOS": "Los Ríos",
    "DE MAGALLANES Y DE LA ANTÁRTICA CHILENA": "Magallanes y de la Antártica Chilena",
    "DE TARAPACÁ": "Tarapacá",
    "DE VALPARAÍSO": "Valparaíso",
    "DE ÑUBLE": "Ñuble",
    "DEL BÍOBÍO": "Biobío",
    "DEL LIBERTADOR GENERAL BERNARDO O'HIGGINS": "Libertador General Bernardo O'Higgins",
    "DEL MAULE": "Maule",
    "METROPOLITANA DE SANTIAGO": "Metropolitana de Santiago",
}

CAMBIOS_COLUMNAS_INNOMINADOS = {
    "MES_INFORMACION": "MES_INFORMACION",
    "TITULAR_CARGA": "TITULAR_CARGA",
    "TRAMO_FONASA": "TRAMO",
    "SEXO": "SEXO",
    "EDAD_TRAMO": "EDAD_TRAMO",
    "NACIONALIDAD": "NACIONALIDAD",
    "TIPO_ASEGURADO": "TIPO_ASEGURADO",
    "REGIÓN_BENEFICIARIO": "REGION",
    "COMUNA_BENEFICIARIO": "COMUNA",
}

COLUMNAS_A_OCUPAR_FONASA = [
    "MES_INFORMACION",
    "TITULAR_CARGA",
    "TRAMO",
    "SEXO",
    "EDAD_TRAMO",
    "NACIONALIDAD",
    "REGION",
    "COMUNA",
    "CUENTA_BENEFICIARIOS",
]


def procesar_ine(ruta_base_de_datos):
    print("> Procesando Base de datos INE")
    # Define ruta al archivo INE
    ruta_a_ine = (
        f"{ruta_base_de_datos}/1_poblacion_ine/estimaciones-y-proyecciones-2002-2035-comunas.xlsx"
    )
    # Lee la base de datos
    df = pd.read_excel(ruta_a_ine).iloc[:-2]

    # Elimina el sufijo 'Poblacion' de las columnas
    df.columns = df.columns.str.replace("Poblacion ", "")

    # Renombra columna de hombres o mujeres
    df = df.rename(columns={"Sexo\n1=Hombre\n2=Mujer": "hombre_mujer"})

    # Convierte el nombre de las comunas a title y elimina espacios en blanco
    df["Nombre Comuna"] = df["Nombre Comuna"].str.title().str.strip()

    # Indica si es adulto o infantil
    df["grupo_etario_poblacion"] = np.where(df["Edad"] >= 15, "Adulto", "Infantil")

    return df


def procesar_fonasa(ruta_base_de_datos):
    print("> Procesando Base de datos FONASA")
    # Indica la ruta de los archivos FONASA
    ruta_a_fonasa = f"{ruta_base_de_datos}/2_poblacion_fonasa/*.csv"

    # Encuentra todos los archivos que coinciden con el patrón
    archivos = glob.glob(ruta_a_fonasa)

    # Lee y concatena todos los archivos CSV
    df_fonasa = pd.concat(
        pd.read_csv(archivo, encoding="latin-1", usecols=COLUMNAS_A_OCUPAR_FONASA)
        for archivo in archivos
    )

    # Extrae el año de la columna MES_INFORMACION
    df_fonasa["ANO_INFORMACION"] = df_fonasa["MES_INFORMACION"].astype(str).str[:4]

    # Ordena la base de datos segun el anio de informacion
    df_fonasa = df_fonasa.sort_values("ANO_INFORMACION")

    # Formatea la columna REGION
    df_fonasa["REGION"] = df_fonasa["REGION"].str.upper().str.strip()
    df_fonasa["REGION"] = df_fonasa["REGION"].replace(
        {
            "ÑUBLE": "DE ÑUBLE",
            "DEL LIBERTADOR B. O'HIGGINS": "DEL LIBERTADOR GENERAL BERNARDO O'HIGGINS",
        }
    )

    # Elimina los registros con REGION desconocida o nula
    df_fonasa = df_fonasa.query("REGION != 'DESCONOCIDA'").copy()
    df_fonasa = df_fonasa.query("REGION.notna()").copy()

    # Aplica la transformación de estratos
    df_fonasa["REGION"] = df_fonasa["REGION"].replace(TRANSFORMACION_ESTRATOS_FONASA)

    # Formatea otras columnas
    df_fonasa["SEXO"] = df_fonasa["SEXO"].str.upper().str.strip()
    # df_fonasa["SERVICIO_SALUD"] = df_fonasa["SERVICIO_SALUD"].str.upper().str.strip()
    df_fonasa["COMUNA"] = df_fonasa["COMUNA"].str.title().str.strip()

    # Limpia y transforma la columna EDAD_TRAMO
    df_fonasa["EDAD_TRAMO"] = df_fonasa["EDAD_TRAMO"].replace(
        {"Más de 99 años": "99 años", "S.I.": "-1", "Sin información": "-1"}
    )
    df_fonasa["EDAD_TRAMO"] = df_fonasa["EDAD_TRAMO"].str.split().str[0].astype(int)

    return df_fonasa


def procesar_FONASA_innominadas(ruta_base_de_datos):
    print("> Procesando bases de datos FONASA innominadas")

    # Define la ruta de archivos innominados
    ruta_a_fonasa = (
        f"{ruta_base_de_datos}/2_poblacion_fonasa/innominados/Beneficiarios Fonasa 2023.csv"
    )

    df = pl.read_csv(ruta_a_fonasa, encoding="latin-1")

    cuenta_fonasa = (
        df.rename(CAMBIOS_COLUMNAS_INNOMINADOS)
        .group_by(CAMBIOS_COLUMNAS_INNOMINADOS.values())
        .len(name="CUENTA_BENEFICIARIOS")
    )

    return cuenta_fonasa


### Planilla de Incidencias ###


def leer_planilla_incidencias(ruta_planilla, columnas_a_utilizar):
    """
    Lee la planilla de incidencias y devuelve un DataFrame filtrado por las columnas especificadas.
    """
    return pd.read_excel(ruta_planilla, usecols=columnas_a_utilizar)


def filtrar_diagnosticos_limitados_por_oferta(df_incidencias):
    """
    Filtra los diagnósticos limitados por oferta de un DataFrame de incidencias.
    """
    incidencias_sin_acotados_por_oferta = df_incidencias.query(
        "`Estadística` != 'Acotado por oferta'"
    )
    acotados_por_oferta = df_incidencias.query("`Estadística` == 'Acotado por oferta'")
    return incidencias_sin_acotados_por_oferta, acotados_por_oferta


def convertir_incidencia_a_numeros(df_incidencias):
    """
    Convierte las incidencias a números e imputa las faltantes como NaN.
    """
    df_incidencias["Casos (Cada 100.000)"] = pd.to_numeric(
        df_incidencias["Casos (Cada 100.000)"], errors="coerce"
    )
    return df_incidencias


def calcular_rate_incidencia(df_incidencias):
    """
    Calcula la fracción de la incidencia para multiplicar con la población directamente.
    """
    df_incidencias["rate_incidencia"] = df_incidencias["Casos (Cada 100.000)"] / 100000
    return df_incidencias


def corregir_prevalencias(df_incidencias):
    """
    Corrige las prevalencias en 5 años.
    """
    idx_prevalencias = df_incidencias["Estadística"] == "Prevalencia"
    df_incidencias.loc[idx_prevalencias, "rate_incidencia"] = (
        df_incidencias.loc[idx_prevalencias, "rate_incidencia"] / 5
    )
    return df_incidencias


def procesar_incidencias(ruta_planilla, columnas_a_utilizar):
    """
    Procesa las incidencias desde la lectura hasta la corrección de prevalencias.
    """
    incidencias = leer_planilla_incidencias(ruta_planilla, columnas_a_utilizar)
    incidencias, limitados_por_oferta = filtrar_diagnosticos_limitados_por_oferta(incidencias)
    incidencias = convertir_incidencia_a_numeros(incidencias)
    incidencias = calcular_rate_incidencia(incidencias)
    incidencias = corregir_prevalencias(incidencias)
    return incidencias, limitados_por_oferta


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # Procesa Y guarda FONASA innominados
    fonasa_innonimados_procesada = procesar_FONASA_innominadas(input_filepath)
    ruta_output_fonasa_innominados = (
        f"{input_filepath}/2_poblacion_fonasa/Beneficiarios Fonasa 2023.csv"
    )
    fonasa_innonimados_procesada.to_pandas().to_csv(
        ruta_output_fonasa_innominados, encoding="latin-1"
    )

    # Procesa INE y FONASA
    ine_procesada = procesar_ine(input_filepath)
    fonasa_procesada = procesar_fonasa(input_filepath)

    # Define nombres de archivos output de INE y FONASA
    ruta_output_ine = f"{output_filepath}/df_ine.csv"
    ruta_output_fonasa = f"{output_filepath}/df_fonasa.csv"

    # Guarda base INE y FONASA
    ine_procesada.to_csv(ruta_output_ine)
    fonasa_procesada.to_csv(ruta_output_fonasa)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
