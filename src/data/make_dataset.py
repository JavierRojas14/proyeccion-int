# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


import pandas as pd
import numpy as np


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

    # Indica si es adulto o infantil
    df["grupo_etario_poblacion"] = np.where(df["Edad"] >= 15, "Adulto", "Infantil")

    return df


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    ine_procesada = procesar_ine(input_filepath)
    ruta_output_ine = f"{output_filepath}/df_ine.csv"
    ine_procesada.to_csv(ruta_output_ine)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
