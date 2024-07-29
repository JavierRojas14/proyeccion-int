import pandas as pd
from tabulate import tabulate

from src.features import build_features

ANIO_INICIO = 2017
ANIO_TERMINO = 2035
COLUMNAS_POBLACION_INE = [f"{i}" for i in range(ANIO_INICIO, ANIO_TERMINO + 1)]


def cargar_casos_area_de_influencia(ruta):
    """
    Carga los casos del área de influencia y formatea los diagnósticos.

    Args:
    ruta (str): Ruta del archivo de Excel.

    Returns:
    DataFrame: Casos del área de influencia con diagnósticos formateados.
    """
    df = pd.read_excel(ruta, sheet_name="casos_area_de_influencia_INT")
    df["Diagnostico"] = df["Diagnostico"].str.split(" - ").str[0]
    print("Casos del área de influencia cargados y formateados:")
    print(tabulate(df.head(), headers="keys", tablefmt="pretty"))
    print()
    return df.set_index("Diagnostico")


def cargar_porcentajes_de_quirurgicos(ruta):
    """
    Carga los porcentajes de quirúrgicos y formatea los diagnósticos.

    Args:
    ruta (str): Ruta del archivo de Excel.

    Returns:
    Series: Porcentajes quirúrgicos con diagnósticos formateados.
    """
    df = pd.read_excel(ruta, sheet_name="porcentajes_trazadoras")
    df["Diagnostico"] = df["Diagnostico"].str.split(" - ").str[0]
    porcentajes = df.set_index("Diagnostico")["Porcentaje Quirúrgico"].dropna()
    print("Porcentajes quirúrgicos cargados y formateados:")
    print(tabulate(porcentajes.head().reset_index(), headers="keys", tablefmt="pretty"))
    print()
    return porcentajes


def calcular_casos_quirurgicos(casos_area_de_influencia, porcentajes_de_quirurgicos):
    """
    Calcula los casos quirúrgicos multiplicando por los porcentajes quirúrgicos.

    Args:
    casos_area_de_influencia (DataFrame): Casos del área de influencia.
    porcentajes_de_quirurgicos (Series): Porcentajes quirúrgicos.

    Returns:
    DataFrame: Casos quirúrgicos calculados.
    """
    casos_quirurgicos = casos_area_de_influencia[COLUMNAS_POBLACION_INE].mul(
        porcentajes_de_quirurgicos, axis=0
    )
    print("Casos quirúrgicos calculados:")
    print(tabulate(casos_quirurgicos.head(), headers="keys", tablefmt="pretty"))
    print()
    return casos_quirurgicos


def cargar_duraciones_int_q(ruta, diags_area_de_influencia):
    """
    Carga y filtra las duraciones de intervenciones quirúrgicas por diagnóstico.

    Args:
    ruta (str): Ruta del archivo de Excel.
    diags_area_de_influencia (Index): Diagnósticos del área de influencia.

    Returns:
    Series: Duraciones promedio de intervenciones quirúrgicas por diagnóstico.
    """
    df = pd.read_excel(ruta)
    df["diag_01_principal_cod"] = (
        df["diag_01_principal_cod"].str.replace(".", "", regex=False).str.ljust(4, "X")
    )

    df = df.query("diag_01_principal_cod.isin(@diags_area_de_influencia)").set_index(
        "diag_01_principal_cod"
    )

    # Convierte las duraciones a tiempo
    columnas_duracion = ["mean", "std", "min", "25%", "50%", "75%", "max"]
    df[columnas_duracion] = df[columnas_duracion].apply(lambda x: pd.to_timedelta(x, unit="D"))
    duraciones = df.query("ano_de_egreso == 2019")["mean"]
    print("Duraciones de intervenciones quirúrgicas cargadas y filtradas:")
    print(tabulate(duraciones.head().reset_index(), headers="keys", tablefmt="pretty"))
    print()
    return duraciones


def calcular_tiempo_utilizado_pabellon(casos_quirurgicos, duraciones_int_q):
    """
    Calcula el tiempo utilizado en pabellón multiplicando casos quirúrgicos por duraciones.

    Args:
    casos_quirurgicos (DataFrame): Casos quirúrgicos calculados.
    duraciones_int_q (Series): Duraciones promedio de intervenciones quirúrgicas.

    Returns:
    DataFrame: Tiempo utilizado en pabellón en horas.
    """
    tiempo_utilizado_pabellon = casos_quirurgicos.mul(duraciones_int_q, axis=0)
    tiempo_utilizado_pabellon_horas = tiempo_utilizado_pabellon.apply(
        lambda x: x.dt.total_seconds() / 3600
    )
    print("Tiempo utilizado en pabellón calculado (en horas):")
    print(tabulate(tiempo_utilizado_pabellon_horas.head(), headers="keys", tablefmt="pretty"))
    print()
    return tiempo_utilizado_pabellon_horas


def calcular_horas_laborales(anio_inicio, anio_termino, horas_por_dia):
    """
    Calcula la cantidad de horas laborales por año.

    Args:
    anio_inicio (int): Año de inicio.
    anio_termino (int): Año de término.
    horas_por_dia (int): Horas de trabajo por día.

    Returns:
    Series: Horas laborales por año.
    """
    cantidad_dias_laborales = build_features.obtener_cantidad_de_dias_laborales_por_anio(
        f"01-01-{anio_inicio}", f"01-01-{anio_termino + 1}"
    )
    cantidad_dias_laborales.index = cantidad_dias_laborales.index.year.astype(str)
    horas_laborales = cantidad_dias_laborales * horas_por_dia
    print("Horas laborales por año calculadas:")
    print(tabulate(horas_laborales.head().reset_index(), headers="keys", tablefmt="pretty"))
    print()
    return horas_laborales


def calcular_cantidad_de_pabellones_necesarios(tiempo_utilizado_pabellon_horas, horas_laborales):
    """
    Calcula la cantidad de pabellones necesarios.

    Args:
    tiempo_utilizado_pabellon_horas (DataFrame): Tiempo utilizado en pabellón en horas.
    horas_laborales (Series): Horas laborales por año.

    Returns:
    DataFrame: Cantidad de pabellones necesarios por año.
    """
    cantidad_de_pabellones_necesarios = tiempo_utilizado_pabellon_horas.div(horas_laborales, axis=1)
    print("Cantidad de pabellones necesarios calculada:")
    print(tabulate(cantidad_de_pabellones_necesarios.head(), headers="keys", tablefmt="pretty"))
    print()
    return cantidad_de_pabellones_necesarios
