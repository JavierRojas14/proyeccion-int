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
    df = df.set_index("Diagnostico").dropna(subset="Porcentaje Quirúrgico")

    porcentajes = df["Porcentaje Quirúrgico"].replace(["Separado", "Preguntar"], "0").astype(float)
    especialidades = df["Especialidad Quirurgica"]

    print("Porcentajes quirúrgicos cargados y formateados:")
    print(tabulate(porcentajes.head().reset_index(), headers="keys", tablefmt="pretty"))
    print()
    return porcentajes, especialidades


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


def calcular_casos_quirurgicos_por_especialidad(casos_quirurgicos, especialidades):
    tmp = casos_quirurgicos.copy()
    tmp["especialidades"] = especialidades
    tmp = tmp.groupby("especialidades").sum()

    return tmp


def reasignar_diagnosticos(df, columna_diag, df_diagnosticos_a_reasignar):
    """
    Reasigna los códigos de diagnósticos en la base de datos.

    Args:
        df (DataFrame): Base de datos que contiene los diagnósticos a reasignar.
        columna_diag: Nombre de la columna que indica los diagnosticos en la base original
        diagnosticos_a_reasignar (DataFrame): DataFrame que contiene los códigos nuevos y
        antiguos de los diagnósticos.

    Returns:
        DataFrame: Base de datos con los diagnósticos actualizados.
    """
    df = df.copy()
    for row in df_diagnosticos_a_reasignar.itertuples(index=False):
        diagnostico_nuevo, diagnosticos_antiguos = row
        print(f"Cambiando {diagnosticos_antiguos} a {diagnostico_nuevo}")
        df[columna_diag] = df[columna_diag].replace(diagnosticos_antiguos, diagnostico_nuevo)

    return df


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


def obtener_resumen_ocurrencia_complicacion(df, df_filtrada):
    # Obtiene el resumen de la cantidad de ocurrencias del DataFrame total y el filtrado
    resumen = pd.DataFrame(
        {
            "totales": df.groupby(["ano_de_intervencion", "especialidad"]).size(),
            "ocurrencia_filtrado": df_filtrada.groupby(
                ["ano_de_intervencion", "especialidad"]
            ).size(),
        }
    )

    # Obtiene el resumen acumulado en el periodo
    resumen_acumulado = resumen.sum()

    # Obtiene los porcentajes de ocurrencia desglosados y acumulados
    resumen["fraccion"] = resumen["ocurrencia_filtrado"] / resumen["totales"]
    porcentaje_acumulado = resumen_acumulado["ocurrencia_filtrado"] / resumen_acumulado["totales"]

    # Obtiene el resumen por especialidad acumulado
    resumen_acumulado_por_especialidad = (
        resumen.reset_index().groupby("especialidad")[["totales", "ocurrencia_filtrado"]].sum()
    )
    resumen_acumulado_por_especialidad["fraccion"] = (
        resumen_acumulado_por_especialidad["ocurrencia_filtrado"]
        / resumen_acumulado_por_especialidad["totales"]
    )

    return resumen, resumen_acumulado, resumen_acumulado_por_especialidad, porcentaje_acumulado


def buscar_nombre_operacion_pabellon(df, operaciones):
    # Filtra la base de datos segun el nombre de la operacion
    return df[df["nombre_de_la_operacion"].fillna("").str.contains(operaciones, regex=True)]


def buscar_nombre_diagnosticos_pabellon(df, diagnosticos):
    # Filtra la base de datos segun el nombre del diagnostico 1 y 2
    return df[
        (df["primer_diagnostico"].fillna("").str.contains(diagnosticos, regex=True))
        | (df["segundo_diagnostico"].fillna("").str.contains(diagnosticos, regex=True))
    ]


def iterar_en_complicaciones_a_buscar(df, dict_textos_a_buscar, tipo_complicacion):
    # Decide que parametro a buscar en la base de datos
    busqueda_a_realizar = {
        "intervencion_quirurgica": buscar_nombre_operacion_pabellon,
        "diagnostico": buscar_nombre_diagnosticos_pabellon,
    }
    funcion_a_ocupar_para_buscar = busqueda_a_realizar[tipo_complicacion]

    # Itera por el diccionario de busqueda y guarda los resultados
    df_resultado = pd.DataFrame()
    for nombre_complicacion, textos_a_buscar in dict_textos_a_buscar.items():
        df_filtrada = funcion_a_ocupar_para_buscar(df, textos_a_buscar)
        resumen_filtrado = obtener_resumen_ocurrencia_complicacion(df, df_filtrada)
        tiempo_pabellon_75 = df_filtrada["duracion"].describe()["75%"]

        # Concatena resultados acumulados en el periodo por complicacion
        resultado_acumulado = resumen_filtrado[2]
        resultado_acumulado["complicacion"] = nombre_complicacion
        resultado_acumulado["texto_a_buscar"] = textos_a_buscar
        resultado_acumulado["tiempo_operacion_75%"] = tiempo_pabellon_75
        df_resultado = pd.concat([df_resultado, resultado_acumulado])

    return df_resultado
