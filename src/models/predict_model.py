import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


def evaluate_metrics(metrics, y_true, y_pred):
    """
    Evaluate machine learning model performance using specified metrics.

    Parameters:
    - metrics (dict): A dictionary of metric functions where keys are metric names and values are
    functions.
    - y_true (numpy.ndarray): True target values.
    - y_pred (numpy.ndarray): Model predictions.

    Returns:
    - results (dict): A dictionary containing metric names and their corresponding values.
    """
    results = {}

    for metric_name, metric_func in metrics.items():
        # Calculate the metric value
        metric_value = round(metric_func(y_true, y_pred), 2)
        results[metric_name] = metric_value
        print(f"{metric_name}: {metric_value}")

    print()

    return results


def create_prediction_dataframe(ds, y_true, yhat):
    """
    Create a DataFrame containing true values, predicted values, and a DateTimeIndex.

    Parameters:
    - ds (pd.Series): DateTimeIndex for the DataFrame.
    - y_true (np.ndarray or pd.Series): True values.
    - yhat (np.ndarray or pd.Series): Predicted values.

    Returns:
    - df (pd.DataFrame): DataFrame containing "ds," "y_true," and "yhat" columns.
    """
    df = pd.DataFrame({"ds": ds, "y_true": y_true, "yhat": yhat})

    df.set_index("ds", inplace=True)

    return df


def evaluar_desempeno_train_y_test_serie_tiempo(
    datetime_index_train, y_train, yhat_train, datetime_index_test, y_test, yhat_test, metrics
):
    # Evalua rendimiento en conjunto de entrenamiento y testeo
    print("Train")
    resultados_train = evaluate_metrics(metrics, y_train, yhat_train)
    resultados_train = {f"train_{key}": value for key, value in resultados_train.items()}
    print("Test")
    resultados_test = evaluate_metrics(metrics, y_test, yhat_test)
    resultados_test = {f"test_{key}": value for key, value in resultados_test.items()}

    # Crea DataFrame con el valor real y predicho
    df_train_yhat = create_prediction_dataframe(datetime_index_train, y_train, yhat_train)
    df_test_yhat = create_prediction_dataframe(datetime_index_test, y_test, yhat_test)

    # Grafica valores reales y predichos
    fig, axis = plt.subplots(2, 1, figsize=(20, 12))

    df_train_yhat.plot(ax=axis[0])
    df_test_yhat.plot(ax=axis[1])

    primera_metrica = next(iter(metrics))
    metrica_train = f"train_{primera_metrica}"
    metrica_test = f"test_{primera_metrica}"

    # Reporta el valor de la primera metrica ingresada
    axis[0].title.set_text(f"Train - {primera_metrica} {resultados_train[metrica_train]}")
    axis[1].title.set_text(f"Test - {primera_metrica} {resultados_test[metrica_test]}")

    return resultados_train, resultados_test, df_train_yhat, df_test_yhat, fig


def create_prediction_dataframe_multivariate(ds, y_true, yhat, products):
    df = pd.DataFrame(
        {
            "DIAG1": products,
            "y_true": y_true,
            "yhat": yhat,
        },
        index=ds,
    )

    return df


def evaluar_desempeno_train_y_test_serie_tiempo_multivariada(
    datetime_index_train,
    y_train,
    yhat_train,
    products_train,
    datetime_index_test,
    y_test,
    yhat_test,
    products_test,
    metrics,
):
    # Crea DataFrame con el valor real y predicho
    df_train_yhat = create_prediction_dataframe_multivariate(
        datetime_index_train, y_train, yhat_train, products_train
    )

    df_test_yhat = create_prediction_dataframe_multivariate(
        datetime_index_test, y_test, yhat_test, products_test
    )

    # Hace un resample mensual, sumando los egresos de todos los diags
    df_train_yhat_resampled = df_train_yhat.resample("M").sum()
    df_test_yhat_resampled = df_test_yhat.resample("M").sum()

    # Evalua rendimiento en conjunto de entrenamiento y testeo
    print("Train")
    resultados_train = evaluate_metrics(
        metrics, df_train_yhat_resampled["y_true"], df_train_yhat_resampled["yhat"]
    )
    resultados_train = {f"train_{key}": value for key, value in resultados_train.items()}
    print("Test")
    resultados_test = evaluate_metrics(
        metrics, df_test_yhat_resampled["y_true"], df_test_yhat_resampled["yhat"]
    )
    resultados_test = {f"test_{key}": value for key, value in resultados_test.items()}

    # Grafica valores reales y predichos
    fig, axis = plt.subplots(2, 1, figsize=(20, 12))

    df_train_yhat_resampled.plot(ax=axis[0])
    df_test_yhat_resampled.plot(ax=axis[1])

    primera_metrica = next(iter(metrics))
    metrica_train = f"train_{primera_metrica}"
    metrica_test = f"test_{primera_metrica}"

    # Reporta el valor de la primera metrica ingresada
    axis[0].title.set_text(f"Train - {primera_metrica} {resultados_train[metrica_train]}")
    axis[1].title.set_text(f"Test - {primera_metrica} {resultados_test[metrica_test]}")

    plt.tight_layout()

    return resultados_train, resultados_test, df_train_yhat, df_test_yhat, fig
