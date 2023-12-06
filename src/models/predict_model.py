def evaluate_metrics(metrics, predictions, targets):
    """
    Evaluate machine learning model performance using specified metrics.

    Parameters:
    - metrics (dict): A dictionary of metric functions where keys are metric names and values are
    functions.
    - predictions (numpy.ndarray): Model predictions.
    - targets (numpy.ndarray): True target values.

    Returns:
    - results (dict): A dictionary containing metric names and their corresponding values.
    """
    results = {}

    for metric_name, metric_func in metrics.items():
        # Calculate the metric value
        metric_value = round(metric_func(predictions, targets), 2)
        results[metric_name] = metric_value
        print(f"{metric_name}: {metric_value}")

    print()

    return results
