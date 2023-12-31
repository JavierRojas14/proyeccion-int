from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import numpy as np


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_elements = y_true != 0
    return (
        np.mean(
            np.abs((y_true[nonzero_elements] - y_pred[nonzero_elements]) / y_true[nonzero_elements])
        )
        * 100
    )


class ModeloHibrido:
    def __init__(self, modelo_1, modelo_2):
        self.modelo_1 = modelo_1
        self.modelo_2 = modelo_2
        self.param_grid_1 = None
        self.param_grid_2 = None
        self.best_params_1 = None
        self.best_params_2 = None
        self.n_splits = None
        self.cv_results_1 = None
        self.cv_results_2 = None
        self.test_predictions = None

    def fit_with_hyperparameter_tuning(self, X_1, X_2, y, param_grid_1, param_grid_2, n_splits):
        # Hyperparameter tuning for modelo_1
        print("Tuning modelo_1...")
        grid_search_1 = GridSearchCV(
            self.modelo_1,
            param_grid_1,
            cv=TimeSeriesSplit(n_splits=n_splits),
            scoring="neg_mean_absolute_error",
            return_train_score=True,
            refit=True,
        )
        grid_search_1.fit(X_1, y)
        self.modelo_1 = grid_search_1
        self.best_params_1 = grid_search_1.best_params_
        self.cv_results_1 = grid_search_1.cv_results_

        print("Best hyperparameters for modelo_1:", self.best_params_1)
        print("CV Results for modelo_1:")
        print("  MAE Mean Train Score:", np.mean(self.cv_results_1["mean_train_score"]))
        print("  MAE Mean Test Score:", np.mean(self.cv_results_1["mean_test_score"]))
        print("")

        # Predict residuals
        y_predict_1 = self.modelo_1.predict(X_1)
        y_resid = y - y_predict_1

        # Hyperparameter tuning for modelo_2
        print("Tuning modelo_2...")
        grid_search_2 = GridSearchCV(
            self.modelo_2,
            param_grid_2,
            cv=TimeSeriesSplit(n_splits=n_splits),
            scoring="neg_mean_absolute_error",
            return_train_score=True,
            refit=True,
        )
        grid_search_2.fit(X_2, y_resid)
        self.modelo_2 = grid_search_2
        self.best_params_2 = grid_search_2.best_params_
        self.cv_results_2 = grid_search_2.cv_results_

        print("Best hyperparameters for modelo_2:", self.best_params_2)
        print("CV Results for modelo_2:")
        print("  MAE Mean Train Score:", np.mean(self.cv_results_2["mean_train_score"]))
        print("  MAE Mean Test Score:", np.mean(self.cv_results_2["mean_test_score"]))
        print("")
        print("Training process completed.")

    def fit(self, X_1, X_2, y):
        self.modelo_1.fit(X_1, y)

        # Predict Residuals
        y_predict_1 = self.modelo_1.predict(X_1)
        y_resid = y - y_predict_1

        self.modelo_2.fit(X_2, y_resid)

    def predict(self, X_1, X_2):
        y_predict_1 = self.modelo_1.predict(X_1)
        y_predict_2 = self.modelo_2.predict(X_2)

        yhat = y_predict_1 + y_predict_2

        return yhat

    def check_test_score(self, X_1_test, X_2_test, y_test):
        test_predictions = self.predict(X_1_test, X_2_test)

        mae_test = mean_absolute_error(y_test, test_predictions)
        mape_test = mean_absolute_percentage_error(y_test, test_predictions)

        print("Performance on test Set:")
        print("  MAE:", mae_test)
        print("  MAPE:", mape_test)
        print("")

        self.test_predictions = test_predictions
