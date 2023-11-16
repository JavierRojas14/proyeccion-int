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
    def __init__(self, modelo_1, modelo_2, param_grid_1, param_grid_2, n_splits=5):
        self.modelo_1 = modelo_1
        self.modelo_2 = modelo_2
        self.param_grid_1 = param_grid_1
        self.param_grid_2 = param_grid_2
        self.best_params_1 = None
        self.best_params_2 = None
        self.n_splits = n_splits
        self.cv_results_1 = None
        self.cv_results_2 = None
        self.validation_predictions = None

    def hyperparameter_tuning(self, X_1, X_2, y):
        # Hyperparameter tuning for modelo_1
        print("Tuning modelo_1...")
        grid_search_1 = GridSearchCV(
            self.modelo_1,
            self.param_grid_1,
            cv=TimeSeriesSplit(n_splits=self.n_splits),
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
            self.param_grid_2,
            cv=TimeSeriesSplit(n_splits=self.n_splits),
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

    def fit(
        self,
        X_1_cv,
        X_2_cv,
        y_cv,
        X_1_valid=None,
        X_2_valid=None,
        y_valid=None,
        hyperparameter_tuning=True,
    ):
        if hyperparameter_tuning:
            # Perform hyperparameter tuning
            self.hyperparameter_tuning(X_1_cv, X_2_cv, y_cv)

        print("Training process completed.")

        if X_1_valid is not None and X_2_valid is not None and y_valid is not None:
            validation_predictions = self.predict(X_1_valid, X_2_valid)

            mae_validation = mean_absolute_error(y_valid, validation_predictions)
            mape_validation = mean_absolute_percentage_error(y_valid, validation_predictions)

            print("Performance on Validation Set:")
            print("  MAE:", mae_validation)
            print("  MAPE:", mape_validation)
            print("")

            self.validation_predictions = validation_predictions

    def predict(self, X_1, X_2):
        y_predict_1 = self.modelo_1.predict(X_1)
        y_predict_2 = self.modelo_2.predict(X_2)

        yhat = y_predict_1 + y_predict_2

        return yhat
