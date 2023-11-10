from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


class ModeloHibrido:
    def __init__(self, modelo_1, modelo_2, n_splits=5):
        self.modelo_1 = modelo_1
        self.modelo_2 = modelo_2
        self.best_params_1 = None
        self.best_params_2 = None
        self.n_splits = n_splits

    def hyperparameter_tuning(self, param_grid_1, param_grid_2, X_1, X_2, y):
        # Hyperparameter tuning for modelo_1
        grid_search_1 = GridSearchCV(
            self.modelo_1,
            param_grid_1,
            cv=TimeSeriesSplit(n_splits=self.n_splits, test_size=365),
            scoring="neg_mean_squared_error",
        )
        grid_search_1.fit(X_1, y)
        self.modelo_1 = grid_search_1.best_estimator_
        self.best_params_1 = grid_search_1.best_params_

        # Predict residuals
        y_predict_1 = self.modelo_1.predict(X_1)
        y_resid = y - y_predict_1

        # Hyperparameter tuning for modelo_2
        grid_search_2 = GridSearchCV(
            self.modelo_2,
            param_grid_2,
            cv=TimeSeriesSplit(n_splits=self.n_splits, test_size=365),
            scoring="neg_mean_squared_error",
        )
        grid_search_2.fit(X_2, y_resid)
        self.modelo_2 = grid_search_2.best_estimator_
        self.best_params_2 = grid_search_2.best_params_

    def fit(self, X_1, X_2, y):
        # If hyperparameter tuning is not done, you can do it here or beforehand
        # ...

        self.modelo_1.fit(X_1, y)

        y_predict_1 = self.modelo_1.predict(X_1)
        y_resid = y - y_predict_1

        self.modelo_2.fit(X_2, y_resid)

    def predict(self, X_1, X_2):
        y_predict_1 = self.modelo_1.predict(X_1)
        y_predict_2 = self.modelo_2.predict(X_2)

        yhat = y_predict_1 + y_predict_2

        return yhat
