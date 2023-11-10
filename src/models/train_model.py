class ModeloHibrido:
    def __init__(self, modelo_1, modelo_2):
        self.modelo_1 = modelo_1
        self.modelo_2 = modelo_2

    def fit(self, X_1, X_2, y):
        self.modelo_1.fit(X_1, y)

        y_predict_1 = self.modelo_1.predict(X_1)
        y_resid = y - y_predict_1

        self.modelo_2.fit(X_2, y_resid)

    def predict(self, X_1, X_2):
        y_predict_1 = self.modelo_1.predict(X_1)
        y_predict_2 = self.modelo_2.predict(X_2)

        yhat = y_predict_1 + y_predict_2

        return yhat
