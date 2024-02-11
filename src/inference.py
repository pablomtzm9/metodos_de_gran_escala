# inference.py
'''
Este código es para realizar la inferencia del modelo
'''
# Importar paquetes

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

# Ejecución de la inferencia

model = joblib.load("../models/train_model.joblib")
data = pd.read_csv("../data/base_general.csv")

def postprocess_data(data):
    # Your data preprocessing steps here
    test_df = data[data.Base == 0]
    test_df = test_df.drop(['Base','Id'], axis=1)

    X = test_df.drop('SalePrice', axis=1)
    y = test_df["SalePrice"]
    return X, y

def prediccion(modelo,data):
    # Trabajar con el modelo entrenado
    X, y = postprocess_data(data)
    y_pred = model.predict(X)
    return y_pred

def main():
    # Prediccion
    y_pred = prediccion(model,data)
    pd.DataFrame(y_pred).to_excel(excel_writer = r"../data/predicciones.xlsx")

    # Real
    X_test, y_test = postprocess_data(data)

    # Resultados
    accuracy = r2_score(list(y_test), list(y_pred))
    print("Accuracy:", accuracy)
    mae = mean_absolute_error(list(y_test), list(y_pred))
    print("MAE:", mae)
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values")
    plt.show()

if __name__ == "__main__":
    main()
