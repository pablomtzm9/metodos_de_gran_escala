# inference.py
'''
Este código es para realizar la inferencia del modelo
'''
# Importar paquetes

import os
import joblib
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from datetime import datetime

# Ejecución de la inferencia

try:
    model = joblib.load("/Users/pablomartinez/Documents/Maestria/Primavera 2024/Métodos de Gran Escala/Tareas/Tarea 3 PMM/models/train_model.joblib")
    data = pd.read_csv("/Users/pablomartinez/Documents/Maestria/Primavera 2024/Métodos de Gran Escala/Tareas/Tarea 3 PMM/data/base_general.csv")
except:
    logging.error(f"ErrorElNombreDelPokemon: Lectura de data")

parser = argparse.ArgumentParser()
parser.add_argument('print_resultados')
args = parser.parse_args()
now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M%S")
name_aux = f"codigo/logs/{date_time}_inference.log"

logging.basicConfig(
    filename= name_aux,
    level=logging.DEBUG,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

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
    row, col = y_pred.shape
    try:
        pd.DataFrame(y_pred).to_excel(excel_writer = r"/Users/pablomartinez/Documents/Maestria/Primavera 2024/Métodos de Gran Escala/Tareas/Tarea 3 PMM/data/predicciones.xlsx")
        logging.info("Generación de predicciones.xlsx")
        logging.debug(f"row, col = {row, col}")
    except:
        logging.error(f"ErrorElNombreDelPokemon: Impresion de base")

    # Real
    X_test, y_test = postprocess_data(data)

    # Resultados
    accuracy = r2_score(list(y_test), list(y_pred))
    print("Accuracy:", accuracy)
    mae = mean_absolute_error(list(y_test), list(y_pred))
    print("MAE:", mae)
    if args.print_resultados == "plot":
        print("Resultados Gráficos:")
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted Values")
        plt.show()

