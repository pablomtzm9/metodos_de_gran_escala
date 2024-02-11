# scripts.py
'''
Este código es para Refactorizar lo realizado en el notebook.
Se presentan las funciones realizadas a lo largo del ejercicio
'''

# Importar paquetes

!pip install grpcio
!pip install tensorflow
!pip install tensorflow-cpu
!pip install tensorflow-intel
!pip install tensorflowjs
!pip install tensorflow_decision_forests
!pip install seaborn
!pip install matplotlib
!pip install ultralytics
!pip install graphviz

import pandas as pd
import numpy as np
import graphviz
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os
import argparse
import joblib

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, 
recall_score, ConfusionMatrixDisplay, r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.tree import export_graphviz
from IPython.display import Image

# Funciones

def clean_dataset(df):
    '''Limpieza de un dataframe de: na, nan, inf

    Parámetros
    ----------
    df : dataframe
        archivo que se quiere limpiar

    Returns
    -------
    dataframe
        archivo sin los renglones que tuvieron na, nan o inf
    '''
  
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return df[indices_to_keep].astype(np.float64)

def load_data(data_dir):
    # Funcion para leer el archivo
    data_path = os.path.join(data_dir, 'base_general.csv')
    return pd.read_csv(data_path)

def preprocess_data(data):
    # Your data preprocessing steps here
    train_df = data[data.Base == 1]
    train_df = train_df.drop(['Base','Id'], axis=1)

    X = train_df.drop('SalePrice', axis=1)
    y = train_df["SalePrice"]
    return X, y

def train_model(X, y):
    # Funcion para ejecutar el modelo
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

def save_model(model, output_dir):
    # Funcion para guardar el modelo creado
    model_path = os.path.join(output_dir, 'train_model.joblib')
    dump(model, model_path)
    print(f"Modelo guardado en {model_path}")

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

def prep():
    '''Manejo de la base de datos. Limpieza
    '''
    
    url = 'https://raw.githubusercontent.com/pablomtzm9/metodos_de_gran_escala 51442d74abc66a1d7db47d7d6b4c40cad76a82dd/data/'

    dataset = pd.read_csv(url+'train.csv')
    dataset_test_id = pd.read_csv(url+'test.csv')
    y_pred = pd.read_csv(url+'sample_submission.csv')

    # Preprocesar datos

    dataset_test = dataset_test_id.drop('Id', axis=1)
    print(dataset['SalePrice'].describe())
    plt.figure(figsize=(9, 8))
    sns.distplot(dataset['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.4});
    df_num = dataset.select_dtypes(include = ['float64', 'int64'])
    df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);

    # Hacer joins de las tablas

    base_test = dataset_test_id.merge(y_pred, left_on='Id', right_on='Id')
    base_test['Base'] = 0 #test
    dataset['Base'] = 1 #train
    base_general = pd.concat([base_test, dataset], ignore_index=True)
    base_general = base_general[['Id','MSSubClass','LotFrontage','LotArea','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF','GrLivArea','GarageYrBlt','MoSold','YrSold','SalePrice','Base']]
    base_general = pd.get_dummies(base_general)
    base_general = clean_dataset(base_general)

    base_general.to_csv(r"../data/base_general.csv", index=False)

def train():
    '''Esta función es para entrenar el modelo
    '''
    # Read data
    data = load_data("../data/")

    # Preprocess data
    X, y = preprocess_data(data)

    # Train model
    model = train_model(X, y)

    # Save the trained model
    save_model(model, "../models/")

def inference():
    '''Esta función es para realizar la inferencia
    '''
    # Ejecución de la inferencia

    model = joblib.load("../models/train_model.joblib")
    data = pd.read_csv("../data/base_general.csv")

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
