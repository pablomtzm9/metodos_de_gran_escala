# scripts.py
'''
Este código es para Refactorizar lo realizado en el notebook.
Se presentan las funciones realizadas a lo largo del ejercicio
'''

# Importar paquetes

# !pip install pyyaml

import os
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
import random

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# config.yaml
with open("/Users/pablomartinez/Documents/Maestria/Primavera 2024/Métodos de Gran Escala/Tareas/Tarea 3 PMM/codigo/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Funciones

def clean_dataset(data):
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

    assert isinstance(data, pd.DataFrame), "df needs to be a pd.DataFrame"
    data.dropna(inplace=True)
    indices_to_keep = ~data.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return data[indices_to_keep].astype(np.float64)

def load_data(data_dir):
    '''Funcion para leer el archivo
    '''

    data_path = os.path.join(data_dir, 'base_general.csv')
    return pd.read_csv(data_path)

def preprocess_data(data):
    '''Your data preprocessing steps here
    '''

    train_df = data[data.Base == 1]
    train_df = train_df.drop(['Base','Id'], axis=1)

    base_train = train_df.drop('SalePrice', axis=1)
    vector_train = train_df["SalePrice"]
    return base_train, vector_train

def train_model(base_train, vector_train):
    '''Funcion para ejecutar el modelo
    '''
    random.seed(config['modeling']['random_seed'])
    model = RandomForestRegressor(n_estimators=config['modeling']['random_forest']['n_estimators'],
                                  max_depth=config['modeling']['random_forest']['max_depth'],
                                  min_samples_split=config['modeling']['random_forest']['min_samples_split'])
    model.fit(base_train, vector_train)
    return model

def save_model(model, output_dir):
    '''Funcion para guardar el modelo creado
    '''
    model_path = os.path.join(output_dir, 'train_model.joblib')
    joblib.dump(model, model_path)
    print(f"Modelo guardado en {model_path}")

def postprocess_data(data):
    '''Your data preprocessing steps here
    '''
    test_df = data[data.Base == 0]
    test_df = test_df.drop(['Base','Id'], axis=1)

    base_test = test_df.drop('SalePrice', axis=1)
    vector_test = test_df["SalePrice"]
    return base_test, vector_test

def prediccion(modelo,data):
    '''Trabajar con el modelo entrenado
    '''
    base_test = postprocess_data(data)[0]
    vector_pred = modelo.predict(base_test)
    return vector_pred

def prep():
    '''Manejo de la base de datos. Limpieza
    '''
    url = "https://raw.githubusercontent.com/pablomtzm9/metodos_de_gran_escala/main/data/"

    dataset = pd.read_csv(url+'train.csv')
    dataset_test_id = pd.read_csv(url+'test.csv')
    vector_pred = pd.read_csv(url+'sample_submission.csv')

    # Preprocesar datos

    print(dataset['SalePrice'].describe())
    plt.figure(figsize=(9, 8))
    sns.distplot(dataset['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.4})
    df_num = dataset.select_dtypes(include = ['float64', 'int64'])
    df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)

    # Hacer joins de las tablas

    base_test = dataset_test_id.merge(vector_pred, left_on='Id', right_on='Id')
    base_test['Base'] = 0 #test
    dataset['Base'] = 1 #train
    base_general = pd.concat([base_test, dataset], ignore_index=True)
    base_general = base_general[['Id','MSSubClass','LotFrontage','LotArea', \
    'YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF','GrLivArea', \
    'GarageYrBlt','MoSold','YrSold','SalePrice','Base']]
    base_general = pd.get_dummies(base_general)
    base_general = clean_dataset(base_general)

    base_general.to_csv(r"../data/base_general.csv", index=False)

def train():
    '''Esta función es para entrenar el modelo
    '''
    # Read data
    data = load_data("../data/")

    # Preprocess data
    base_train, vector_train = preprocess_data(data)

    # Train model
    model = train_model(base_train, vector_train)

    # Save the trained model
    save_model(model, "../models/")

def inference():
    '''Esta función es para realizar la inferencia
    '''
    # Ejecución de la inferencia

    model = joblib.load("../models/train_model.joblib")
    data = pd.read_csv("../data/base_general.csv")

    # Prediccion
    vector_pred = prediccion(model,data)
    pd.DataFrame(vector_pred).to_excel(excel_writer = r"../data/predicciones.xlsx")

    # Real
    vector_test = postprocess_data(data)[1]

    # Resultados
    accuracy = r2_score(list(vector_test), list(vector_pred))
    print("Accuracy:", accuracy)
    mae = mean_absolute_error(list(vector_test), list(vector_pred))
    print("MAE:", mae)
    plt.scatter(vector_test, vector_pred)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values")
    plt.show()
