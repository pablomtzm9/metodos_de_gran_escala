# prep.py
'''
Este código es para leer y trabajar las bases de datos
'''

# Importar paquetes

#!pip install grpcio
#!pip install tensorflow
#!pip install tensorflow-cpu
#!pip install tensorflow-intel
#!pip install tensorflowjs
#!pip install tensorflow_decision_forests
#!pip install seaborn
#!pip install matplotlib
#!pip install ultralytics
#!pip install graphviz

import pandas as pd
import argparse
import numpy as np
#import graphviz
#import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import math
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.tree import export_graphviz
from IPython.display import Image
from scripts import clean_dataset
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('print_anal_explotario')
args = parser.parse_args()
now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M%S")
name_aux = f"codigo/logs/{date_time}_prep.log"

logging.basicConfig(
    filename= name_aux,
    level=logging.DEBUG,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # Read inputs
    logging.info("Training starting ...")

# Descargar datos

try:
    # url = 'https://raw.githubusercontent.com/pablomtzm9/metodos_de_gran_escala 51442d74abc66a1d7db47d7d6b4c40cad76a82dd/data/'
    url = '/Users/pablomartinez/Documents/Maestria/Primavera 2024/Métodos de Gran Escala/Tareas/Tarea 3 PMM/data/'

    dataset = pd.read_csv(url+'train.csv')
    dataset_test_id = pd.read_csv(url+'test.csv')
    y_pred = pd.read_csv(url+'sample_submission.csv')
except:
    logging.error(f"ErrorElNombreDelPokemon: Lectura de data")

# Preprocesar datos

dataset_test = dataset_test_id.drop('Id', axis=1)
print(dataset['SalePrice'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(dataset['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.4});
df_num = dataset.select_dtypes(include = ['float64', 'int64'])
if args.print_anal_explotario == "plot":
    df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);

# Hacer joins de las tablas

base_test = dataset_test_id.merge(y_pred, left_on='Id', right_on='Id')
base_test['Base'] = 0 #test
dataset['Base'] = 1 #train
base_general = pd.concat([base_test, dataset], ignore_index=True)
base_general = base_general[['Id','MSSubClass','LotFrontage','LotArea','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF','GrLivArea','GarageYrBlt','MoSold','YrSold','SalePrice','Base']]
base_general = pd.get_dummies(base_general)
base_general = clean_dataset(base_general)

try: 
    base_general.to_csv(r"../data/base_general.csv", index=False)
    logging.info("Generación de base_general.csv")
    row, col = base_general.shape
    logging.debug(f"row, col = {row, col}")
except:
    logging.error(f"ErrorElNombreDelPokemon: Impresion de base")