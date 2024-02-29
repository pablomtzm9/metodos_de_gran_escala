# train.py
'''
Este código es para entrenar el modelo
'''

import os
import argparse
import pandas as pd
import yaml
import random
import argparse
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from datetime import datetime

# config.yaml
# "../codigo/config.yaml"
with open("/Users/pablomartinez/Documents/Maestria/Primavera 2024/Métodos de Gran Escala/Tareas/Tarea 3 PMM/codigo/config.yaml", "r") as file:
    config = yaml.safe_load(file)

now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M%S")
name_aux = f"codigo/logs/{date_time}_train.log"

logging.basicConfig(
    filename= name_aux,
    level=logging.DEBUG,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def load_data(data_dir):
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
    random.seed(config['modeling']['random_seed'])
    model = RandomForestRegressor(n_estimators=config['modeling']['random_forest']['n_estimators'],
                                  max_depth=config['modeling']['random_forest']['max_depth'],
                                  min_samples_split=config['modeling']['random_forest']['min_samples_split'])
    model.fit(X, y)
    return model

def save_model(model, output_dir):
    model_path = os.path.join(output_dir, 'train_model.joblib')
    dump(model, model_path)
    print(f"Modelo guardado en {model_path}")

def main():
    parser = argparse.ArgumentParser(description="Train a model on input data.")
    parser.add_argument("data_dir", help="Directory containing input data")
    parser.add_argument("output_dir", help="Directory to save the trained model")
    args = parser.parse_args()

    try:
        # Load data
        data = load_data(args.data_dir)
        logging.info("Lectura de data")

        # Preprocess data
        X, y = preprocess_data(data)
        logging.info("Procesamiento de data")

        # Train model
        model = train_model(X, y)
        logging.info("Creación del modelo")

        # Save the trained model
        save_model(model, args.output_dir)
        logging.info("Guardar el modelo")

    except: 
        logging.error(f"ErrorElNombreDelPokemon: Lectura de data")
    

if __name__ == "__main__":
    main()
