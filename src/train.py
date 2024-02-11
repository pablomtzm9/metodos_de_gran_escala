# train.py
'''
Este código es para entrenar el modelo
'''

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

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
    model = RandomForestRegressor()
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

    # Load data
    data = load_data(args.data_dir)

    # Preprocess data
    X, y = preprocess_data(data)

    # Train model
    model = train_model(X, y)

    # Save the trained model
    save_model(model, args.output_dir)

if __name__ == "__main__":
    main()
