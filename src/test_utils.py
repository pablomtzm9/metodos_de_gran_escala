# test_utils.py

import pandas as pd
import joblib
from train import load_data
from train import train_model
from train import save_model
from train import preprocess_data
from inference import postprocess_data
from inference import prediccion

def test_load_data_inexistente():
    assert(load_data("../data_prueba/"))

def test_load_data_otra_carpeta():
    assert(load_data("../models/"))

def test_postprocess_data_inexistente():
    assert(postprocess_data(pd.read_csv("../data/base_general_prueba.csv")))

def test_prediccion_inexistente():
    assert(prediccion(joblib.load("../models/train_model_prueba.joblib"),pd.read_csv("../data/base_general_prueba.csv")))

def test_train_model_dif_dim():
    X, y = preprocess_data(load_data("../data/"))
    assert(train_model(X,X))

def test_save_model_directorio_erroneo():
    assert(save_model(joblib.load("../models/train_model.joblib"),"../carpeta_falsa/"))
