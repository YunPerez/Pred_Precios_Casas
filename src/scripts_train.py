'''
Este script contiene las funciones necesarias para entrenar
un modelo de machine learning con los datos de entrada en formato .csv
que se encuentran en la carpeta data/prep y guardar el modelo
entrenado en la carpeta models con formato .joblib
'''
import yaml
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def load_config(config_path='config.yaml'):
    '''
    Esta función carga el archivo de configuración
    en formato .yaml que se encuentra en la carpeta raíz
    del proyecto
    '''
    with open(config_path, 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
    return config


def train_model(data_input, config, output_dir):
    '''
    Esta función entrena el modelo de machine learning
    con los datos de entrada en formato .csv que se
    encuentran en la carpeta data/prep y guarda el modelo
    entrenado en la carpeta models con formato .joblib
    '''
    # Cargamos la base con las variables para el modelo de predicción
    data = pd.read_csv(data_input)
    x_data = data.drop('SalePrice', axis=1)
    y_data = data['SalePrice']
    x_train, _, y_train, _ = train_test_split(
        x_data, y_data, test_size=0.2, random_state=123)
    # Obtiene los parámetros del modelo
    # desde el archivo de configuración
    rf_params = config['random_forest']
    # Se entrena con el modelo Random Forest Regressor
    rfr_model = RandomForestRegressor(**rf_params)
    rfr_model.fit(x_train, y_train)
    # Guardamos el modelo entrenado en la carpeta models
    joblib.dump(rfr_model, output_dir + "rfr_model.joblib")
    print(f"El modelo fue entrenado y guardado en {output_dir}")
