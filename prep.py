'''
En este módulo se hará el preprocesamiento de los datos
para el modelo de predicción de precios de casas

- Este script leerá los achivos de train y test de la carpeta "raw".
- Y guardara en la carpeta de "prep" la base de datos que se ocupará
  para el modelo.
'''

import os
import logging
from datetime import datetime
import argparse
import pandas as pd
from src.scripts_prep import load_data, preprocess_data

if not os.path.exists("logs/"):
    os.makedirs("logs/")
# Setup Logging
now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M%S")
log_prep_file_name = f"logs/{date_time}_prep.log"
logging.basicConfig(
    filename=log_prep_file_name,
    level=logging.DEBUG,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # Read inputs
    logging.info("Empezando el preproceso ...")
    # Se configura el parser de argumentos
    parser = argparse.ArgumentParser(
        description='Preprocessing script for house price prediction model')
    parser.add_argument('--input_train', type=str,
                        default='./data/raw/train.csv',
                        help='Path to the input train data')
    parser.add_argument('--input_test', type=str,
                        default='./data/raw/test.csv',
                        help='Path to the input test data')
    parser.add_argument('--output_prep', type=str,
                        default='./data/prep',
                        help='Path to the output preprocessed data')

    # Se obtienen los argumentos proporcionados por el usuario
    args = parser.parse_args()

    # Se asegura de que la carpeta de salida exista, si no, se crea
    if not os.path.exists(args.output_prep):
        os.makedirs(args.output_prep)

    # Se cargan los datos
    data = load_data(args.input_train,
                     args.input_test, output_prep_data='data/raw/data.csv')
    logging.info("Los datos fueron cargados correctamente")

    # Se hace el primer preprocesamiento de los datos
    data = preprocess_data('data/raw/data.csv',
                           output_prep_data='data/prep/data_prep.csv')
    #logging.info("Los datos fueron preprocesados correctamente")
