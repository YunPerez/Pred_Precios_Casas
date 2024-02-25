'''
Este modulo es un script que se usa para
entrenar un modelo de machine learning
usando la información previamente preprocesada y
que fue guardada en la carpeta ./data/prep
dividiendo la base de datos en train y test
para dicho proposito.
'''
# Se importan las librerías necesarias
import os
import logging
from datetime import datetime
import argparse
from src.scripts_train import load_config, train_model

if not os.path.exists("logs/"):
    os.makedirs("logs/")
# Setup Logging
now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M%S")
log_prep_file_name = f"logs/{date_time}_train.log"
logging.basicConfig(
    filename=log_prep_file_name,
    level=logging.DEBUG,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    logging.info("Empezando el entrenamiento ...")
    # Se configura el parser de argumentos
    parser = argparse.ArgumentParser(
        description='Entrenar modelo de machine learning')
    parser.add_argument('--input', type=str,
                        default='./data/prep/data_prep.csv',
                        help='Ruta del archivo de entrada')
    parser.add_argument('--output', type=str,
                        default='./models/',
                        help='Ruta de la carpeta de salida')
    parser.add_argument('--config', type=str,
                        default='config.yaml',
                        help='Ruta del archivo de configuración')

    # Se obtienen los argumentos proporcionados por el usuario
    args = parser.parse_args()

    # Se asignan los paths de entrada y salida
    INPUT = args.input
    OUTPUT_MOD = args.output
    config = load_config(args.config)
    logging.info("La configuración del modelo fue cargada correctamente")
    # Asegurarse de que la carpeta de salida exista, si no, crearla
    if not os.path.exists(OUTPUT_MOD):
        os.makedirs(OUTPUT_MOD)

    # Se entrena el modelo
    train_model(INPUT, config, OUTPUT_MOD)
    print("El modelo fue entrenado y guardado en la carpeta models")
    logging.info("El modelo fue entrenado y guardado en la carpeta models")
