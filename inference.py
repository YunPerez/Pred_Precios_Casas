'''
Este módulo es un script que se usa para
hacer inferencia con un modelo de machine learning
previamente entrenado y que fue guardado en la carpeta ./models
La salida de este script es un archivo .csv con las predicciones
'''
# Importar librerías
import os
import logging
from datetime import datetime
import argparse
from src.scripts_inference import inference

if not os.path.exists("logs/"):
    os.makedirs("logs/")
# Setup Logging
now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M%S")
log_prep_file_name = f"logs/{date_time}_inference.log"
logging.basicConfig(
    filename=log_prep_file_name,
    level=logging.DEBUG,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def main(command_line_args):
    '''
    Función principal que ejecuta la inferencia
    '''
    # Directorios de entrada y salida
    output_pred = command_line_args.output_path

    if not os.path.exists(output_pred):
        os.makedirs(output_pred)
    output_file = os.path.join(output_pred, "predictions.csv")
    logging.info("La predicción ya fue guardada en ./data/predictions")
    # Se ejecuta la inferencia
    inference(output_file)


if __name__ == "__main__":
    logging.info("Empezando la inferencia ...")
    parser = argparse.ArgumentParser(
        description='Script para hacer inferencia con un modelo de ML')
    parser.add_argument('--model_path', type=str,
                        default='./models/rfr_model.joblib',
                        help='Ruta del modelo entrenado')
    parser.add_argument('--output_path', type=str,
                        default='./data/predictions',
                        help='Ruta de salida para las predicciones')
    args = parser.parse_args()

    main(args)
