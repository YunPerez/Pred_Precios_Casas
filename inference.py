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
from src.scripts_inference import inference, get_user_input

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


logging.info("Inferencia iniciada ...")


def main(command_line_args):
    '''
    Función principal que ejecuta la inferencia
    '''
    logging.info("Cargando el modelo ...")
    # Directorios de entrada y salida
    output_pred = command_line_args.output_path
    logging.info("El modelo fue cargado exitosamente")
    if not os.path.exists(output_pred):
        os.makedirs(output_pred)
    output_file = os.path.join(output_pred, "predictions.csv")
    logging.info("La predicción ya fue guardada en ./data/predictions")

    # Definir las características necesarias para la predicción
    feature_columns = ['Id', 'OverallQual',
                       'GrLivArea', 'FullBath',
                       'YearBuilt', 'GarageCars',
                       'GarageArea', 'ExterQual',
                       'BsmtQual']

    # Solicitar entrada del usuario
    user_input = get_user_input(feature_columns)
    logging.info("Entrada del usuario obtenida")
    # Se ejecuta la inferencia
    inference(output_file, user_input)
    logging.info("Inferencia finalizada")


if __name__ == "__main__":
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
