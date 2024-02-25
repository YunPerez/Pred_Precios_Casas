'''
Este módulo contiene las funciones necesarias para hacer inferencia
con un modelo de machine learning previamente entrenado y que fue
guardado en la carpeta ./models
'''
# Se importan las librerías necesarias
import pandas as pd
import joblib


def get_user_input():
    '''
    Solicita al usuario ingresar los valores de las variables para predecir
    el precio de la casa.
    '''
    user_input = {}
    user_input['OverallQual'] = float(input(
        "OverallQual - Calidad de materiales y acabados "
        "(valor entre 1 y 10): "))
    user_input['GrLivArea'] = float(input(
        "GrLivArea - Superficie habitable (nivel del suelo) pies cuadrados "
        "(valor entre 0 y 5642): "))
    user_input['FullBath'] = float(input(
        "FullBath - Número de baños completos (valor entre 1 y 4): "))
    user_input['YearBuilt'] = float(input(
        "YearBuilt - Año de construcción (valor entre 1872 y 2010): "))
    user_input['GarageCars'] = float(input(
        "GarageCars - Tamaño garaje en # de coches (valor entre 0 y 5): "))
    user_input['GarageArea'] = float(input(
        "GarageArea - Tamaño garaje pies cuadrados (valor entre 0 y 1488):"))
    user_input['ExterQual'] = input(
        "ExterQual - Calidad de materiales exteriores (valor entre 0 y 3): ")
    user_input['BsmtQual'] = input(
        "BsmtQual - Altura del sótano (valor entre 0 y 4): ")
    return user_input


def inference(output_file_path):
    '''
    Esta función se encarga de hacer inferencia
    con un modelo de machine learning previamente
    entrenado y que fue guardado en la carpeta ./models
    La salida de este script es un archivo .csv con las predicciones
    '''
    # Definir las características necesarias para la predicción
    feature_columns = ['Id', 'OverallQual',
                       'GrLivArea', 'FullBath',
                       'YearBuilt', 'GarageCars',
                       'GarageArea', 'ExterQual',
                       'BsmtQual']

    # Solicitar al usuario ingresar los valores de las variables
    user_input = get_user_input()

    # Crear un DataFrame con las características ingresadas por el usuario
    user_input_df = pd.DataFrame(
        {'Id': [5000], **user_input},
        columns=feature_columns)

    # Cargar el modelo previamente entrenado
    loaded_rfr = joblib.load("./models/rfr_model.joblib")

    # Realizar la predicción
    prediction = loaded_rfr.predict(user_input_df)

    # Guardar predicciones en archivo CSV
    pd.DataFrame(prediction).to_csv(
        output_file_path, index=False, header=["Predicted_SalePrice"])
    print(f"El precio de la casa es de: {prediction[0]}")
    print(f"Predicción guardada en {output_file_path}")
