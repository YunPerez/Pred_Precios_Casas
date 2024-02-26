'''
Este módulo contiene las funciones necesarias para hacer inferencia
con un modelo de machine learning previamente entrenado y que fue
guardado en la carpeta ./models
'''
# Se importan las librerías necesarias
import pandas as pd
import joblib


def get_user_input(default_values=None, input_function=input):
    '''
    Solicita al usuario ingresar los valores de las variables para predecir
    el precio de la casa.
    '''
    user_input = {}

    variable_ranges = {
        'OverallQual - Calidad de materiales y acabados ': (1, 10),
        'GrLivArea - Superficie habitable (nivel del suelo) pies cuadrados':
        (0, 5642),
        'FullBath - Número de baños completos': (1, 4),
        'YearBuilt - Año de construcción': (1872, 2010),
        'GarageCars - Tamaño garaje en # de coches': (0, 5),
        'GarageArea -  Tamaño garaje pies cuadrados': (0, 1488),
        'ExterQual - Calidad de materiales exteriores': (0, 3),
        'BsmtQual - Altura del sótano': (0, 4)
    }

    for variable, value_range in variable_ranges.items():
        while True:
            try:
                if default_values and variable in default_values:
                    default_value = default_values[variable]
                    user_input[variable] = float(default_value)
                    print(f"Usando el valor default"
                          f" {default_value} for '{variable}'.")
                else:
                    user_input[variable] = float(input_function(
                        f"{variable} - {value_range[0]}-{value_range[1]}: "
                    ))
                if value_range[0] <= user_input[variable] <= value_range[1]:
                    break
            except ValueError:
                print(f"Error: Por favor, ingrese un valor "
                      f"numérico válido para '{variable}'.")

    return user_input


def inference(output_file_path, user_input_df):
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

    # Check if user_input_df is None
    if user_input_df is None:
        # If not provided, solicit input from the user
        user_input = get_user_input()
        user_input_df = pd.DataFrame({'Id': [5000],
                                      **user_input}, columns=feature_columns)

    # Cargar el modelo previamente entrenado
    loaded_rfr = joblib.load("./models/rfr_model.joblib")

    # Realizar la predicción
    prediction = loaded_rfr.predict(user_input_df)

    # Guardar predicciones en archivo CSV
    pd.DataFrame(prediction).to_csv(
        output_file_path, index=False, header=["Predicted_SalePrice"])
    print(f"El precio de la casa es de: {prediction[0]}")
    print(f"Predicción guardada en {output_file_path}")
