
'''
Este módulo contiene las funciones necesarias para hacer
inferencia con un modelo de machine learning previamente
entrenado y que fue guardado en la carpeta ./models
Ademas de solicitar al usuario los valores de las características
necesarias para hacer la predicción.
'''
import pandas as pd
import joblib


def get_user_input(feature_columns=None,
                   default_values=None,
                   input_function=input):
    '''
    Esta función se encarga de solicitar al usuario
    los valores de las características necesarias para
    hacer la predicción. Si no se proporciona un valor
    por defecto, se solicitará al usuario que ingrese
    el valor correspondiente.
    '''
    if feature_columns is None:
        feature_columns = [
            'Id', 'OverallQual', 'GrLivArea', 'FullBath', 'YearBuilt',
            'GarageCars', 'GarageArea', 'ExterQual', 'BsmtQual'
        ]

    if default_values is None:
        default_values = {}

    user_input_data = {}

    # Definir rangos de variables
    variable_ranges = {
        'OverallQual': (1, 10),
        'GrLivArea': (0, 5642),
        'FullBath': (1, 4),
        'YearBuilt': (1872, 2010),
        'GarageCars': (0, 5),
        'GarageArea': (0, 1488),
        'ExterQual': (0, 3),
        'BsmtQual': (0, 4)
    }

    for column in feature_columns:
        if column == 'Id':
            user_input_data[column] = 5000
            continue

        variable_info = get_variable_info(column)
        variable_name = variable_info['name']
        variable_range = variable_info['range']

        while True:
            user_input = input_function(
                f"Ingresa el valor para '{variable_name}' "
                f"({variable_range}): ").strip()
            if user_input == '':
                user_input = default_values.get(column, None)
            else:
                try:
                    user_input = float(user_input)
                    if not validate_input(user_input, variable_ranges[column]):
                        print(
                            "Error: El valor ingresado "
                            "está fuera del rango permitido "
                            f"{variable_ranges[column]}. Por favor, "
                            "ingresa un valor dentro del rango"
                        )
                        continue
                except ValueError:
                    print("Error: Por favor, "
                          "ingresa un número decimal válido.")
                    continue
            break

        user_input_data[column] = user_input

    user_input_df = pd.DataFrame([user_input_data], columns=feature_columns)
    return user_input_df


def validate_input(user_input, variable_range):
    '''
    Esta función se encarga de validar que el valor
    de entrada esté dentro del rango permitido
    '''
    # Comprobar si el valor de entrada está dentro del rango permitido
    return variable_range[0] <= user_input <= variable_range[1]


def get_variable_info(column):
    """
    Obtiene información adicional sobre la variable.

    Parameters:
    - column (str): Nombre de la columna.

    Returns:
    - variable_info (dict): Diccionario con el nombre y rango de la variable.
    """
    variable_info = {}

    if column == 'Id':
        variable_info['name'] = 'Identificador'
        variable_info['range'] = 'Entero positivo'
    elif column == 'OverallQual':
        variable_info['name'] = 'Calidad general'
        variable_info['range'] = 'Entero de 1 a 10'
    elif column == 'GrLivArea':
        variable_info['name'] = 'Área habitable (pies cuadrados)'
        variable_info['range'] = 'Entero de 0 a 5642'
    elif column == 'FullBath':
        variable_info['name'] = 'Baños completos'
        variable_info['range'] = 'Entero de 1 a 4'
    elif column == 'YearBuilt':
        variable_info['name'] = 'Año de construcción'
        variable_info['range'] = 'Entero de 1872 a 2010'
    elif column == 'GarageCars':
        variable_info['name'] = 'Capacidad del garaje (número de autos)'
        variable_info['range'] = 'Entero de 0 a 5'
    elif column == 'GarageArea':
        variable_info['name'] = 'Área del garaje (pies cuadrados)'
        variable_info['range'] = 'Entero de 0 a 1488'
    elif column == 'ExterQual':
        variable_info['name'] = 'Calidad del material exterior'
        variable_info['range'] = 'Entero de 0 a 3'
    elif column == 'BsmtQual':
        variable_info['name'] = 'Calidad del sótano'
        variable_info['range'] = 'Entero de 0 a 4'

    return variable_info


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
        user_input = get_user_input(feature_columns)
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
