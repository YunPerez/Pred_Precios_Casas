'''
Este módulo contiene las pruebas unitarias 
para la función inference del módulo scripts_inference.
'''
import pandas as pd
import pytest
from scripts_inference import inference, get_user_input

def test_inference(tmp_path, monkeypatch):
    '''
    Prueba unitaria para la función inference
    '''
    # Establecer los parámetros de entrada para inference
    output_file_path = tmp_path / "predictions.csv"
    # Definir valores predeterminados para la función get_user_input
    default_values = {
        'Id': 6000,
        'OverallQual': 8,
        'GrLivArea': 1694,
        'FullBath': 2,
        'YearBuilt': 2004,
        'GarageCars': 2.0,
        'GarageArea': 636.0,
        'ExterQual': 2,
        'BsmtQual': 0
    }
    feature_columns = [
    'Id',
    'OverallQual',
    'GrLivArea',
    'FullBath',
    'YearBuilt',
    'GarageCars',
    'GarageArea',
    'ExterQual',
    'BsmtQual']

   # Simular la entrada de valores por el usuario durante la prueba
    monkeypatch.setattr('builtins.input', lambda prompt: default_values.get(prompt.strip(), ''))

    # Llamar a la función get_user_input con valores predeterminados
    user_input = get_user_input(feature_columns, default_values=default_values, input_function=lambda prompt: str(default_values.get(prompt.strip(), '')))
    print(user_input)

    # Asegurar que el DataFrame de entrada tenga los valores correctos, manejando NaN
    assert user_input.shape == (1, len(feature_columns)), "Número incorrecto de filas o columnas."
    for column in feature_columns:
        if column == 'Id':
            # Ignorar la columna 'Id' en la comparación
            continue
        assert column in user_input.columns, f"La columna {column} no está presente en el DataFrame."
        if pd.notna(default_values[column]):
            # Asegurarse de que solo se verifica si el valor no es NaN
            assert user_input[column].iloc[0] == default_values[column], f"Valor incorrecto en la columna {column}."

        # Agregar una condición para romper el bucle si el valor es igual al valor predeterminado
        if pd.notna(default_values[column]) and user_input[column].iloc[0] == default_values[column]:
            break

    # Crear un DataFrame con las características ingresadas por el usuario
    user_input_df = pd.DataFrame(
        {'Id': [5000], **user_input},
        columns=feature_columns)  # Asegúrate de incluir la columna 'Id'

    inference(output_file_path, user_input_df)

    # Verificar que el archivo de salida se haya creado
    assert output_file_path.exists(), f"El archivo {output_file_path} no se creó correctamente."

    # Leer el archivo de salida
    predictions_df = pd.read_csv(output_file_path)

    # Imprimir el DataFrame para depuración
    print(predictions_df)

    # Verificar que el DataFrame tenga una columna llamada "Predicted_SalePrice"
    assert "Predicted_SalePrice" in predictions_df.columns, "La columna 'Predicted_SalePrice' no está presente en el DataFrame."

    # Verificar que la columna "Predicted_SalePrice" tenga al menos una predicción
    assert len(predictions_df) > 0, "El DataFrame no contiene ninguna predicción."

if __name__ == "__main__":
    pytest.main(["-s"])
