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
        'OverallQual - Calidad de materiales y acabados ': 8,
        'GrLivArea - Superficie habitable (nivel del suelo) pies cuadrados': 1694,
        'FullBath - Número de baños completos': 2,
        'YearBuilt - Año de construcción': 2004,
        'GarageCars - Tamaño garaje en # de coches': 2.0,
        'GarageArea -  Tamaño garaje pies cuadrados': 636.0,
        'ExterQual - Calidad de materiales exteriores': 2,
        'BsmtQual - Altura del sótano': 0
    }
    feature_columns = ['Id', 'OverallQual',
                       'GrLivArea', 'FullBath',
                       'YearBuilt', 'GarageCars',
                       'GarageArea', 'ExterQual',
                       'BsmtQual']

    # Simular la entrada de valores por el usuario durante la prueba
    monkeypatch.setattr('builtins.input', lambda _: str(default_values.get(_.strip(), _)))

    # Llamar a la función get_user_input con valores predeterminados
    user_input = get_user_input(default_values=default_values, input_function=input)

    # Asegurar que el diccionario de entrada tenga los valores correctos
    assert user_input == default_values
    # Crear un DataFrame con las características ingresadas por el usuario
    user_input_df = pd.DataFrame(
        {'Id': [5000], **user_input},
        columns=feature_columns)
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
    pytest.main(["-s"])  # Incluye la opción -s para desactivar la captura de salida estándar
