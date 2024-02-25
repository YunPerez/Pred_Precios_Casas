'''
Este módulo contiene las pruebas unitarias
para el módulo scripts_prep.py
'''
import pandas as pd
import pytest
from scripts_prep import preprocess_data


def test_preprocess_data():
    '''
    Test de la función
    preprocess_data
    '''
    # Establecer los parámetros de entrada
    input_data = 'data/raw/data.csv'
    output_prep_data = "data/prep/data_prep.csv"

    # Ejecutar la función
    result = preprocess_data(input_data, output_prep_data)

    # Verificar el tipo de dato de salida
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert set(result.columns) == {'Id', 'SalePrice', 'OverallQual',
                                   'GrLivArea', 'FullBath', 'YearBuilt',
                                   'GarageCars', 'GarageArea', 'ExterQual',
                                   'BsmtQual'}
    assert result.isnull().sum().sum() == 0
    assert result['Id'].dtype == int
    assert result['SalePrice'].dtype in (int, float)
    assert result['OverallQual'].dtype in (int, float)
    assert result['GrLivArea'].dtype in (int, float)
    assert result['FullBath'].dtype in (int, float)
    assert result['YearBuilt'].dtype in (int, float)
    assert result['GarageCars'].dtype in (int, float)
    assert result['GarageArea'].dtype in (int, float)
    assert (result['ExterQual'] == result['ExterQual'].astype(object)).all()
    assert (result['BsmtQual'] == result['BsmtQual'].astype(object)).all()


if __name__ == "__main__":
    pytest.main()
