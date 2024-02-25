'''
Este módulo contiene tests para el script de entrenamiento
'''
import os
import yaml
import pytest
import pandas as pd
from scripts_train import load_config, train_model

TEST_CONFIG_PATH = 'config.yaml'
TEST_DATA_PATH = 'data_test.csv'
TEST_OUTPUT_DIR = 'models/'


def test_load_config(tmpdir):
    '''
    Test para la función load_config
    '''
    # Crea un archivo de configuración de prueba
    test_config_content = {
        'random_forest': {
            'bootstrap': True,
            'ccp_alpha': 0.0,
            'criterion': 'squared_error',
            'max_depth': None,
            'max_features': 1.0,
            'max_leaf_nodes': None,
            'max_samples': None,
            'min_impurity_decrease': 0.0,
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'min_weight_fraction_leaf': 0.0,
            'n_estimators': 100,
            'n_jobs': None,
            'random_state': None,
            'verbose': 0,
            'warm_start': False
        }
    }
    test_config_path = os.path.join(str(tmpdir), TEST_CONFIG_PATH)

    with open(test_config_path, 'w', encoding='utf-8') as test_config_file:
        yaml.dump(test_config_content, test_config_file)

    # Carga la configuración usando la función load_config
    config = load_config(config_path=test_config_path)

    # Verifica que la configuración se cargue correctamente
    assert config == test_config_content


def test_train_model(tmpdir):
    '''
    Test para la función train_model
    '''
    # Crea un archivo de datos de prueba
    test_data_content = {
        'Id': [6001, 6002, 6003, 6004, 6005],
        'SalePrice': [223500, 307000, 150000, 129900, 118000],
        'OverallQual': [7, 5, 3, 6, 5],
        'GrLivArea': [1717, 1786, 1694, 1108, 1362],
        'FullBath': [2, 3, 1, 2, 2],
        'YearBuilt': [2001, 2002, 2003, 2004, 2005],
        'GarageCars': [3, 2, 2, 1, 2],
        'GarageArea': [642.0, 548.0, 352.0, 270.0, 205.0],
        'ExterQual': [3, 2, 3, 2, 3],
        'BsmtQual': [2, 2, 3, 0, 2]
    }

    test_data_path = os.path.join(str(tmpdir), TEST_DATA_PATH)

    test_data = pd.DataFrame(test_data_content)
    test_data.to_csv(test_data_path, index=False)

    # Crea un archivo de configuración de prueba
    test_config_content = {
        'random_forest': {
            'bootstrap': True,
            'ccp_alpha': 0.0,
            'criterion': 'squared_error',
            'max_depth': None,
            'max_features': 1.0,
            'max_leaf_nodes': None,
            'max_samples': None,
            'min_impurity_decrease': 0.0,
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'min_weight_fraction_leaf': 0.0,
            'n_estimators': 100,
            'n_jobs': None,
            'random_state': None,
            'verbose': 0,
            'warm_start': False
        }
    }
    test_config_path = os.path.join(str(tmpdir), TEST_CONFIG_PATH)

    with open(test_config_path, 'w', encoding='utf-8') as test_config_file:
        yaml.dump(test_config_content, test_config_file)
    # Creamos el directorio de salida si no existe
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    # Define el directorio de salida de prueba
    test_output_dir = os.path.join(str(tmpdir), TEST_OUTPUT_DIR)

    # Entrena el modelo usando la función train_model
    train_model(data_input=test_data_path,
                config=test_config_content, output_dir=test_output_dir)

    # Verifica que el modelo se haya entrenado y guardado correctamente
    assert os.path.exists(os.path.join(test_output_dir, "rfr_model.joblib"))


if __name__ == "__main__":
    pytest.main()
