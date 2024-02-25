'''
Este módulo contiene tests para el script de entrenamiento
'''
import os
import yaml
import pytest
from scripts_train import load_config

TEST_CONFIG_PATH = 'config.yaml'


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


if __name__ == "__main__":
    pytest.main()
