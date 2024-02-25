'''
Este módulo es un script que provee
las funciones que se emplean en los
scripts de:
* Preprocesamiento de datos

El índice de las funciones es el siguiente:
* load_data
* label_encode_categorical_columns
* impute_missing_values
* train_random_forest_regressor
* impute_missing_data
* impute_categorical_missing_data
* impute_continuous_missing_data
* preprocess_data
'''
# Se importan las librerías necesarias
# pylint: disable = unused-import
import os
import warnings
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split


def load_data(data_train, data_test, output_prep_data="data/raw/data.csv"):
    '''
    Esta función se encarga de cargar los datos
    de entrada en formato .csv que se encuentran
    en la carpeta data/raw y los devuelve en un
    DataFrame de pandas.

    Parameters:
    input_data_train (str): Ruta del archivo de
                            datos de entrada en formato .csv.
    input_data_test (str): Ruta del archivo de datos de
                            entrada en formato .csv.
    path_output_data (str): Ruta del archivo de salida
                            donde se guardarán los datos
                            preprocesados en formato .csv.

    Returns:
    data (DataFrame): DataFrame de pandas que contiene
                    los datos de entrada (train y test)
    '''
    try:
        # Leer datos de entrada .csv
        df_train = pd.read_csv(data_train)
        df_test = pd.read_csv(data_test)
        # Se unen las bases
        data = pd.concat([df_train, df_test], axis=0)
        # Se eliminan las columnas que contienen más del 60% de valores nulos
        data = data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1)
        label_encoder = LabelEncoder()
        for col in data.columns:
            if data[col].dtypes == 'object':
                data[col] = label_encoder.fit_transform(data[col].astype(str))
        # Se eliminaran filas y columnas duplicadas
        data = data.drop_duplicates()
        data = data.reset_index(drop=True)

        # Guardar el resultado en un nuevo archivo CSV
        data.to_csv(output_prep_data, index=False)
        print(
            f"Las bases train y test fueron unidas y guardadas "
            f"{output_prep_data}")

        # Assert
        assert isinstance(data, pd.DataFrame)
        assert os.path.exists(output_prep_data)
        assert len(data) == len(
            pd.read_csv(data_train)) + len(pd.read_csv(data_test))
        return data
    except (FileNotFoundError, pd.errors.EmptyDataError) as exc:
        print(f"Ocurrió un error con la lectura de archivos: {exc}")
        return None


def encode_columns(data):
    '''
    Esta función se encarga de codificar las columnas
    categóricas de un DataFrame de pandas.
    '''
    label_encoder = LabelEncoder()
    for col in data.select_dtypes(['object', 'category']).columns:
        data[col] = label_encoder.fit_transform(data[col])
    return data


def impute_missing_values(data, passed_col, missing_data_cols):
    '''
    Esta función se encarga de imputar valores faltantes
    en un DataFrame de pandas.
    '''
    iterative_imputer = IterativeImputer(
        estimator=RandomForestRegressor(random_state=123), add_indicator=True)
    other_missing_col = [col for col in missing_data_cols if col != passed_col]
    for col in other_missing_col:
        if data[col].isnull().sum() > 0:
            col_missing_val = data[col].values.reshape(-1, 1)
            data[col] = iterative_imputer.fit_transform(col_missing_val)[:, 0]
    return data


def impute_categorical_missing_data(cols, data, missing_data_col, bool_cols):
    '''
    Esta función se encarga de imputar valores faltantes
    en columnas categóricas de un DataFrame de pandas.
    '''
    df_null = data[data[cols].isnull()]
    df_not_null = data[data[cols].notnull()]
    label_encoder = LabelEncoder()

    x_not_null = encode_columns(df_not_null.drop(cols, axis=1))
    y_not_null = label_encoder.fit_transform(
        df_not_null[cols]) if cols in bool_cols else df_not_null[cols]

    x_not_null = impute_missing_values(x_not_null, cols, missing_data_col)
    x_train, _, y_train, _ = train_test_split(
        x_not_null, y_not_null, test_size=0.2, random_state=123)

    rf_classifier = RandomForestRegressor()
    rf_classifier.fit(x_train, y_train)

    x_null = encode_columns(df_null.drop(cols, axis=1))
    x_null = impute_missing_values(x_null, cols, missing_data_col)

    if len(df_null) > 0:
        df_null[cols] = rf_classifier.predict(x_null)
        if cols in bool_cols:
            df_null[cols] = df_null[cols].map({0: False, 1: True})

    df_combined = pd.concat([df_not_null, df_null])
    return df_combined[cols]


def impute_continuous_missing_data(cols, data, missing_data_col, bool_cols):
    '''
    Esta función se encarga de imputar valores faltantes
    en columnas continuas de un DataFrame de pandas.
    '''
    df_null = data[data[cols].isnull()]
    df_not_null = data[data[cols].notnull()]

    x_not_null = df_not_null.drop(cols, axis=1)
    y_not_null = df_not_null[cols]

    label_encoder = LabelEncoder()
    x_not_null = encode_columns(df_not_null.drop(cols, axis=1))
    y_not_null = label_encoder.fit_transform(
        df_not_null[cols]) if cols in bool_cols else df_not_null[cols]

    x_not_null = impute_missing_values(x_not_null, cols, missing_data_col)
    x_train, _, y_train, _ = train_test_split(
        x_not_null, y_not_null, test_size=0.2, random_state=123)

    x_train, _, y_train, _ = train_test_split(
        x_not_null, y_not_null, test_size=0.2, random_state=123)

    rf_regressor = RandomForestRegressor()
    rf_regressor.fit(x_train, y_train)

    x_null = encode_columns(df_null.drop(cols, axis=1))
    x_null = impute_missing_values(x_null, cols, missing_data_col)

    if len(df_null) > 0:
        df_null[cols] = rf_regressor.predict(x_null)

    df_combined = pd.concat([df_not_null, df_null])
    return df_combined[cols]


def preprocess_data(input_data, output_prep_data="data/prep/data_prep.csv"):
    """
    Preprocesa los datos de entrada imputando valores faltantes y
    seleccionando variables relevantes para el modelo.
    Parámetros:
    input_data (cadena): la ruta al archivo de datos de entrada en formato CSV.
    output_prep_data (str): la ruta para guardar los datos
    preprocesados en formato CSV. El valor predeterminado
    es "data/prep/data_prep.csv".
    Salida:
    pandas.DataFrame: los datos preprocesados con variables seleccionadas.
    """
    # Cargar datos desde el archivo CSV
    data = pd.read_csv(input_data)
    # Se definen columnas categoricas y numericas de la base de datos
    categorical_cols = ['MSZoning',
                        'Street',
                        'LotShape',
                        'LandContour',
                        'Utilities',
                        'LotConfig',
                        'LandSlope',
                        'Neighborhood',
                        'Condition1',
                        'Condition2',
                        'BldgType',
                        'HouseStyle',
                        'RoofStyle',
                        'RoofMatl',
                        'Exterior1st',
                        'Exterior2nd',
                        'MasVnrType',
                        'ExterQual',
                        'ExterCond',
                        'Foundation',
                        'BsmtQual',
                        'BsmtCond',
                        'BsmtExposure',
                        'BsmtFinType1',
                        'BsmtFinType2',
                        'Heating',
                        'HeatingQC',
                        'CentralAir',
                        'Electrical',
                        'KitchenQual',
                        'Functional',
                        'FireplaceQu',
                        'GarageType',
                        'GarageFinish',
                        'GarageQual',
                        'GarageCond',
                        'PavedDrive',
                        'SaleType',
                        'SaleCondition']
    bools_cols = []

    numeric_cols = ['MSSubClass',
                    'LotFrontage',
                    'LotArea',
                    'OverallQual',
                    'OverallCond',
                    'YearBuilt',
                    'YearRemodAdd',
                    'MasVnrArea',
                    'BsmtFinSF1',
                    'BsmtFinSF2',
                    'BsmtUnfSF',
                    'TotalBsmtSF',
                    '1stFlrSF',
                    '2ndFlrSF',
                    'LowQualFinSF',
                    'GrLivArea',
                    'BsmtFullBath',
                    'BsmtHalfBath',
                    'FullBath',
                    'HalfBath',
                    'BedroomAbvGr',
                    'KitchenAbvGr',
                    'TotRmsAbvGrd',
                    'Fireplaces',
                    'GarageYrBlt',
                    'GarageCars',
                    'GarageArea',
                    'WoodDeckSF',
                    'OpenPorchSF',
                    'EnclosedPorch',
                    '3SsnPorch',
                    'ScreenPorch',
                    'PoolArea',
                    'MiscVal',
                    'MoSold',
                    'YrSold',
                    'SalePrice']

    warnings.filterwarnings('ignore')
    missing_data_cols = data.isnull().sum()[
        data.isnull().sum() > 0].index.tolist()

    # Imputamos informacioón en valores vacíos con nuestras funciones
    for col in missing_data_cols:
        if col in categorical_cols:
            data[col] = impute_categorical_missing_data(
                col, data, missing_data_cols, bools_cols)
        elif col in numeric_cols:
            data[col] = impute_continuous_missing_data(
                col, data, missing_data_cols, bools_cols)
        else:
            pass
    # Se selecciona las variables de interés para el modelo
    data = data[['Id', 'SalePrice', 'OverallQual',
                 'GrLivArea', 'FullBath', 'YearBuilt',
                 'GarageCars', 'GarageArea', 'ExterQual', 'BsmtQual']]

    # Guardar el resultado en un nuevo archivo CSV
    data.to_csv(output_prep_data, index=False)

    print(f"La base fue catalogada y guardada en {output_prep_data}")
    return data
