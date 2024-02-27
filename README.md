
# Predicción precio casas
# (MCD ITAM Primavera 2024)
Repositorio de un producto de datos para la predicción del precio de casas 

## Autor

| Nombre                        |  CU    | Correo Electrónico             | Usuario Github |
|-------------------------------|--------|--------------------------------|----------------|
| Yuneri Pérez Arellano         | 199813 | yperezar@itam.mx               | YunPerez       |


# Contexto  🧠
* Estamos trabajando en una start up de bienes raices y necesitamos construir un producto de datos que ayude a soportar una aplicación para  que nuestros clientes (compradores/vendedores) puedan consultar una estimación
del valor de una propiedad de bienes raíces.

* Aún el CEO no tiene claro como debe de diseñarse esta aplicación. Nostros como data scientists proponemos una Prueba de Concepto, que permita experimentar rápido, dar un look an feel de la experiencia y nos permita
fallar rápido para probar una siguiente iteración.

# Objetivo del proyecto  🎯
Desarrollar un prototipo que integre el uso de un modelo predictivo en Python para estimar el precio de una casa dadas algunas características que el usuario deberá proporcionar a través de un front al momento de la inferencia.


# Base de datos  ✍
Se empleo el [conjunto de precios de compra-venta de casas de la  ciudad Ames, Iowa en Estados Unidos](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

# Infraestructura y Ejecución  ⚙

## Requerimientos de Software herramientas recomendadas

1. [Cuenta de Github](https://github.com)
2. [VSCodeIDE](https://code.visualstudio.com/)

Para ejecutar este producto de datos se necesita lo siguiente:
- Sistema operativo Linux/Mac.
- Clonar el repositorio en el equipo.
- Activar el entorno virtual, corriendo la siguiente línea de comando en la terminal:
  ```bash
  conda env create --file environments.yml
  conda activate precioscasas
  ```
- Correr los scripts en el siguiente orden:
  1. prep.py
  2. train.py
  3. inference.py

## Estructura del repositorio  📂

- [README.md](README.md)
- [data](data)
  - [predictions](data/predictions)
  - [prep](data/prep)
  - [raw](data/raw)
- [enviroment.yml](environment.yml)
- [inference.py](inference.py)
- [logs](logs)
- [models](models)
  - [rfr_model.joblib](models/rfr_model.joblib)
- [notebooks](notebooks)
  - [02_tarea_Yuneri_Perez.ipynb](notebooks/02_tarea_Yuneri_Perez.ipynb)
- [prep.py](prep.py)
- [src](src)
  - [scripts_inference.py](src/scripts_inference.py)
  - [scripts_prep.py](src/scripts_prep.py)
  - [scripts_train.py](src/scripts_train.py)
  - [test_scripts_inference.py](src/test_scripts_inference.py)
  - [test_scripts_prep.py](src/test_scripts_prep.py)
  - [test_scripts_train.py](src/test_scripts_train.py)
- [train.py](train.py)

```
.
├── ./LICENSE
├── ./README.md
├── ./data
│   ├── ./data/predictions
│   ├── ./data/prep
│   └── ./data/raw
├── ./environment.yml
├── ./inference.py
├── ./logs
│   ├── ./logs/20240225_181924_prep.log
│   ├── ./logs/20240225_182539_train.log
│   └── ./logs/20240226_183029_inference.log
├── ./models
│   └── ./models/rfr_model.joblib
├── ./notebooks
│   └── ./notebooks/02_tarea_Yuneri_Pérez.ipynb
├── ./prep.py
├── ./src
│   ├── ./src/scripts_inference.py
│   ├── ./src/scripts_prep.py
│   ├── ./src/scripts_train.py
│   ├── ./src/test_scripts_inference.py
│   ├── ./src/test_scripts_prep.py
│   └── ./src/test_scripts_train.py
└── ./train.py


## Tests

Para correr los tests individuales para cada script, correr lo siguiente en la terminal:

```bash
  python src/test_scripts_prep.py
```
Resultado:
============================================================================================= test session starts ==============================================================================================
platform darwin -- Python 3.11.8, pytest-8.0.1, pluggy-1.4.0
rootdir: /Users/yunperez/Documents/00. Academico/00.DataScience ITAM/2do semestre (Primavera 2024)/Métodos de Gran Escala/Tareas/Tarea 03 - Codigo Limpio/Pred_Precios_Casas
plugins: anyio-4.3.0
collected 4 items

src/test_scripts_inference.py .                                                                                                                                                                          [ 25%]
src/test_scripts_prep.py .                                                                                                                                                                               [ 50%]
src/test_scripts_train.py ..                                                                                                                                                                             [100%]

============================================================================================== 4 passed in 20.82s ==============================================================================================

```bash
  python src/test_scripts_train.py
```
Resultado:
============================================================================================= test session starts ==============================================================================================
platform darwin -- Python 3.11.8, pytest-8.0.1, pluggy-1.4.0
rootdir: /Users/yunperez/Documents/00. Academico/00.DataScience ITAM/2do semestre (Primavera 2024)/Métodos de Gran Escala/Tareas/Tarea 03 - Codigo Limpio/Pred_Precios_Casas
plugins: anyio-4.3.0
collected 4 items

src/test_scripts_inference.py .                                                                                                                                                                          [ 25%]
src/test_scripts_prep.py .                                                                                                                                                                               [ 50%]
src/test_scripts_train.py ..                                                                                                                                                                             [100%]

============================================================================================== 4 passed in 20.78s ==============================================================================================

```bash
  python src/test_scripts_inference.py
```
Resultado:
============================================================================================= test session starts ==============================================================================================
platform darwin -- Python 3.11.8, pytest-8.0.1, pluggy-1.4.0
rootdir: /Users/yunperez/Documents/00. Academico/00.DataScience ITAM/2do semestre (Primavera 2024)/Métodos de Gran Escala/Tareas/Tarea 03 - Codigo Limpio/Pred_Precios_Casas
plugins: anyio-4.3.0
collected 4 items

src/test_scripts_inference.py      Id  OverallQual  GrLivArea  FullBath  YearBuilt  GarageCars  GarageArea  ExterQual  BsmtQual
0  5000            8       1694         2       2004         2.0       636.0          2         0
El precio de la casa es de: 293061.4511000001
Predicción guardada en /private/var/folders/_l/ynx4pv1119nfknfzrs_tsthr0000gn/T/pytest-of-yunperez/pytest-47/test_inference0/predictions.csv
   Predicted_SalePrice
0          293061.4511
.
src/test_scripts_prep.py La base fue catalogada y guardada en data/prep/data_prep.csv
.
src/test_scripts_train.py .El modelo fue entrenado y guardado en /private/var/folders/_l/ynx4pv1119nfknfzrs_tsthr0000gn/T/pytest-of-yunperez/pytest-47/test_train_model0/models/
.

============================================================================================== 4 passed in 21.52s ==============================================================================================

