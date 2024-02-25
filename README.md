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
  ```
- Correr los scripts en el siguiente orden:
  1. prep.py
  2. train.py
  3. inference.py

## Estructura del repositorio  📂

- [README.md](README.md)
- [data](data)
  - [predictions](data/predictions)
    - [predictions.csv](data/predictions/predictions.csv)
  - [prep](data/prep)
    - [data_prep.csv](data/prep/data_prep.csv)
    - [test.csv](data/prep/test.csv)
    - [train.csv](data/prep/train.csv)
  - [raw](data/raw)
    - [data.csv](data/raw/data.csv)
    - [test.csv](data/raw/test.csv)
    - [train.csv](data/raw/train.csv)
- [inference.py](inference.py)
- [logs](logs)
- [models](models)
  - [rfr_model.joblib](models/rfr_model.joblib)
- [notebooks](notebooks)
  - [02_tarea_Yuneri_Perez.ipynb](notebooks/02_tarea_Yuneri_Perez.ipynb)
- [prep.py](prep.py)
- [src](src)
  - [pycache](src/pycache)
    - [scripts_inference.cpython-311.pyc](src/pycache/scripts_inference.cpython-311.pyc)
    - [scripts_train.cpython-311.pyc](src/pycache/scripts_train.cpython-311.pyc)
  - [scripts_inference.py](src/scripts_inference.py)
  - [scripts_prep.py](src/scripts_prep.py)
  - [scripts_train.py](src/scripts_train.py)
- [train.py](train.py)

```
.
├── LICENSE
├── README.md
├── config.yaml
├── data
│   ├── predictions
│   │   └── predictions.csv
│   ├── prep
│   │   ├── data_prep.csv
│   │   ├── test.csv
│   │   └── train.csv
│   └── raw
│       ├── data.csv
│       ├── test.csv
│       └── train.csv
├── inference.py
├── logs
├── models
│   └── rfr_model.joblib
├── notebooks
├── prep.py
├── src
│   ├── __pycache__
│   │   ├── scripts_inference.cpython-311.pyc
│   │   ├── scripts_prep.cpython-311.pyc
│   │   └── scripts_train.cpython-311.pyc
│   ├── scripts_inference.py
│   ├── scripts_prep.py
│   └── scripts_train.py
└── train.py