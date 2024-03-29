# Tarea 03

_En el Capítulo 4 se aprendió sobre la importancia de escribir código limpio cuando se tiene que crear un producto de datos. Se muestra el script con el código de la Tarea 01._

### Objetivo:
-----------
Prototipar un modelo en Python que permita estimar el precio de una casa dadas algunas características que el usuario deberá proporcionar a través de un front al momento de la inferencia.
 - Python 3.11.5

``` bash
import sys
print(sys.version)
```

### Dependencias:

### Inputs/Outputs:

* En vista de que el CEO no tiene mucha claridad, podemos construir un dataset con dato sintéticos o tomar alguno otro como referencia, para poder desarrollar nuestra idea.
* Para lo cual usaremos el [conjunto de precios de compra-venta de casas de la ciudad Ames, Iowa en Estados Unidos](https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/)

### Estructura:
------------

La estructura del proyecto se ve de la siguiente forma:

```
└── project
    ├── .gitignore
    ├── README.md
    ├── src
    │   ├── .gitkeep
    │   ├── __init__.py
    │   ├── module1.py
    │   └── module2.py
    ├── data
    │   ├── raw.csv
    │   └── clean.csv
    ├── tests
    │   ├── .gitkeep
    │   ├── .test1.py
    │   └── .test2.py
    └── main_program.py
```

### Ejecución:
------------

    cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science
    py.test tests


### Arquitectura:
------------

[![Alt Text](arquitectura.png)](https://viewer.diagrams.net/?tags=%7B%7D&highlight=0000ff&edit=_blank&layers=1&nav=1&title=Tarea05_PMM.drawio#Uhttps%3A%2F%2Fdrive.google.com%2Fuc%3Fid%3D1JFuzl_OA7cfC965Q5HhdOjppitzWU7gH%26export%3Ddownload) 

### Instalación:
------------

    pip install -r requirements.txt

