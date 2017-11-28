# Actividad - Estimación de precio de vinos 

## Instalación

### 1 - Crear un entorno aislado (virtualenv o conda) con Python3 como intérprete

Ejemplo virtualenv

    `$ virtualenv -p python3 ./winepriceenv`
    `$ source ./winepriceenv/bin/activate`

### 2 - Instalar dependencias

    `$ pip install -r requirements.txt`

### 3 - Ejecutar script

    `$ python ./src/price_prediction.py`

## Dataset

El dataset utilizado, [winemag-data_first150k.csv](./data/winemag-data_first150k.csv?raw=true), tiene un registro de 150k vinos con las siguientes columnas: 

- country (texto): País de origen del vino.
- description (texto): Descripción del vino.
- designation (texto): El viñedo dentro de la bodega del cúal proviene el vino. 
- points (númerico): Un número en el rango [1,100] con una calificación realizada por expertos. Los vinos considerados tienen calificación > 80.
- price (númerico): Precio de una botella de vino en dolares.
- province (texto): Provincia de la cual proviene el vino.
- region_1 (texto): El área de producción de vino dentro de una provincia o estado (ejemplo Napa).
- region_2 (texto): Región más específica (este valor puede estar en blanco en algunos casos).
- variety (texto): El tipo de uva usado para hacer el vino (ejemplo malbec)
- winery (texto): La bodega que hizo el vino.

Como parte del código provisto en la actividad se remueven los duplicados y se separa el data set en 2 partes:

- Train dataset (60% sin duplicados): Utilizado para entrenar el modelo.
- Test dataset (40% sin duplicados): Utilizado para obtener las métricas sobre el modelo.

** Fuente Dataset [Kaggle](https://www.kaggle.com/ps2811/exploring-wine-reviews/data)

## Problema

A partir del dataset presentado anteriormente obtener el modelo que mejor ajuste la estimación del precio.

## Interfáz modelo

```python
class WinePriceModel:
    """Esta clase define la interfaz que debe ser implementada por un modelo."""
    
    def preprocess_dataset(self, dataset):
        """Recibe como parametro un dataset del tipo pandas.DataFrame
        y lo procesa antes de ser utilizado."""
        pass

    def preprocess_x(self, dataset):
        """Recibe como parametro una set de entrada X del tipo pandas.DataFrame
          y lo procesa antes de ser predecido"""
        pass

    def fit(self, x_df, y_df):
        """Entrena el modelo partiendo de dos pandas.DataFrame x_df e y_df."""
        pass

    def predict(self, x_df):
        """Retorna la predicción para las entradas en el pandas.DataFrame x_df."""
        pass
```

## Encoding

La siguiente clase permita manejar la codificación de columnas del tipo texto, por ejemplo si consideramos la columna "country", tal cual como está no puede ser ingresa a un modelo de regresión, previamente debe ser transformada en un valor númerico. El encoder utilizado por este clase le asigna un valor en el rango [0, N-1] a las N clases de una columna.
Adicionales encoders puedes ser considerados y agregados por cada modelo.

```python
class EncodedWinePriceModel(WinePriceModel):
    """Esta es una clase de modelo abstracta que agrega la capacidad de manejar 
    encoding por columnas. El encoder implementado es LabelEncoder el cual genera
    a partir N diferentes clases de valores, sus correspondientes valores numericos
    desde 0 a N-1."""
    
    def __init__(self):
        self.model = None
        self.encoders = {}

    def transform_label_column(self, column_name, dataset):
        """Aplica el enconder asociado a la columna column_name en el dataset."""
        le = self.encoders[column_name]
        dataset[column_name] =  dataset[column_name].apply(str)
        dataset[column_name] = le.transform(dataset[column_name].values)
        return dataset
        
    def fit_and_transform_label_column(self, column_name, dataset):
        """Procesa una columna del dataset utilizando un LabelEncoder."""
        le = None
        if column_name in self.encoders:
            le = self.encoders[column_name]
        else:
            le = preprocessing.LabelEncoder()
            self.encoders[column_name] = le
        dataset[column_name] =  dataset[column_name].apply(str)
        vals = dataset[column_name].values
        le.fit(vals)
        return self.transform_label_column(column_name, dataset)
```

## Invocación

Para invocar un modelo diferente al provisto, cambiar la clase con la implementación que corresponda en el siguiente código:

```python
    predictor = WinePricePredictor("./data/winemag-data_first150k.csv", KNeighborsMode())
    metrics_dict = predictor.play()
```

## Métricas

El método _play()_ retorna un diccionario con 3 métricas utilizadas para medir el modelo:

- [Mean Absolute Error 'mae'](http://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-error)
- [Mean Squared Error 'mse'](http://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error)  
- [Explained variance score 'evs'](http://scikit-learn.org/stable/modules/model_evaluation.html#explained-variance-score)