import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error
from sklearn import neighbors, preprocessing

pd.options.mode.chained_assignment = None

class WinePricePredictor:

    TRAIN_RATIO = 0.6

    def __init__(self, dataset_file, model):
        df = pd.read_csv(dataset_file)
        self.dataset = df[np.isfinite(df['price'])]
        self.dataset = self.dataset.drop_duplicates(subset=["country","price","designation","points","province","region_1","region_2","variety","winery"])
        self.model = model
    
    def dataset_len(self):
        return len(self.dataset.index)

    def train_dataset(self):
        data_len = self.dataset_len()
        train_limit = int(data_len*WinePricePredictor.TRAIN_RATIO)
        return self.dataset[:train_limit]

    def test_dataset(self):
        data_len = self.dataset_len()
        train_limit = int(data_len*WinePricePredictor.TRAIN_RATIO)
        return self.dataset[train_limit:]

    def x_columns(self):
        return ["country","description","designation","points","province","region_1","region_2","variety","winery"]

    def y_columns(self):
        return ["price"]

    def predict(self, X):
        df = pd.DataFrame(X)
        df = self.model.preprocess_x(df)
        return self.model.predict(df)

    def play(self):
        self.dataset = self.model.preprocess_dataset(self.dataset)
        train_df = self.train_dataset()
        x_df = train_df[self.x_columns()]
        y_df = train_df[self.y_columns()]
        self.model.fit(x_df, y_df)
        test_df = self.test_dataset()
        self.y_pred = self.model.predict(test_df[self.x_columns()])
        self.y_true = test_df[self.y_columns()]

        return {
            'mae': mean_absolute_error(self.y_true, self.y_pred),
            'mse': mean_squared_error(self.y_true, self.y_pred),
            'evs': explained_variance_score(self.y_true, self.y_pred)
        }
    
    def print_real_pred(self):
        items = np.stack((self.y_true.values[:,0], self.y_pred[:,0]), axis=-1)
        df = pd.DataFrame(data=items, columns=["Real", "Pred"])
        print(df)

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

class KNeighborsMode(EncodedWinePriceModel):
    
    def __init__(self):
        super().__init__()
        self.model = neighbors.KNeighborsRegressor(3, weights="distance")

    def preprocess_dataset(self, dataset):
        self.fit_and_transform_label_column('country', dataset)
        self.fit_and_transform_label_column('variety', dataset)
        self.fit_and_transform_label_column('province', dataset)
        self.fit_and_transform_label_column('region_1', dataset)
        self.fit_and_transform_label_column('designation', dataset)
        self.fit_and_transform_label_column('winery', dataset)
        return dataset

    def preprocess_x(self, dataset):
        self.transform_label_column('country', dataset)
        self.transform_label_column('variety', dataset)
        self.transform_label_column('province', dataset)
        self.transform_label_column('region_1', dataset)
        self.transform_label_column('designation', dataset)
        self.transform_label_column('winery', dataset)
        return dataset

    def fit(self, x_df, y_df):
        self.model.fit(x_df[["country", "variety", "province", "region_1", "designation", "winery", "points"]].values, y_df.values)

    def predict(self, x_df):
        return self.model.predict(x_df[["country", "variety", "province", "region_1", "designation", "winery", "points"]].values)

if __name__ == "__main__":
    predictor = WinePricePredictor("./data/winemag-data_first150k.csv", KNeighborsMode())
    metrics_dict = predictor.play()
    print("Mean Absolute Error {} Mean Square Error {} Explained Variance Score {}".format(
        metrics_dict['mae'],
        metrics_dict['mse'],
        metrics_dict['evs']))
    predictor.print_real_pred()
    X = {'country': ['Australia'],
        'description': ['Starts off with grassy, fresh herbal aromas and flavors of almond. After a second look, it''s all lemon and grapefruit flavors, with an edge of sweetness—almost like a lemon tart. Slim in size, soft in feel. An easy quaff.'],
        'designation': ['Zeepaard'],
        'points': [84],
        'province': ['Western Australia'],
        'region_1': ['Western Australia'],
        'region_2': [''],
        'variety': ['Sauvignon Blanc'],
        'winery': ['West Cape Howe'] }
    p = predictor.predict(X) ## real = $10
    print("Real ${:.2f} predicted ${:.2f}".format(10,p[0][0]))
    X = {'country': ['Argentina'],
        'description': ['Standard aromas in the berry and beet range is what you get on the nose. The body hits hard on the palate, where the fruit flavors seem rugged, juicy and a bit tangy. Settles on the finish, offering clarity in a tight, modest package. Imported by Biagio Cru & Estate Wines, LLC.'],
        'designation': ['Silver Reserve'],
        'points': [84],
        'province': ['Other'],
        'region_1': ['Famatina Valley'],
        'region_2': [''],
        'variety': ['Syrah'],
        'winery': ['Raza'] }
    p = predictor.predict(X) ## real = $13
    print("Real ${:.2f} predicted ${:.2f}".format(13,p[0][0]))
