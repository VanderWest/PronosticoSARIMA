import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
from statsmodels.tsa.statespace.sarimax import SARIMAX

#Creacion de pronostico a partir de ambos modelos.
def Rolling_Forecast(df:pd.DataFrame, 
            Strain: int,         #tamaño del conjunto de entrenamiento   
            horizon: int,        #tamaño del conjunto de prueba
            window: int,         #cuanto queremos pronosticar (en meses)
            method: str) -> list:
    
    total = Strain + horizon
    iex = Strain

    if method == 'Seasonal':
        pred_Seasonal = []
        for i in tqdm_notebook(range(Strain, total, window)):
            Season = df['Cantidad de Prescripciones'][i-window:i].values
            pred_Seasonal.extend(Season)

        return pred_Seasonal

    elif method == 'SARIMA':
        pred_SARIMA = []
        for i in tqdm_notebook(range(Strain, total, window)):
            modelo = SARIMAX(train, order=(2, 1, 3), seasonal_order=(0, 1, 1, 12), simple_differencing=False)
            modelo_SARIMA = modelo.fit(disp=False)
            Pred = modelo_SARIMA.get_prediction(start=0, end=i+window-1)
            Out = Pred.predicted_mean.iloc[-window:]
            pred_SARIMA.extend(Out)

        return pred_SARIMA