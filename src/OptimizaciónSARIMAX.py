import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Union
from itertools import product

#Definición de función que encuentra los AIC mas bajos que optimizan el modelo SARIMAX.
def OptSarimax(endog: Union[pd.Series,list], exog: Union[pd.Series,list], order_list: list, d: int, D: int, s:int):
    Resultados = []
    for order in tqdm_notebook(order_list):
        try:
            model = SARIMAX(endog, exog, order=(order[0], d, order[1]), seasonal_order=(order[2], D, order[3], s), simple_differencing=False)
            model_fit = model.fit(disp=False)
        except:
            continue
        Resultados.append([order, model_fit.aic])

    Resultados_df = pd.DataFrame(Resultados)
    Resultados_df.columns = ['Orden', 'AIC']
    Resultados_df = Resultados_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    return Resultados_df