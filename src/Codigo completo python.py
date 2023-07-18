import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Union
from itertools import product

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Data Limpia.csv')
#renombramos las columnas a Fecha y Cantidad de Prescripciones
df.columns = ['Fecha', 'Cantidad de Prescripciones']

#Utilizamos otro gráfico ya que claramente vemos una tendencia pero no podemos ver los datos con claridad.
#utilizamos un subplot
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(df['Fecha'], df['Cantidad de Prescripciones'])
ax.set(xlabel='Fecha', ylabel='Prescripciones', title='Número de prescripciones por mes')
ax.grid()
x_ticks = plt.gca().get_xticks()
plt.xticks(rotation=45)
plt.gca().set_xticks(x_ticks[::12])
plt.show()

#Utilizaremos el paquete STL de statsmodels para observar las componentes de la serie de tiempo
stl = STL(df['Cantidad de Prescripciones'], period=12)   #periodo de 12 meses
res = stl.fit()
fig = res.plot()
#cambiamos los nombres de las leyendas de cada uno de los graficos
fig.axes[0].set_ylabel('Prescripciones')
fig.axes[1].set_ylabel('Tendencia')
fig.axes[2].set_ylabel('Estacionalidad')
fig.axes[3].set_ylabel('Residuos')
#tamaño de la figura
fig.set_figwidth(20)
fig.set_figheight(10)
#Configurar el eje X con fechas solo por años dentro de la columna Date
plt.xticks(np.arange(6, 203, 12), np.arange(1992, 2009, 1))
plt.show()

#Función para probar la estacionaridad de la serie de tiempo
ADF = adfuller(df['Cantidad de Prescripciones'])
print('ADF Statistic: %f' % ADF[0])
print('p-value: %f' % ADF[1])

Diff = np.diff(df['Cantidad de Prescripciones'],1)
ADF = adfuller(Diff)
print('ADF Statistic: %f' % ADF[0])
print('p-value: %f' % ADF[1])

Season_Diff = np.diff(df['Cantidad de Prescripciones'],12)
ADF = adfuller(Season_Diff)
print('ADF Statistic: %f' % ADF[0])
print('p-value: %f' % ADF[1])

train = df['Cantidad de Prescripciones'][:156]
test = df['Cantidad de Prescripciones'][156:]

fig, ax = plt.subplots()
ax.plot(df['Cantidad de Prescripciones'], label='Prescripciones')
ax.set_xlabel('Fecha')
ax.set_ylabel('Número de Prescripciones')
ax.axvspan(156, 204, color='grey')
fig.autofmt_xdate()
plt.legend()
plt.xticks(np.arange(6, 203, 12), np.arange(1992, 2009, 1))
plt.tight_layout()

#Modelo de optimización SARIMAX
def OptSarimax(endog: Union[pd.Series,list], exog: Union[pd.Series,list], order_list: list, d: int, D: int, s:int):
    Resultados = []
    for order in order_list:
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
ps = range(0, 5, 1)
qs = range(0, 5, 1)
Ps = range(0, 5, 1)
Qs = range(0, 5, 1)
order_list = list(product(ps, qs, Ps, Qs))
d=1   #diferenciación
D=1   #diferenciación estacional
s=12  #periodo de la estacionalidad

#Optimización del modelo SARIMAX
#Resultados_SARIMA = OptSarimax(train, None, order_list, d, D, s)
#Resultados_SARIMA

#Modelo SARIMA
SARIMA = SARIMAX(train, order=(2, 1, 3), seasonal_order=(0, 1, 1, 12), simple_differencing=False)
SARIMA_fit = SARIMA.fit(disp=False)
SARIMA_fit.plot_diagnostics(figsize=(15, 12))

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
        for i in range(Strain, total, window):
            Season = df['Cantidad de Prescripciones'][i-window:i].values
            pred_Seasonal.extend(Season)

        return pred_Seasonal
    elif method == 'SARIMA':
        pred_SARIMA = []
        for i in range(Strain, total, window):
            modelo = SARIMAX(train, order=(2, 1, 3), seasonal_order=(0, 1, 1, 12), simple_differencing=False)
            modelo_SARIMA = modelo.fit(disp=False)
            Pred = modelo_SARIMA.get_prediction(start=0, end=i+window-1)
            Out = Pred.predicted_mean.iloc[-window:]
            pred_SARIMA.extend(Out)
        return pred_SARIMA

Predicciones = df[156:]
Predicciones['Seasonal'] = Rolling_Forecast(df, 156, 48, 12, 'Seasonal')
Predicciones['SARIMA'] = Rolling_Forecast(df, 156, 48, 12, 'SARIMA')
#Las graficamos
fig, ax = plt.subplots()
ax.plot(df['Cantidad de Prescripciones'])
ax.plot(Predicciones['Cantidad de Prescripciones'],'b-',color='black', label='Prescripciones')
ax.plot(Predicciones['Seasonal'], 'r:', color='red', label='Seasonal')
ax.plot(Predicciones['SARIMA'], 'b--', color='green', label='SARIMA')
ax.set_xlabel('Fecha')
ax.set_ylabel('Número de Prescripciones')
ax.axvspan(156, 204, color='grey')
fig.autofmt_xdate()
plt.legend()
plt.xticks(np.arange(6, 203, 12), np.arange(1992, 2009, 1))
plt.xlim(100, 204)
plt.tight_layout()

#Optimización del modelo SARIMAX con una muestra de 3 años en la prueba
train2 = df['Cantidad de Prescripciones'][:168]
test2 = df['Cantidad de Prescripciones'][168:]
def OptSarimax(endog: Union[pd.Series,list], exog: Union[pd.Series,list], order_list: list, d: int, D: int, s:int):
    Resultados = []
    for order in order_list:
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
ps = range(0, 5, 1)
qs = range(0, 5, 1)
Ps = range(0, 5, 1)
Qs = range(0, 5, 1)
order_list = list(product(ps, qs, Ps, Qs))
d=1
D=1
s=12
#Resultados_SARIMA2 = OptSarimax(train2, None, order_list, d, D, s)
#Resultados_SARIMA2

#Modelamos y pronosticamos con los datos de 1992 a 2008
def Rolling_Forecast2(df:pd.DataFrame, 
            Strain: int,         #tamaño del conjunto de entrenamiento   
            horizon: int,        #tamaño del conjunto de prueba
            window: int,         #cuanto queremos pronosticar (en meses)
            method: str) -> list:
    total = Strain + horizon
    iex = Strain
    if method == 'Seasonal':
        pred_Seasonal = []
        for i in range(Strain, total, window):
            Season = df['Cantidad de Prescripciones'][i-window:i].values
            pred_Seasonal.extend(Season)
        return pred_Seasonal

    elif method == 'SARIMA':
        pred_SARIMA = []
        for i in range(Strain, total, window):
            modelo = SARIMAX(train2, order=(3, 1, 1), seasonal_order=(1, 1, 3, 12), simple_differencing=False)
            modelo_SARIMA = modelo.fit(disp=False)
            Pred = modelo_SARIMA.get_prediction(start=0, end=i+window-1)
            Out = Pred.predicted_mean.iloc[-window:]
            pred_SARIMA.extend(Out)
        return pred_SARIMA
Predicciones_2 = df[168:]
Predicciones_2['Seasonal'] = Rolling_Forecast2(df, 168, 36, 12, 'Seasonal')
Predicciones_2['SARIMA'] = Rolling_Forecast2(df, 168, 36, 12, 'SARIMA')
fig, ax = plt.subplots()
ax.plot(df['Cantidad de Prescripciones'])
ax.plot(Predicciones_2['Cantidad de Prescripciones'],'b-',color='black', label='Prescripciones')
ax.plot(Predicciones_2['Seasonal'], 'r:', color='red', label='Seasonal')
ax.plot(Predicciones_2['SARIMA'], 'b--', color='green', label='SARIMA')
ax.set_xlabel('Fecha')
ax.set_ylabel('Número de Prescripciones')
ax.axvspan(168, 204, color='grey')
fig.autofmt_xdate()
plt.legend()
plt.xticks(np.arange(6, 203, 12), np.arange(1992, 2009, 1))
plt.xlim(100, 204)
plt.tight_layout()

#MAPE
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
MAPE_Seasonal   = mape(Predicciones['Cantidad de Prescripciones'], Predicciones['Seasonal'])
MAPE_SARIMA_T48 = mape(Predicciones['Cantidad de Prescripciones'], Predicciones['SARIMA'])     #(2,3,0,1)
MAPE_SARIMA_T36 = mape(Predicciones_2['Cantidad de Prescripciones'], Predicciones_2['SARIMA']) #(3,1,1,3)
print('MAPE Seasonal: ', MAPE_Seasonal, 'MAPE SARIMA T48: ', MAPE_SARIMA_T48, 'MAPE SARIMA T36: ', MAPE_SARIMA_T36)

#Forecast
forecast_sarima = []
modelo = SARIMAX(train2, order=(3, 1, 1), seasonal_order=(1, 1, 3, 12), simple_differencing=False)
modelo_SARIMA = modelo.fit(disp=False)
Pred = modelo_SARIMA.get_forecast(6)
Out = Pred.predicted_mean
forecast_sarima.extend(Out)

df = pd.DataFrame({
    'Fecha': ['2008-08-01', '2008-09-01', '2008-10-01', '2008-11-01', '2008-12-01', '2009-01-01'],
    'Cantidades pronosticadas de prescripciones': forecast_sarima
})
df