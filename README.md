# Pronóstico de prescripciones de medicinas anti-diabetes en Australia para el segundo semestre del año 2008.

Se busca pronosticar a través de series de tiempo el número de prescripciones de medicamentos anti-diabetes para los próximos 6 meses en Australia utilizando una muestra sobre el numero de prescripciones realizadas entre los años 1991 y 2008, Para ello se propone el uso del modelo de machine learning SARIMA, el cual es un modelo de pronósticos en base a la integración de coeficientes y errores tomando en consideración datos que tienen una tendencia estacional, a diferencia de ARIMA que omite la estacionalidad para su entrenamiento.
Veremos las razones para elegir SARIMA en lugar de otros modelos y sus respectivas métricas para medir la calidad del modelo.

Los datos para este desafío fueron obtenidos de https://www.key2stats.com/data-set/view/745 lo cuales corresponden al número de medicamentos anti-diabetes prescritos en Australia entre los años 1991 y 2008.

Data ya procesada:

![Data limpia](https://github.com/VanderWest/Proyecto/blob/Reports/Imagenes/DF.PNG?raw=true)

**Análisis exploratorio**

Partamos por asumir que no sabemos que modelo utilizar, realizamos un análisis exploratorio sobre la data para ver sí existen patrones que puedan guiarnos, lo único que sabemos es que estamos trabajando con series de tiempo y queremos pronosticar (de esta forma se reduce en número de modelos a utilizar en la bolsa). Graficando los datos obtenemos una idea sobre que está ocurriendo:


![Gráfica](https://github.com/VanderWest/Proyecto/blob/Reports/Imagenes/Gr%C3%A1fica%20prescripciones%20por%20mes.png?raw=true)

Con esta sola gráfica se pueden observar 3 patrones: Primero la tendencia creciente que tienen los datos conforme pasa los años. Segundo, la estacionalidad de los datos, creando patrones repetitivos en los años. Tercero, una suerte de ruido blanco, mostrando lo errático que se van volviendo los valores de los datos sin perderse en la aleatoriedad.

Dado lo anterior podemos realizar un desglose de la gráfica utilizando el paquete STL (seasonal-trend decomposition) para ver en detalle que es lo que ocurre:

![STL](https://github.com/VanderWest/Proyecto/blob/Reports/Imagenes/STL.png?raw=true)

- En la primera grafica observamos lo anterior, el numero de prescripciones entre los años 1991 y 2008.
- La segunda, tendencia, nos muestra el comportamiento creciente del número de prescripciones recetadas al paso de los años.
- La tercera, estacionalidad, muestra el patrón cíclico del número de prescripciones recetadas entre los años 1991 y 2008.
- La cuarta, residuos, denota las diferencias que hay entre los valores pronosticados, juntando la tendencia y la estacionalidad, contrastado con los valores reales presentados en la data, mientras mas cerca del 0 estén los valores, mas acertado es el pronóstico.

Ahora queremos desarrollar el modelo que nos ayudará a pronosticar los próximos 6 meses, para ello, debemos empezar por ver que nuestra data presentada no tenga tendencia, sea estacionario y no tenga autocorrelación entre sus datos, aplicamos de esta forma el método ADF comprobarlo. ADF nos ayuda a comprobar si es que existe tendencia y estacionalidad en la serie de tiempo.

Que sea estacionario es una de las características mas fundamentales dado que de esta forma el comportamiento de las series de tiempo puede ser predecible debido a que sus propiedades estadísticas (como la media o la varianza) se mantienen constantes en el tiempo, no solo eso, sino que el modelo que buscamos aplicar actúa sobre series de tiempo estacionarias.

ADF:
- ADF Statistic: 3.145190
- p-value: 1.000000

Ambos valores nos indican que nuestra data no es estacionaria y que existe una tendencia en los datos, ADF utiliza hipótesis para comprobar ambos dos, si esque un p-value menor a 0.05 y el ADF es un número negativo alto, se puede rechazar lo que llamamos *hipótesis nula* y confirmar que se trata de una serie de tiempo estacional y sin tendencia.

Aquí aplicamos un método de transformación para que la serie de tiempo sea estacionaria, utilizando la diferenciación, para ello solo aplicamos la función diff de numpy, que ira tomando la diferencia en los pares de datos y creando una nueva serie de tiempo, esperando que esta última si sea estacionaria.

ADF:
- ADF Statistic: -2.495174
- p-value: 0.116653

Obtenemos que el ADF ahora es negativo, pero aún nos falta que el valor p sea menor a 0.05, por lo que volveremos a transformar, ahora en vez de realizar una diferenciación simple aplicaremos una diferenciación estacional, ya que observamos anteriormente que nuestra serie de tiempo presenta patrones estacionales, para ello utilizaremos la misma función pero esta vez con un hiperparametro denotando la cantidad de pasos por estación (n=12).

- ADF Statistic: -18.779673
- p-value: 0.000000

Luego de estas transformaciones, una diferenciación (d=1) y una diferenciación estacional (D=1), obtenemos que nuestra serie de tiempo es estacionaria y sin tendencia, por lo que podemos pasar al siguiente paso, modelar.

**Modelamiento**

Para el entrenamiento del modelo no podemos ocupar el clásico train_test_split, esto porque es imperativo que las series de tiempo sean trabajadas en orden, el TTS toma porcentajes aleatorios de la muestra como train y test, pero para casos de este proyecto, se debe tomar un entrenamiento con cierto porcentaje de la data desde su valor inicial hasta un valor arbitrario, de acuerdo a lo que sea pertinente al modelo.

Tenemos data de 204 meses, por lo que utilizaremos 156 meses para el entrenamiento y los ultimos 4 años para realizar las pruebas:

![traintest](https://github.com/VanderWest/Proyecto/blob/Reports/Imagenes/Gris.png?raw=true)

La parte en gris denota la muestra que será utilizada para le prueba, mientras que el resto de la data será para entrenar el modelo.

Una vez definidas las muestras de entrenamiento y prueba, procedemos a desarrollar el modelo. Como se mencionó inicialmente, el modelo a utilizar es SARIMA, la razón para ocupar este por sobre ARIMA es que nuestra serie de tiempo presenta patrones estacionales, siendo SARIMA una extensión de ARIMA que toma en cuenta estos patrones. 

Hay que tomar en cuenta que SARIMA es un caso especial de SARIMAX, la mayor cualidad de este último es que toma en consideración variables exógenas, siendo esta cualquiera característica externa que afecte a la serie de tiempo, para casos de este proyecto, solo tomamos en consideración la cantidad de prescripciones realizadas en un mes, no hay factores externos que influyan.

Respecto al modelo elegido, SARIMA es un modelo que utiliza parámetros p, d, q, P, D, Q o bien SARIMA(p,d,q)(P,D,Q)m, para su modelación se seguirá la estructura de un SARIMAX (dado que es un caso especial), por lo que primero hay que empezar encontrar los valores de parámetros que optimicen el SARIMAX, conocemos ya 3 de ellos, siendo estos d, D y m, como 1, 1 y 12 respectivamente (estos valores de d y D son los que transforman nuestra serie de tiempo en estacionaria).

Hondando un poco en el elegido, SARIMA es un modelo integrado estacional de dos otros dos modelos también utilizados para pronosticar series de tiempo:
- AR(p) o *autoregressive model* el cual toma la una regresión linea de una variable con ella misma, considerando la dependencia lineal del valor actual con los valores pasados, donde el parámetro p corresponde a la cantidad de valores pasados que afectan al valor actual.
- MA(q) o *moving average model* el cual toma la dependencia lineal entre los errores del valor actual y los errores de los valores pasados, estos errores suelen tener una distribución normal, donde el parámetro q corresponde a la cantidad de errores pasados que afectan al valor actual.
- Los parámetros P y Q tienen la misma interpretación solo que toman la cantidad de patrones estacionales respecto a los valores y a los errores pasados (en periodo estacional) que afectan al valor actual.

Con SARIMA(p,1,q)(P,1,Q)12 solo queda terminar de optimizar para encontrar valores para los parámetros faltantes que minimicen el AIC (Akaike's information criterion), este criterio nos ayuda a seleccionar el mejor modelo estadístico entre varios modelos presentados.

Llamamos a la función OptSarimax(train, None, order_list, d, D, s) para optimizar (el código pueden encontrarlo en la Branch de desarrollo&prueba):
- *train* recibe el conjunto de variables endógenas que se utiliza en el entrenamiento del modelo.
- *None* es el argumento que utilizamos para decirle a nuestra función que no usaremos variables exógenas para entrenar el modelo.
- *order_list* recibe las múltiples combinaciones de valores para utilizar en los parámetros, esta función está usando 625 combinaciones diferentes para optimizar, así es como lo definimos:
![ejemploCombinaciones](https://github.com/VanderWest/PronosticoSARIMA/blob/Reports/Imagenes/Combinaciones.PNG?raw=true)
- *D* y *d* corresponden a las cantidades de diferenciaciones estacionales y diferenciaciones simples que se utilizaron para que nuestra serie de tiempo sea estacionaria, siendo ambos valores 1 para este caso.
- *s* corresponde a la duración en puntos del patrón estacional, en nuestro caso el patron tiene una duración de 12 meses.

Utilizando lo mencionado anteriormente se consiguen las siguientes combinaciones de patrones en forma de (p,q,P,Q)

![Primer AIC](https://github.com/VanderWest/Proyecto/blob/Reports/Imagenes/AIC.PNG?raw=true)

Obtenemos 625 combinaciones diferentes para encontrar los parámetros adecuados.

De nuevo, ¿Para qué queriamos estos valores? estos son los valores de los parámetros para los cuales se minimiza el AIC, en el orden (p,q,P,Q), como observamos en el DataFrame, estos ya estan ordenados en orden ascendente con respecto a su AIC, por lo que los valores que minimizan mejor el AIC son (2, 3, 0, 1).

Ahora tenemos todos los valores para los parámetros del mejor modelo SARIMA (según la minimización del AIC), con esto por fin podemos modelar completamente el modelo.

SARIMA(p,d,q)(P,D,Q)m = SARIMA(2,1,3)(0,1,1)12

El modelo entrenado con los parámetros anteriores nos entrega el siguiente diagnóstico:

![Diagnóstico](https://github.com/VanderWest/Proyecto/blob/Reports/Imagenes/diagnostic.png?raw=true)

¿Que observamos en estos gráficos?

- Superior izquierdo nos indica la tendencia, no hay, la varianza pareciera mantenerse constante, por lo podemos asumir que tenemos un modelo estacionario.
- Superior derecho muestra la distribución que tiene el residuo siendo similar a una Normal.
- Inferior izquierdo muestra la relación lineal que existe entre los valores de muestra y los teoricos dados por el modelo.
- Inferior derecho indica la correlación existente entre las variables, tampoco hay, estos pequeños valores se asimilan a lo que llamamos anteriormente como ruido blanco.

**Predicciones y pronóstico**

Hay que tener un poco de cuidado, utilizaremos 4 años de data para testear y así pronosticar 6 meses, pero no queremos caer en que aun así sigue siendo muy pequeña la muestra, por lo que además del modelo SARIMA, se utilizará también un modelo estacional como referencia, que utilizará los 12 últimos meses de la data como muestra para pronosticar los siguientes 6 meses.

Llamamos a la función Rolling_Forecast(df, Strain, horizon, window, method) para realizar las predicciones (el código pueden encontrarlo en la Branch de desarrollo&prueba), esta recibe los siguientes parámetros:
- *df* : El Dataframe trabajado.
- *Strain* : Tamaño de la muestra de entrenamiento.
- *horizon* o Horizonte, que corresponde al tamaño de la muestra de prueba.
- *window* o Ventana, que corresponde a cuantos puntos queremos pronosticar.
- *method* : cuál es el modelo que se utilizará en el pronóstico.

De esta forma podemos crear una comparación entre las predicciones de varios modelos simultáneamente. Asignamos al método los siguientes valores y continuamos con la predicción:

- Strain = 156    Tamaño del conjunto de entrenamiento.
- Horizon = 48    Tamaño del conjunto de prueba.
- Window = 12     Pasos a predecir.

Como se mencionó antes, la razón para trabajar con una ventana de 12 es porque estamos aplicando get_predict en nuestro modelo en lugar de get_forecast, mas adelante utilizaremos la segunda función pero primero queremos evaluar el desempeño del modelo.

Finalmente utilizamos el en *method* a ‘SARIMA’ y luego ‘Seasonal’ para guardar sus predicciones y así mostrar su comportamiento en el siguiente gráfico para el Rolling Forecast:

![Rolling1](https://github.com/VanderWest/Proyecto/blob/Reports/Imagenes/GrafoProno1.png?raw=true)

 *considerar que este modelo lleva un SARIMA entrenado con los parámetros (2,1,3)(0,1,1)12, si se desea cambiar los parámetros habrá que hacerlo desde el archivo de Python*

Podemos observar que poco a poco SARIMA se aleja de los valores reales, y esto quizas es debido a que ocupamos pocos datos para el entrenamiento, asique volveremos a entrenar un nuevo SARIMA, pero en lugar de dejar 4 años para prueba, dejaremos solo 3 y el resto lo dejaremos para entrenamiento, para esto debemos realizar los mismo pasos que hemos seguido hasta hora desde cero:

- Optimizar para encontrar parámetros.
- Modelar con los parámetros encontrados.
- Pronosticar.

Optimizando con 168 meses obtenemos las siguientes combinaciones para AIC:

![AIC2](https://github.com/VanderWest/Proyecto/blob/Reports/Imagenes/AIC2.PNG?raw=true)

Con esto entendemos que los parámetros (p,q,P,Q) toman los valores (3,1,1,3) para así minimizar el criterio.

Entrenamos nuestro SARIMA con estos valores y el Rolling Forecast toma la siguente forma:

![Rolling2](https://github.com/VanderWest/Proyecto/blob/Reports/Imagenes/GrafoProno2.png?raw=true)

Notamos que ahora las predicciones de SARIMA se acercan más a los valores reales, por lo que ahora procedemos a evaluar el desempeño de los modelos, así podemos tener una idea de la calidad de estos a través de sus métricas.

**Evaluación de los pronósticos**

Aquí evaluaremos la calidad de los modelos a través de la métrica MAPE (Mean Absolute Percentage Error) el cual es el indicador de desempeño para modelos de pronostico en series de tiempo. Este evalua los modelos midiendo el error porcentual promedio entre los valores pronosticados y los valores reales:

![MAPE](https://github.com/VanderWest/Proyecto/blob/Reports/Imagenes/MAPE.PNG?raw=true)

- n corresponde al número de pares de valores predicción-real.
- A es el valor real. 
- F es el valor predicho.

Los MAPE obtenidos:
- Seasonal               = 22.58%
- SARIMA(2,1,3)(0,1,1)12 = 10.07%
- SARIMA(3,1,1)(1,1,3)12 = 9.33%

El modelo desarrollado utilizando la predicción a través de la última temporada tiene un porcentaje de error promedio más alto que ambos SARIMA, y en general ambos SARIMA tiene un rendimiento muy bueno, particularmente SARIMA(3,1,1)(1,1,3) es mejor que el anterior con una diferencia no muy significativa entre ellos a pesar de tener año de data de entrenamiento más que SARIMA(2,1,3)(0,1,1).

Podemos intentar reducir aún más el horizonte para entrenar, pero estaremos arriesgando un overfitting en el modelo.

**Volviendo al objetivo**

Finalmente nos enfocamos en el objetivo inicial, pronosticar los siguientes 6 meses de prescripciones anti-diabeticas en Australia, para ello debemos retocar nuestro modelo.
Antes estábamos prediciendo en el modelo para las muestras utilizando la función get_predict, sin embargo, ahora queremos pronosticar 6 meses que desconocemos, por lo que utilizamos get_forecast.

Get_forecast recibe la cantidad de pasos que queremos pronosticar a partir de nuestro modelo y nos entrega estos valores, así, colocando una ventana igual a 6 obtenemos el pronostico de los meses que desconocemos.

![Cantidades](https://github.com/VanderWest/Proyecto/blob/Reports/Imagenes/Cantidades%20pronosticadas.PNG?raw=true)

**Conclusiones**

- Luego de modelar SARIMA obtenemos un error porcentual de 9.33% de que las cantidades pronosticadas difieran de las cantidades reales que vayan a ser descritas en lo que queda del 2008.

- Mientras mas bajo sea el valor del MAPE, mayor es la calidad de pronostico del modelo, y un MAPE de 9.33% es un porcentaje aceptable de error para la cantidad de data que estamos trabajando, sin embargo, esto podría mejorarse si esque tuvieramos una mayor muestra de datos, recordemos que tenemos muestras mensuales entre los años 1991 y 2008.

- También es posible que el modelo pudiera mejorarse a partir de variables exógenas, aunque este modele no posee tales variables, recordemos que SARIMA es un caso especial de SARIMAX, donde este último utiliza tanto endógenas como exógenas para entrenar el modelo, sin embargo, nuestra data carece de estas variables.


Con esto finalmente obtenemos el pronostico de la cantidad aproximada de medicinas anti-diabeticas que podrian ser prescritas en Australia a partir de agosto hasta fines del 2008.

![Cantidades pronosticadas](https://github.com/VanderWest/Proyecto/blob/Reports/Imagenes/Meses%20pronosticados.PNG?raw=true) 
