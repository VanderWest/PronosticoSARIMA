# Proyecto

Se busca pronosticar a través de series de tiempo el número de prescripciones de medicamentos anti-diabetes para los próximos 6 meses en Australia utilizando una muestra sobre el numero de prescripciones realizadas entre los años 1991 y 2008, Para ello se propone el uso del modelo de machine learning SARIMA, el cual es un modelo de pronósticos en base a la integración de coeficientes y errores tomando en consideración datos que tienen una tendencia estacional, a diferencia de ARIMA que omite la estacionalidad para su entrenamiento.
Veremos las razones para elegir SARIMA en lugar de otros modelos y sus respectivas métricas para medir la calidad del modelo.

Los datos para este desafío fueron obtenidos de https://www.key2stats.com/data-set/view/745 lo cuales corresponden al número de medicamentos anti-diabetes prescritos en Australia entre los años 1991 y 2008.

Data ya procesada:

![Data limpia](https://github.com/VanderWest/Proyecto/blob/Reports/Imagenes/DF.PNG?raw=true)

Partamos por asumir que no sabemos que modelo utilizar, realizamos un análisis exploratorio sobre la data para ver sí existen patrones que puedan guiarnos, lo único que sabemos es que estamos trabajando con series de tiempo y queremos pronosticar (de esta forma se reduce en número de modelos a utilizar en la bolsa). Graficando los datos obtenemos una idea sobre que está ocurriendo:


![Gráfica](https://github.com/VanderWest/Proyecto/blob/Reports/Imagenes/Gr%C3%A1fica%20prescripciones%20por%20mes.png?raw=true)

Con está sola gráfica se pueden observar 3 patrones: Primero la tendencia creciente que tienen los datos conforme pasa los años. Segundo, la estacionalidad de los datos, creando patrones repetitivos en los años. Tercero, una suerte de ruido blanco, mostrando lo errático que se van volviendo los valores de los datos sin perderse en la aleatoriedad.

Dado lo anterior podemos realizar un desglose de la gráfica utilizando el paquete STL (seasonal-trend decomposition) para ver en detalle que es lo que ocurre:

![STL](https://github.com/VanderWest/Proyecto/blob/Reports/Imagenes/STL.png?raw=true)

- En la primera grafica observamos lo anterior, el numero de prescripciones entre los años 1991 y 2008.
- La segunda, tendencia, nos muestra el comportamiento creciente del número de prescripciones recetadas al paso de los años.
- La tercera, estacionalidad, muestra el patrón cíclico del número de prescripciones recetadas entre los años 1991 y 2008.
- La cuarta, residuos, denota las diferencias que hay entre los valores pronosticados, juntando la tendencia y la estacionalidad, contrastado con los valores reales presentados en la data.

