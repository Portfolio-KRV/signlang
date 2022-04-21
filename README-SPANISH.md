# Signlang
Desarrollado en el curso de Redes Neuronales por: Diego Quezada y Kevin Reyes.
## Objetivos
- Reconocer letras a partir de imágenes con simbologías del lenguaje de señas utilizando redes neuronales convolucionales.
- Identificar pares de simbolos conflictivos al momento de realizar una predicción.
- Comparar el rendimiento de distintas arquitecturas de redes neuronales convoluciones.
- Aumentar la cantidad de datos aplicando transformaciones sobre el conjunto de datos original.
- Evaluar el rendimiento al añadir capas de Batch Normalization.

## Descripción
Evaluación de modelos de redes neuronales convolucionales para el reconocimiento del lenguaje de señas, utilizando batch normalization y métodos de aumentación de datos.

## Conclusiones
- La aplicación de métodos de aumentación de datos generó mejoras en el rendimiento de la red convolucional.
- Las señas que más se confunden son la letra N con S y C con O.
- Las señas no necesariamente deben ser similares para que el modelo tenga dificultad en diferenciarlas, por ejemplo, el par L-R tiene un porcentaje considerable de error respecto de otros pares.
- El uso de batch normalization no generó mejoras en la métrica de evaluación, pero si en término numéricos del algoritmo.

## Stack de tecnologías 
- Tensorflow.
- Keras.
- Pandas.
- Numpy.
- Matplotlib.
- Seaborn.
- Scikit-learn.