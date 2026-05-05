# Reconocimiento de Lenguaje de Señas

Evaluación de modelos de redes neuronales convolucionales para reconocimiento de lenguaje de señas usando batch normalization y data augmentation.

## Objetivos

- Reconocer letras desde imágenes con símbolos de lenguaje de señas usando CNNs.
- Identificar pares de símbolos conflictivos al hacer predicciones.
- Comparar el rendimiento de diferentes arquitecturas CNN con y sin Batch Normalization.

## Tecnologías

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

## Hallazgos Clave

- La CNN alcanzó más del 99% de precisión en el conjunto de prueba para clasificación de 24 clases.
- Los pares de señas más confundidos son N-S y C-O, a pesar de no ser visualmente similares.
- Data augmentation (rotación, zoom, desplazamiento) mejoró la generalización en 3-5%.
- Batch normalization mejoró la estabilidad numérica pero no aumentó la precisión.

## Curso

Redes Neuronales

## Co-Autores

- Diego Quezada

## Demo

Este proyecto tiene una demo disponible en el sitio web del portfolio.

## Repositorio

[https://github.com/Portfolio-KRV/signlang](https://github.com/Portfolio-KRV/signlang)
