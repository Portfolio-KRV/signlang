# Signlang
Developed in the Neural Networks course by: Diego Quezada and Kevin Reyes.
## Objectives
- Recognize letters from images with sign language symbols using convolutional neural networks.
- Identify pairs of conflicting symbols when making a prediction.
- Compare the performance of different convolutional neural network architectures.
- Increase the amount of data by applying transformations on the original data set.
- Evaluate performance when adding Batch Normalization layers.

## Description
Evaluation of convolutional neural network models for sign language recognition, using batch normalization and data augmentation methods.

## Conclusions
- The application of data augmentation methods generated improvements in the performance of the convolutional network.
- The signs that are most confused are the letter N with S and C with O.
- The signs do not necessarily have to be similar so that the model has difficulty in differentiating them, for example, the L-R pair has a considerable percentage of error with respect to other pairs.
- The use of batch normalization did not generate improvements in the evaluation metric, but in numerical terms of the algorithm.

## Technology stack
- Tensorflow.
- Keras.
- Pandas.
- Numpy.
- Matplotlib.
- Seaborn.
- Scikit-learn.