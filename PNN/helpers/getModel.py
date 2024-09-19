from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf


def getModel(inputDim, nDense=2, nNodes=[12, 12]):

    model = Sequential()
    model.add(tf.keras.layers.Input(shape = (inputDim,))) 
    for i in range(nDense):
        model.add(Dense(units=nNodes[i],  kernel_initializer = tf.keras.initializers.glorot_normal( seed=1999)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
    model.add(Dense(units=1, kernel_initializer = tf.keras.initializers.glorot_normal( seed=1999)))
    model.add(tf.keras.layers.Activation('sigmoid'))
    return model