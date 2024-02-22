from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

def getLowPtModel(inputDim):
    model = Sequential()
    model.add(tf.keras.layers.Input(shape = inputDim)) 
    model.add(Dense(units=32,  kernel_initializer = tf.keras.initializers.glorot_normal( seed=1999)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Dense(units=16,  kernel_initializer = tf.keras.initializers.glorot_normal( seed=1999)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Dense(units=8,  kernel_initializer = tf.keras.initializers.glorot_normal( seed=1999)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Dense(units=1, kernel_initializer = tf.keras.initializers.glorot_normal( seed=1999)))
    model.add(tf.keras.layers.Activation('sigmoid'))
    return model


def getHighPtModel(inputDim):
    model = Sequential()
    model.add(tf.keras.layers.Input(shape = inputDim)) 
    #model.add(Dense(units=32,  kernel_initializer = tf.keras.initializers.glorot_normal( seed=1999)))
    #model.add(tf.keras.layers.BatchNormalization())
    #model.add(tf.keras.layers.Activation('relu'))
    model.add(Dense(units=16,  kernel_initializer = tf.keras.initializers.glorot_normal( seed=1999)))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    #model.add(Dense(units=8,  kernel_initializer = tf.keras.initializers.glorot_normal( seed=1999)))
    #model.add(tf.keras.layers.BatchNormalization())
    #model.add(tf.keras.layers.Activation('relu'))
    model.add(Dense(units=1, kernel_initializer = tf.keras.initializers.glorot_normal( seed=1999)))
    model.add(tf.keras.layers.Activation('sigmoid'))
    return model