import numpy as np
import tensorflow as tf
from getFeatures import getFeatures
from getModel import getModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from doPlots import doPlotLoss

def PNNClassifier(Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, rWtrain, rWtest, featuresForTraining, hp, inFolder, outFolder):
    
    #featuresForTraining, columnsToRead = getFeatures(inFolder)
    
    modelName = "myModel.h5"

# get the model, optimizer, compile it and fit to data
    model = getModel(inputDim=len(featuresForTraining), nDense=hp['nDense'], nNodes=hp['nNodes'])
    optimizer = Adam(learning_rate = hp['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adam") 
    model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(), weighted_metrics=['accuracy'])
    callbacks = []
    earlyStop = EarlyStopping(monitor = 'val_loss', patience = hp['patienceES'], verbose = 1, restore_best_weights=True)
    callbacks.append(earlyStop)
    fit = model.fit(Xtrain[featuresForTraining], Ytrain, 
                    sample_weight = rWtrain,
                    verbose = 2,
                    epochs=hp['epochs'], validation_split=hp['validation_split'],
                    callbacks=callbacks, shuffle=True)
    
    model.save(outFolder +"/model/"+modelName)
    model = load_model(outFolder +"/model/"+modelName)
    doPlotLoss(fit=fit, outName=outFolder +"/performance/loss.png", earlyStop=earlyStop, patience=hp['patienceES'])

    YPredTest = model.predict(Xtest[featuresForTraining])
    YPredTrain = model.predict(Xtrain[featuresForTraining])


    return YPredTrain, YPredTest, model, featuresForTraining