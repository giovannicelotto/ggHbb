from helpers.getModel import getModel
from helpers.getParams import getParams
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from scipy.integrate import simpson
import tensorflow as tf
import numpy as np


def aucModelEvaluator(featuresForTraining, Xtrain, Xtest, Ytrain, Ytest, rWtrain, lr, bs):
    bs = int(2**(int(bs)))
    hp = getParams()
    model = getModel(inputDim=len(featuresForTraining), nDense=hp['nDense'], nNodes=hp['nNodes'])
    optimizer = Adam(learning_rate = lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adam") 
    model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(), weighted_metrics=['accuracy'])
    callbacks = []
    earlyStop = EarlyStopping(monitor = 'val_loss', patience = hp['patienceES'], verbose = 1, restore_best_weights=True)
    callbacks.append(earlyStop)
    fit = model.fit(Xtrain[featuresForTraining], Ytrain, 
                    sample_weight = rWtrain,
                    batch_size=bs,
                    verbose = 0,
                    epochs=hp['epochs'], validation_split=hp['validation_split'],
                    callbacks=callbacks, shuffle=True)
    YPredTest = model.predict(Xtest[featuresForTraining])
    
    
    # auc
    thresholds = np.linspace(0, 1, 501)
    tpr, fpr = [], []
    signal_predictions = YPredTest[Ytest==1]
    realData_predictions = YPredTest[Ytest==0]

    for t in thresholds:
        tpr.append(np.sum(signal_predictions > t)/len(signal_predictions))
        fpr.append(np.sum(realData_predictions > t)/len(realData_predictions))
    tpr, fpr =np.array(tpr), np.array(fpr)
    # auc
    auc = -simpson(tpr, fpr)
    print("AUC : ", auc)
    return auc