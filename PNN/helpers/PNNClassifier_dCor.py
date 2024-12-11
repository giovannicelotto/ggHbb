# %%
import numpy as np
import tensorflow as tf
from getFeatures import getFeatures
from getModel import getModel
from doPlots import doPlotLoss
import torch
import torch.nn as nn
import torch.optim as optim
# %%




# Custom Loss Function
def custom_loss(y_true, y_pred, dijet_mass, normedweight, lambda_corr=1.0):
    # Compute binary cross-entropy loss
    classifier_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))
    
    # Compute distance correlation loss
    dCorr_loss = distance_corr(y_pred, dijet_mass, normedweight)
    
    # Combine the losses with the lambda_corr regularization term
    total_loss = classifier_loss + lambda_corr * dCorr_loss
    return total_loss

# Example of how to integrate the custom loss into the model training:

def PNNClassifier_dCor(Xtrain, Xtest, Ytrain, Ytest, advFeatureTrain, advFeatrueTest, Wtrain, Wtest, rWtrain, rWtest, featuresForTraining, hp, inFolder, outFolder):
    
    # Load or define the model (getModel function)
    modelName = "myModel.h5"
    model = getModel(inputDim=len(featuresForTraining), nDense=hp['nDense'], nNodes=hp['nNodes'])
    
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    
    # Compile the model with custom loss function
    model.compile(optimizer=optimizer, 
                  loss=lambda y_true, y_pred, advFeatureTrain: custom_loss(y_true, y_pred, dijet_mass=advFeatureTrain, normedweight=rWtrain, lambda_corr=0.1),
                  weighted_metrics=['accuracy'])

    # Early stopping callback
    callbacks = []
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=hp['patienceES'], verbose=1, restore_best_weights=True)
    callbacks.append(earlyStop)
    
    # Fit the model
    fit = model.fit(Xtrain[featuresForTraining], Ytrain, 
                    sample_weight=rWtrain,
                    verbose=2,
                    epochs=hp['epochs'], validation_split=hp['validation_split'],
                    batch_size=hp['batch_size'],
                    callbacks=callbacks, shuffle=True)
    
    # Save model
    model.save(outFolder + "/model/" + modelName)
    model = tf.keras.models.load_model(outFolder + "/model/" + modelName)
    
    # Plot Loss
    doPlotLoss(fit=fit, outName=outFolder + "/performance/loss.png", earlyStop=earlyStop, patience=hp['patienceES'])
    
    # Make predictions
    YPredTest = model.predict(Xtest[featuresForTraining])
    YPredTrain = model.predict(Xtrain[featuresForTraining])
    
    return YPredTrain, YPredTest, model, featuresForTraining
