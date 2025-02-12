import numpy as np
import pandas as pd
def save(Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, YPredTrain, YPredTest, inFolder):
    np.save(inFolder +"/YPredTest.npy", YPredTest)
    np.save(inFolder +"/YPredTrain.npy", YPredTrain)
    Xtrain.to_parquet(inFolder +"/XTrain.parquet")
    Xtest.to_parquet(inFolder +"/XTest.parquet")
    np.save(inFolder +"/WTest.npy", Wtest)
    np.save(inFolder +"/WTrain.npy", Wtrain)
    np.save(inFolder +"/YTest.npy", Ytest)
    np.save(inFolder +"/YTrain.npy", Ytrain)
    return 


def saveXYWP(Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, Wtrain, Wval, Wtest, YPredTrain, YPredVal, YPredTest, inFolder):
    
    np.save(inFolder +"/YPredTrain.npy", YPredTrain)
    np.save(inFolder +"/YPredVal.npy", YPredVal)
    np.save(inFolder +"/YPredTest.npy", YPredTest)
    
    Xtrain.to_parquet(inFolder +"/Xtrain.parquet")
    Xval.to_parquet(inFolder +"/Xval.parquet")
    Xtest.to_parquet(inFolder +"/Xtest.parquet")
    
    np.save(inFolder +"/Wtrain.npy", Wtrain)
    np.save(inFolder +"/Wval.npy", Wval)
    np.save(inFolder +"/Wtest.npy", Wtest)
    
    np.save(inFolder +"/Ytrain.npy", Ytrain)
    np.save(inFolder +"/Yval.npy", Yval)
    np.save(inFolder +"/Ytest.npy", Ytest)
    return 


def saveXYWrW(Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, Wtrain, Wval, Wtest, rWtrain, rWval, genmassTrain, genmassVal, genmassTest, inFolder):

    
    Xtrain.to_parquet(inFolder +"/Xtrain.parquet")
    Xval.to_parquet(inFolder +"/Xval.parquet")
    Xtest.to_parquet(inFolder +"/Xtest.parquet")
    
    np.save(inFolder +"/Wtrain.npy",   Wtrain)
    np.save(inFolder +"/Wval.npy",     Wval)
    np.save(inFolder +"/Wtest.npy",     Wtest)

    np.save(inFolder +"/rWtrain.npy",   rWtrain)
    np.save(inFolder +"/rWval.npy",     rWval)
    
    np.save(inFolder +"/Ytrain.npy",    Ytrain)
    np.save(inFolder +"/Yval.npy",      Yval)
    np.save(inFolder +"/Ytest.npy",     Ytest)

    np.save(inFolder +"/genMassTrain.npy",  genmassTrain)
    np.save(inFolder +"/genMassVal.npy",    genmassVal)
    np.save(inFolder +"/genMassTest.npy",   genmassTest)
    return 