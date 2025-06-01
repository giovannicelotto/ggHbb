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


def saveXYWP(Xtrain, Xval, Ytrain, Yval,  Wtrain, Wval, YPredTrain, YPredVal, inFolder, isTest=False, Xtest=None, Ytest=None, Wtest=None, YPredTest=None):
    
    np.save(inFolder +"/YPredTrain.npy", YPredTrain)
    np.save(inFolder +"/YPredVal.npy", YPredVal)
    
    Xtrain.to_parquet(inFolder +"/Xtrain.parquet")
    Xval.to_parquet(inFolder +"/Xval.parquet")
    
    np.save(inFolder +"/Wtrain.npy", Wtrain)
    np.save(inFolder +"/Wval.npy", Wval)
    
    np.save(inFolder +"/Ytrain.npy", Ytrain)
    np.save(inFolder +"/Yval.npy", Yval)

    if isTest:
        np.save(inFolder +"/YPredTest.npy", YPredTest)
        Xtest.to_parquet(inFolder +"/Xtest.parquet")
        np.save(inFolder +"/Wtest.npy", Wtest)
        np.save(inFolder +"/Ytest.npy", Ytest)
    return 


def saveXYWrW(Xtrain, Xval, Ytrain, Yval, Wtrain, Wval, rWtrain, rWval, genmassTrain, genmassVal, inFolder,isTest=False, Xtest=None,Ytest=None,Wtest=None,genmassTest=None):

    
    Xtrain.to_parquet(inFolder +"/Xtrain.parquet")
    Xval.to_parquet(inFolder +"/Xval.parquet")
    
    np.save(inFolder +"/Wtrain.npy",   Wtrain)
    np.save(inFolder +"/Wval.npy",     Wval)

    np.save(inFolder +"/rWtrain.npy",   rWtrain)
    np.save(inFolder +"/rWval.npy",     rWval)
    
    np.save(inFolder +"/Ytrain.npy",    Ytrain)
    np.save(inFolder +"/Yval.npy",      Yval)

    np.save(inFolder +"/genMassTrain.npy",  genmassTrain)
    np.save(inFolder +"/genMassVal.npy",    genmassVal)

    if isTest:
        Xtest.to_parquet(inFolder +"/Xtest.parquet")
        np.save(inFolder +"/Wtest.npy",     Wtest)
        np.save(inFolder +"/Ytest.npy",     Ytest)
        np.save(inFolder +"/genMassTest.npy",   genmassTest)
    return 