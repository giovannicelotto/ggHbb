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


def saveXYWrW(Xtrain, Xval, Ytrain, Yval, Wtrain, Wval, rWtrain, rWval, genmassTrain, genmassVal, inFolder,isTest=False, Xtest=None,Ytest=None,Wtest=None,genmassTest=None, suffix=""):

    
    Xtrain.to_parquet(inFolder +f"/Xtrain{suffix}.parquet")
    Xval.to_parquet(inFolder +f"/Xval{suffix}.parquet")
    
    np.save(inFolder +f"/Wtrain{suffix}.npy",   Wtrain)
    np.save(inFolder +f"/Wval{suffix}.npy",     Wval)

    np.save(inFolder +f"/rWtrain{suffix}.npy",   rWtrain)
    np.save(inFolder +f"/rWval{suffix}.npy",     rWval)
    
    np.save(inFolder +f"/Ytrain{suffix}.npy",    Ytrain)
    np.save(inFolder +f"/Yval{suffix}.npy",      Yval)

    np.save(inFolder +f"/genMassTrain{suffix}.npy",  genmassTrain)
    np.save(inFolder +f"/genMassVal{suffix}.npy",    genmassVal)

    if isTest:
        Xtest.to_parquet(inFolder +f"/Xtest{suffix}.parquet")
        np.save(inFolder +f"/Wtest{suffix}.npy",     Wtest)
        np.save(inFolder +f"/Ytest{suffix}.npy",     Ytest)
        np.save(inFolder +f"/genMassTest{suffix}.npy",   genmassTest)
    return 