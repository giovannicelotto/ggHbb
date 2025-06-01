import numpy as np
import pandas as pd
def loadSaved(inFolder, rWeights=False):
    Xtrain      = pd.read_parquet(inFolder + "/Xtrain.parquet")
    Xtest       = pd.read_parquet(inFolder + "/Xtest.parquet")
    Ytrain      = np.load(inFolder + "/Ytrain.npy")
    Ytest       = np.load(inFolder + "/Ytest.npy")
    Wtrain      = np.load(inFolder + "/Wtrain.npy")
    Wtest       = np.load(inFolder + "/Wtest.npy")
    YPredTrain  = np.load(inFolder + "/YPredTrain.npy")    
    YPredTest   = np.load(inFolder + "/YPredTest.npy")    

    if rWeights:
        rWtrain  = np.load(inFolder + "/rWTrain.npy")
        rWtest   = np.load(inFolder + "/rWTest.npy")
        return Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, YPredTrain, YPredTest, rWtrain, rWtest
    else:
        return Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, YPredTrain, YPredTest
    
def loadXYWSaved(inFolder):
    Xtrain      = pd.read_parquet(inFolder + "/XTrain.parquet")
    Xtest       = pd.read_parquet(inFolder + "/XTest.parquet")
    Ytrain      = np.load(inFolder + "/YTrain.npy")
    Ytest       = np.load(inFolder + "/YTest.npy")
    Wtrain      = np.load(inFolder + "/WTrain.npy")
    Wtest       = np.load(inFolder + "/WTest.npy")

    return Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest

def loadXYWrWSaved(inFolder, isTest=False):
    Xtrain      = pd.read_parquet(inFolder + "/Xtrain.parquet")
    Xval        = pd.read_parquet(inFolder + "/Xval.parquet")
    Ytrain      = np.load(inFolder + "/Ytrain.npy")
    Yval        = np.load(inFolder + "/Yval.npy")
    Wtrain      = np.load(inFolder + "/Wtrain.npy")
    Wval        = np.load(inFolder + "/Wval.npy")

    rWtrain      = np.load(inFolder + "/rWtrain.npy")
    rWval        = np.load(inFolder + "/rWval.npy")

    genMassTrain      = np.load(inFolder + "/genMassTrain.npy")
    genMassVal        = np.load(inFolder + "/genMassVal.npy")

    if isTest:
        Xtest       = pd.read_parquet(inFolder + "/Xtest.parquet")
        Ytest       = np.load(inFolder + "/Ytest.npy")
        Wtest       = np.load(inFolder + "/Wtest.npy")
        genMassTest       = np.load(inFolder + "/genMassTest.npy")
        return Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, Wtrain, Wval, Wtest, rWtrain, rWval, genMassTrain, genMassVal, genMassTest
    else:

        return Xtrain, Xval, Ytrain, Yval, Wtrain, Wval, rWtrain, rWval, genMassTrain, genMassVal