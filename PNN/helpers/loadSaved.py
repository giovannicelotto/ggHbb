import numpy as np
import pandas as pd
def loadSaved(inFolder, rWeights=False):
    Xtrain      = pd.read_parquet(inFolder + "/XTrain.parquet")
    Xtest       = pd.read_parquet(inFolder + "/XTest.parquet")
    Ytrain      = np.load(inFolder + "/YTrain.npy")
    Ytest       = np.load(inFolder + "/YTest.npy")
    Wtrain      = np.load(inFolder + "/WTrain.npy")
    Wtest       = np.load(inFolder + "/WTest.npy")
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