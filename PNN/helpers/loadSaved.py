import numpy as np
import pandas as pd
def loadSaved(inFolder):
    Xtrain      = pd.read_parquet(inFolder + "/XTrain.parquet")
    Xtest       = pd.read_parquet(inFolder + "/XTest.parquet")
    Ytrain      = np.load(inFolder + "/YTrain.npy")
    Ytest       = np.load(inFolder + "/YTest.npy")
    Wtrain      = np.load(inFolder + "/WTrain.npy")
    Wtest       = np.load(inFolder + "/WTest.npy")
    YPredTrain  = np.load(inFolder + "/YPredTrain.npy")    
    YPredTest   = np.load(inFolder + "/YPredTest.npy")    

    return Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, YPredTrain, YPredTest