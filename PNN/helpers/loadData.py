from functions import loadMultiParquet
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from preprocessMultiClass import preprocessMultiClass


def loadData(nReal, nMC, outFolder, columnsToRead, featuresForTraining, hp):
    flatPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB"
    paths = [
            flatPathCommon + "/Data1A/training",
            flatPathCommon + "/GluGluHToBB/training"]
    
    massHypothesis = [70, 100, 300]
    for m in massHypothesis:
        paths.append(flatPathCommon + "/GluGluH_M%d_ToBB"%(m))
    
    

    dfs = loadMultiParquet(paths=paths, nReal=nReal, nMC=nMC, columns=columnsToRead, returnNumEventsTotal=False)
    dfs = preprocessMultiClass(dfs)

    dfs[0]['massHypo'] = np.random.choice(massHypothesis+[125], size=len(dfs[0]))
    dfs[1]['massHypo'] = 125
    dfs[2]['massHypo'] = 70
    dfs[2]['massHypo'] = 100
    dfs[2]['massHypo'] = 300

    nData, nHiggs = int(1e3), int(1e3)
    for idx, df in enumerate(dfs):
        if idx==0:
            dfs[idx] = df.head(nData)
        else:
            dfs[idx] = df.head(nHiggs)

    lenBkg  = len(dfs[0])
    lenS    = len(dfs[1]) + len(dfs[2]) + len(dfs[3]) + len(dfs[4])
    Y_0 = pd.DataFrame([np.ones(lenBkg),  np.zeros(lenBkg)]).T
    Y_1 = pd.DataFrame([np.zeros(lenS), np.ones(lenS)]).T


    # define a weights vector 1 for data 1 for hbb, xsection for the Z boson dataframes. then concat z bosons. divide every weights by the average of the weights
    W_0 = np.ones(lenBkg)
    W_1 = np.concatenate([dfs[1].sf, dfs[2].sf, dfs[3].sf, dfs[4].sf])

# Each sample has a weight equal to 1 no matter how many events are there
  
    W_0 = W_0/W_0.sum()
    W_1 = W_1/W_1.sum()


    
    Y = np.concatenate((Y_0, Y_1))
    X = pd.concat(dfs)
    W = np.concatenate((W_0, W_1))
    
    X, Y, W = shuffle(X, Y, W, random_state=1999)
    Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest = train_test_split(X, Y, W, test_size=hp['test_split'], random_state=1999)

    assert len(Wtrain)==len(Xtrain)
    assert len(Wtest)==len(Xtest)

    return Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest

