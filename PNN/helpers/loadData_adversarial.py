from functions import loadMultiParquet
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from preprocessMultiClass import preprocessMultiClass


def loadData_adversarial(nReal, nMC, size, outFolder, columnsToRead, featuresForTraining, hp):

    nData, nHiggs = int(size), int(size)

    flatPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/"
    paths = [
            flatPathCommon + "/Data1A/training",
            flatPathCommon + "/GluGluHToBB/training"]
    
    massHypothesis = [50, 70, 100, 200, 300]
    for m in massHypothesis:
        paths.append(flatPathCommon + "/GluGluH_M%d_ToBB"%(m))
    
    

    dfs = loadMultiParquet(paths=paths, nReal=nReal, nMC=nMC, columns=columnsToRead, returnNumEventsTotal=False)

    # Uncomment in case you want discrete parameter for mass hypotheses
    dfs = preprocessMultiClass(dfs)
    
    # method v1
    #dfs[0]['massHypo'] = dfs[0]['dijet_mass'].apply(lambda x: massHypothesis[np.abs(massHypothesis - x).argmin()])
    #for idx, df in enumerate(dfs[1:]):
    #    dfs[idx+1]['massHypo'] = massHypothesis[idx]
    #    print("Process %d Mass %d"%(idx, massHypothesis[idx]))

    if 'massHypo' in featuresForTraining:
        massHypothesis = np.array([125]+massHypothesis)
        for idx, df in enumerate(dfs):
            dfs[idx]['massHypo'] = dfs[idx]['dijet_mass'].apply(lambda x: massHypothesis[np.abs(massHypothesis - x).argmin()])
    advFeature='jet1_btagDeepFlavB'


    
    # Each sample has the same number of elements (note in principle you could use 1/5 of Higgs data since you have 5 samples)
    for idx, df in enumerate(dfs):
        if idx==0:
            dfs[idx] = df.head(nData)
        else:
            dfs[idx] = df.head(nHiggs)
    for idx, df in enumerate(dfs):
        print("%d elements in df %d"%(len(df), idx))
    genMass = np.concatenate([np.zeros(len(dfs[0])),
               np.ones(len(dfs[1]))*massHypothesis[0],
               np.ones(len(dfs[2]))*massHypothesis[1],
               np.ones(len(dfs[3]))*massHypothesis[2],
               np.ones(len(dfs[4]))*massHypothesis[3],
               np.ones(len(dfs[5]))*massHypothesis[4],
               np.ones(len(dfs[6]))*massHypothesis[5]])
    # Create the labels for Background (0) and Signal (1)
    lenBkg  = len(dfs[0])
    lenS = 0
    for df in dfs[1:]:
        lenS = lenS + len(df)
    #lenS    = len(dfs[1]) + len(dfs[2]) + len(dfs[3]) + len(dfs[4])
    Y_0 = pd.DataFrame(np.zeros(lenBkg))
    Y_1 = pd.DataFrame(np.ones(lenS))

    # Each sample has sum = 1. In case it is not possible to have the same number of events
    Ws = [np.ones(lenBkg)]
    for df in dfs[1:]:
        Ws.append(df.sf)    
    for idx,W in enumerate(Ws):
        Ws[idx] = Ws[idx]/np.sum(Ws[idx])

    # For Signal create one unique array of weights in order to have a total weight of 1 for Signal and not nSamples*1
    W_H = np.concatenate(Ws[1:])
    W_H = W_H/np.sum(W_H)

    
    Y = np.concatenate((Y_0, Y_1))
    X = pd.concat(dfs)
    W = np.concatenate([Ws[0], W_H])
    
    X, Y, W = shuffle(X, Y, W, random_state=1999)

    Xtrain, Xtest, Ytrain, Ytest, advFeatureTrain, advFeatureTest, Wtrain, Wtest, genMassTrain, genMassTest = train_test_split(X, Y, X[advFeature], W, genMass, test_size=hp['test_split'], random_state=1999)
    Xtrain = Xtrain.drop([advFeature], axis=1)
    Xtest = Xtest.drop([advFeature], axis=1)

    assert len(Wtrain)==len(Xtrain)
    assert len(Wtest)==len(Xtest)
    Ytrain, Ytest, Wtrain, Wtest = Ytrain.reshape(-1), Ytest.reshape(-1), Wtrain.reshape(-1), Wtest.reshape(-1)
    return Xtrain, Xtest, Ytrain, Ytest, advFeatureTrain, advFeatureTest, Wtrain, Wtest, genMassTrain, genMassTest

