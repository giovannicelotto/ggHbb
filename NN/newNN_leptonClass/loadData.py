import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
from functions import loadMultiParquet
import sys
sys.path.append("/t3home/gcelotto/ggHbb/NN")
from helpersForNN import preprocessMultiClass
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def loadData(featuresForTraining, columnsToRead, leptonClass, nReal, nMC, pTmin, pTmax, suffix, outFolder, hp, logging):
    logging.info("Features for training")
    for feature in featuresForTraining:
        logging.info(">  %s"%feature)
    paths = [
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/"]
    paths.append("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-100to200")
    paths.append("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-200to400")
    paths.append("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-400to600")
    paths.append("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-600to800")
    paths.append("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-800toInf")

    dfs, numEventsList = loadMultiParquet(paths=paths, nReal=nReal, nMC=nMC, columns=columnsToRead, returnNumEventsTotal=True)
        
    dfs = preprocessMultiClass(dfs, leptonClass,  pTmin, pTmax, suffix)

    
    nData, nHiggs, nZ = int(1e5), int(1e5) , int(1e5)
    for idx, df in enumerate(dfs):
        if idx==0:
            dfs[idx] = df.head(nData)
        elif idx==1:
            dfs[idx] = df.head(nHiggs)
        else:
            #pass
            dfs[idx] = df.head(int(nZ/(len(paths)-2)))

    for idx, df in enumerate(dfs):
        logging.info("Length of df %d : %d\n"%(idx, len(df)))

    Y_0 = pd.DataFrame([np.ones(len(dfs[0])),  np.zeros(len(dfs[0])), np.zeros(len(dfs[0]))]).T
    Y_1 = pd.DataFrame([np.zeros(len(dfs[1])), np.ones(len(dfs[1])),  np.zeros(len(dfs[1]))]).T
    #ZJets
    Y_Z, W_Z = [],[]
    for i in range(2, len(paths)):
        Y = pd.DataFrame([np.zeros(len(dfs[i])), np.zeros(len(dfs[i])), np.ones(len(dfs[i]))]).T
        Y_Z.append(Y)

    # define a weights vector 1 for data 1 for hbb, xsection for the Z boson dataframes. then concat z bosons. divide every weights by the average of the weights
    W_0 = dfs[0].sf
    W_1 = dfs[1].sf
    #ZJets
    for i in range(2, len(paths)):
        W = ([5.261e+03, 1012, 114.2, 25.34, 12.99][i-2])/numEventsList[i]*dfs[i].sf
        W_Z.append(W)
    W_ZBos = pd.concat(W_Z)
    Y_ZBos = pd.concat(Y_Z)
    df_ZBos = pd.concat(dfs[2:])

# Each sample has a weight equal to 1 no matter how many events are there
    rW_0 = W_0.copy()
    rW_1 = W_1.copy()
    rW_ZBos = W_ZBos.copy()
    
    W_0 = W_0/W_0.sum()
    W_1 = W_1/W_1.sum()
    W_ZBos = W_ZBos/W_ZBos.sum()

    bins_0 = np.linspace(40, 200, 40)
    bins_1 = np.linspace(40, 200, 40)
    bins_2 = np.linspace(40, 200, 40)
    counts_0 = np.histogram(dfs[0].dijet_mass,  bins=bins_0, weights=rW_0)[0]
    counts_H = np.histogram(dfs[1].dijet_mass,  bins=bins_1, weights=rW_1)[0]
    counts_Z = np.histogram(df_ZBos.dijet_mass, bins=bins_2, weights=rW_ZBos)[0]
    

    rW_ZBos=rW_ZBos*(1/counts_Z[np.digitize(np.clip(df_ZBos.dijet_mass, bins_2[0], bins_2[-1]-0.0001), bins_2)-1])
    rW_0=rW_0*(1/counts_0[np.digitize(np.clip(dfs[0].dijet_mass, bins_0[0], bins_0[-1]-0.0001), bins_0)-1])
    rW_1=rW_1*(1/counts_H[np.digitize(np.clip(dfs[1].dijet_mass, bins_1[0], bins_1[-1]-0.0001), bins_1)-1])

    rW_0 = rW_0/rW_0.sum()
    rW_1 = rW_1/rW_1.sum()
    rW_ZBos = rW_ZBos/rW_ZBos.sum()
    plt.close('all')
    
    fig, ax =plt.subplots(1, 1, figsize=(12,8), constrained_layout=True)
    ax.hist(dfs[0].dijet_mass, bins=bins_0, weights=W_0, label='Data', alpha=0.4, color='gray')
    ax.hist(df_ZBos.dijet_mass, bins=bins_0, weights=W_ZBos, alpha=0.4, label='Z', linewidth=5, color='green')
    ax.hist(dfs[1].dijet_mass, bins=bins_0, weights=W_1, alpha=0.4, label='H', linewidth=5, color='red')
    ax.hist(dfs[0].dijet_mass, bins=bins_0, weights=rW_0, label='Data reweighted', histtype=u'step', linewidth=3, linestyle='dashed', color='blue')
    ax.hist(df_ZBos.dijet_mass, bins=bins_0, weights=rW_ZBos, histtype=u'step', label='Z reweighted', linewidth=5, color='green', linestyle='dashed')
    ax.hist(dfs[1].dijet_mass, bins=bins_0, weights=rW_1, histtype=u'step', label='H reweighted', linewidth=5, linestyle='dotted', color='red')
    ax.set_yscale('log')
    ax.set_xlabel("Dijet Mass [GeV]")
    ax.set_ylabel("Normalized Counts")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlim(40, 200)
    fig.savefig(outFolder+ "/performance/mass_temp.png")

    

    Y_ZBos, W_ZBos, rW_ZBos, df_ZBos = shuffle(Y_ZBos, W_ZBos, rW_ZBos, df_ZBos)
    Y_ZBos.head(nZ), W_ZBos.head(nZ), rW_ZBos.head(nZ), df_ZBos.head(nZ)
    Y = np.concatenate((Y_0, Y_1, Y_ZBos))
    X = pd.concat((dfs[0], dfs[1], df_ZBos))
    rW = np.concatenate((rW_0, rW_1, rW_ZBos))
    W = np.concatenate((W_0, W_1, W_ZBos))
    
    X, Y, W, rW = shuffle(X, Y, W, rW, random_state=1999)
    Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, rWtrain, rWtest = train_test_split(X, Y, W, rW, test_size=hp['test_split'], random_state=1999)
    #Xtrain[Ytrain[:,0]==1] = Xtrain[(Ytrain[:,0]==1) & (Xtrain['dijet_mass']<80) | (Xtrain['dijet_mass']>100) | (Ytrain[:,1]==1) | (Ytrain[:,2]==1)]
    assert len(Wtrain)==len(Xtrain)
    assert len(Wtest)==len(Xtest)
    print(rW_0.sum(), rW_1.sum(), rW_ZBos.sum())
    return Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, rWtrain, rWtest
