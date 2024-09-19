import numpy as np
import matplotlib.pyplot as plt
def flattenWeights(Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, outName):
# **********
# TRAIN
# **********
    Wtrain_QCD = Wtrain[Ytrain==0]
    Wtrain_H = Wtrain[Ytrain==1]

    dfTrain_QCD = Xtrain[Ytrain==0]
    dfTrain_H = Xtrain[Ytrain==1]

    bins = np.linspace(40, 300, 51)
    countsQCDTrain = np.histogram(Xtrain.dijet_mass[Ytrain==0], bins=bins, weights=Wtrain_QCD)[0]
    countsHTrain = np.histogram(Xtrain.dijet_mass[Ytrain==1], bins=bins, weights=Wtrain_H)[0]

    rWtrain_QCD=Wtrain_QCD*(1/countsQCDTrain[np.digitize(np.clip(dfTrain_QCD.dijet_mass, bins[0], bins[-1]-0.0001), bins)-1])
    rWtrain_H=Wtrain_H*(1/countsHTrain[np.digitize(np.clip(dfTrain_H.dijet_mass, bins[0], bins[-1]-0.0001), bins)-1])

    rWtrain_QCD, rWtrain_H = rWtrain_QCD/np.sum(rWtrain_QCD), rWtrain_H/np.sum(rWtrain_H)

# **********
# TEST
# **********
    Wtest_QCD = Wtest[Ytest==0]
    Wtest_H = Wtest[Ytest==1]

    dfTest_QCD = Xtest[Ytest==0]
    dfTest_H = Xtest[Ytest==1]

    bins = np.linspace(40, 300, 51)
    countsQCDTrain = np.histogram(Xtest.dijet_mass[Ytest==0], bins=bins, weights=Wtest_QCD)[0]
    countsHTrain = np.histogram(Xtest.dijet_mass[Ytest==1], bins=bins, weights=Wtest_H)[0]

    rWtest_QCD=Wtest_QCD*(1/countsQCDTrain[np.digitize(np.clip(dfTest_QCD.dijet_mass, bins[0], bins[-1]-0.0001), bins)-1])
    rWtest_H=Wtest_H*(1/countsHTrain[np.digitize(np.clip(dfTest_H.dijet_mass, bins[0], bins[-1]-0.0001), bins)-1])

    rWtest_QCD, rWtest_H = rWtest_QCD/np.sum(rWtest_QCD), rWtest_H/np.sum(rWtest_H)


    fig, ax =plt.subplots(1, 1, figsize=(12,8), constrained_layout=True)
    ax.hist(dfTrain_QCD.dijet_mass, bins=bins, weights=Wtrain_QCD, label='Data', alpha=0.4, color='gray')
    ax.hist(dfTrain_H.dijet_mass, bins=bins, weights=Wtrain_H, alpha=0.4, label='H', linewidth=5, color='red')
    ax.hist(dfTrain_QCD.dijet_mass, bins=bins, weights=rWtrain_QCD, label='Data reweighted', histtype=u'step', linewidth=3, linestyle='dashed', color='blue')

    ax.hist(dfTrain_H.dijet_mass, bins=bins, weights=rWtrain_H, histtype=u'step', label='H reweighted', linewidth=5, linestyle='dotted', color='red')
    ax.set_yscale('log')
    ax.set_xlabel("Dijet Mass [GeV]")
    ax.set_ylabel("Normalized Counts")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlim(40, 300)
    fig.savefig(outName)


    rWtrain = Wtrain.copy()
    rWtest = Wtest.copy()
    rWtrain[Ytrain==0] = rWtrain_QCD
    rWtrain[Ytrain==1] = rWtrain_H
    rWtest[Ytest==0] = rWtest_QCD
    rWtest[Ytest==1] = rWtest_H
    return rWtrain, rWtest