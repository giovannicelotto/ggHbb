import numpy as np
import matplotlib.pyplot as plt
def flattenWeights(Xtrain, Xval, Ytrain, Yval, Wtrain, Wval, inFolder, outName, xmin=40, xmax=300, nbins=51):
# **********
# TRAIN
# **********
    Wtrain_QCD = Wtrain[Ytrain==0]
    Wtrain_H = Wtrain[Ytrain==1]

    dfTrain_QCD = Xtrain[Ytrain==0]
    dfTrain_H = Xtrain[Ytrain==1]

    bins = np.linspace(xmin, xmax, nbins)
    countsQCDTrain = np.histogram(Xtrain.dijet_mass[Ytrain==0], bins=bins, weights=Wtrain_QCD)[0]
    countsHTrain = np.histogram(Xtrain.dijet_mass[Ytrain==1], bins=bins, weights=Wtrain_H)[0]

    rWtrain_QCD=Wtrain_QCD*(1/countsQCDTrain[np.digitize(np.clip(dfTrain_QCD.dijet_mass, bins[0], bins[-1]-0.0001), bins)-1])
    rWtrain_H=Wtrain_H*(1/countsHTrain[np.digitize(np.clip(dfTrain_H.dijet_mass, bins[0], bins[-1]-0.0001), bins)-1])

    rWtrain_QCD, rWtrain_H = rWtrain_QCD/np.sum(rWtrain_QCD), rWtrain_H/np.sum(rWtrain_H)

# **********
# TEST
# **********
    Wval_QCD = Wval[Yval==0]
    Wval_H = Wval[Yval==1]

    dfTest_QCD = Xval[Yval==0]
    dfTest_H = Xval[Yval==1]

    bins = np.linspace(xmin, xmax, nbins)
    countsQCDTrain = np.histogram(Xval.dijet_mass[Yval==0], bins=bins, weights=Wval_QCD)[0]
    countsHTrain = np.histogram(Xval.dijet_mass[Yval==1], bins=bins, weights=Wval_H)[0]

    rWval_QCD=Wval_QCD*(1/countsQCDTrain[np.digitize(np.clip(dfTest_QCD.dijet_mass, bins[0], bins[-1]-0.0001), bins)-1])
    rWval_H=Wval_H*(1/countsHTrain[np.digitize(np.clip(dfTest_H.dijet_mass, bins[0], bins[-1]-0.0001), bins)-1])

    rWval_QCD, rWval_H = rWval_QCD/np.sum(rWval_QCD), rWval_H/np.sum(rWval_H)


    fig, ax =plt.subplots(1, 1, figsize=(12,8), constrained_layout=True)
    ax.hist(dfTrain_QCD.dijet_mass, bins=bins, weights=Wtrain_QCD, label='Data', alpha=0.4, color='gray')
    ax.hist(dfTrain_H.dijet_mass, bins=bins, weights=Wtrain_H, alpha=0.4, label='H', linewidth=5, color='red')
    
    #ax.hist(dfTrain_QCD.dijet_mass, bins=bins, weights=rWtrain_QCD, label='Data reweighted', histtype=u'step', linewidth=3, linestyle='dashed', color='blue')
    #ax.hist(dfTrain_H.dijet_mass, bins=bins, weights=rWtrain_H, histtype=u'step', label='H reweighted', linewidth=5, linestyle='dotted', color='red')
    #ax.set_yscale('log')
    ax.set_xlabel("Dijet Mass [GeV]")
    ax.set_ylabel("Normalized Counts")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlim(xmin, xmax)
    fig.savefig(outName)


    rWtrain = Wtrain.copy()
    rWval = Wval.copy()
    rWtrain[Ytrain==0] = rWtrain_QCD
    rWtrain[Ytrain==1] = rWtrain_H
    rWval[Yval==0] = rWval_QCD
    rWval[Yval==1] = rWval_H
    #np.save(inFolder + "/rWTrain.npy", rWtrain)
    #np.save(inFolder + "/rWTest.npy",  rWval)
    return rWtrain, rWval