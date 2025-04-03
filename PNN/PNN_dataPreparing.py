# %%
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import numpy as np
from sklearn.model_selection import train_test_split

import sys
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
sys.path.append("/t3home/gcelotto/ggHbb/PNN/helpers")
from plotFeatures import plotNormalizedFeatures
# PNN helpers
from helpers.getFeatures import getFeatures, getFeaturesHighPt
from helpers.getParams import getParams
from helpers.loadData_adversarial import loadData_adversarial, loadData_sampling
from helpers.getInfolderOutfolder import getInfolderOutfolder
from helpers.scaleUnscale import scale, unscale
from helpers.dcorLoss import *
from helpers.saveDataAndPredictions import saveXYWrW
from helpers.flattenWeights import flattenWeights
import argparse
import pandas as pd
from helpers.scaleUnscale import test_gaussianity_validation

# %%

# Define folder of input and output. Create the folders if not existing
hp = getParams()
try:
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument("-s", "--sampling", type=int, help="Enable sampling (default: False)", default=1)
    parser.add_argument("-b", "--boosted", type=int, default=2, help="Set boosted value (1 100-160) or 2 160-inf)")
    parser.add_argument("-dt", "--dataTaking", type=str, default='1A', help="1A or 1D")

    args = parser.parse_args()
    sampling = args.sampling
    boosted = args.boosted
    dataTaking = args.dataTaking
except:
    print("Error occurred")
    sampling = 0
    boosted = 0
    dataTaking = '1A'
# %%
if boosted>=1 and sampling==0:
    assert False
if boosted==0 and sampling==1:
    assert False
print("Sampling ", sampling)
print("Boosted ", boosted)

outFolder = "/t3home/gcelotto/ggHbb/PNN/input/data_sampling" if sampling else "/t3home/gcelotto/ggHbb/PNN/input/data"
outFolder = outFolder+"_pt%d_%s"%(boosted, dataTaking)
if not os.path.exists(outFolder):
    os.makedirs(outFolder)

# Define features to read and to train the pNN (+parameter massHypo) and save the features for training in outfolder
if boosted:
    featuresForTraining, columnsToRead = getFeaturesHighPt(outFolder, massHypo=False)
else:
    featuresForTraining, columnsToRead = getFeatures(outFolder, massHypo=False, bin_center=False, simple=True)

# %%
# load data for the samples and preprocess the data(pT cut)
# fill the massHypo column
# reweight each sample to have total weight 1, shuffle and split in train and test
if sampling:
    data = loadData_sampling(nReal=-1, nMC=-1, size=5e6, outFolder=outFolder,
                            columnsToRead=columnsToRead, featuresForTraining=featuresForTraining, test_split=hp["test_split"],
                            boosted=boosted)
else:
    data = loadData_adversarial(nReal=-1, nMC=-1, size=5e6, outFolder=outFolder,
                            columnsToRead=columnsToRead, featuresForTraining=featuresForTraining, test_split=hp["test_split"],
                            boosted=boosted, dataTaking=dataTaking)
#%%
XtrainVal, Xtest, YtrainVal, Ytest, WtrainVal, Wtest, genMassTrainVal, genMassTest = data
Xtrain, Xval, Ytrain, Yval, Wtrain, Wval, genMassTrain, genMassVal = train_test_split(XtrainVal, YtrainVal, WtrainVal, genMassTrainVal, test_size=hp['val_split'], random_state=1999)

# Here it holds:
# Wval[Yval==1].sum() + Wtrain[Ytrain==1].sum() + Wtest[Ytest==1].sum() = 1
# Wval[Yval==0].sum() + Wtrain[Ytrain==0].sum() + Wtest[Ytest==0].sum() = 1
# And also Wtrain[Ytrain==1].sum()/Wtrain[Ytrain==0].sum() for each train/val/test

# %%

# Higgs and Data have flat distribution in m_jj only for training and validation
rWtrain, rWval = flattenWeights(Xtrain, Xval, Ytrain, Yval, Wtrain, Wval, outFolder, outName=outFolder+ "/massReweighted.png",xmin=int(Xtrain.dijet_mass.min()))

# To have typical numbers for lr and batch size set the mean weight to 1
rWtrain = rWtrain/np.mean(rWtrain)
rWval = rWval/np.mean(rWval)

Wtrain = Wtrain/np.mean(Wtrain)
Wval = Wval/np.mean(Wval)

# %%
fig, ax = plt.subplots(1, 1)
bins=np.linspace(int(Xtrain.dijet_mass.min()), 300, 101)
ax.hist(Xtrain.dijet_mass[genMassTrain==125], bins=bins, histtype='step', density=False)
ax.hist(Xtrain.dijet_mass[genMassTrain==50], bins=bins, histtype='step', density=False)
ax.hist(Xtrain.dijet_mass[genMassTrain==70], bins=bins, histtype='step', density=False)
ax.hist(Xtrain.dijet_mass[genMassTrain==100], bins=bins, histtype='step', density=False)
ax.hist(Xtrain.dijet_mass[genMassTrain==200], bins=bins, histtype='step', density=False)
ax.hist(Xtrain.dijet_mass[genMassTrain==300], bins=bins, histtype='step', density=False)
ax.set_xlabel("Dijet Mass [GeV]")
ax.set_ylabel("Unweighted Counts")
fig.savefig(outFolder+"/dijetMass.png", bbox_inches='tight')
# %%


gpuFlag=False
if gpuFlag==False:
    plotNormalizedFeatures(data=[Xtrain[Ytrain==0], Xtrain[Ytrain==1], Xval[Yval==0], Xval[Yval==1]],
                       outFile=outFolder+"/featuresReweighted.png", legendLabels=['Data Train', 'Higgs Train', 'Data Val', 'Higgs Val'],
                       colors=['blue', 'red', 'blue', 'red'], histtypes=[u'step', u'step', 'bar', 'bar'],
                       alphas=[1, 1, 0.4, 0.4], figsize=(20,40), autobins=False,
                       weights=[rWtrain[Ytrain==0], rWtrain[Ytrain==1], rWval[Yval==0], rWval[Yval==1]], error=True)
    plotNormalizedFeatures(data=[Xtrain[Ytrain==0], Xtrain[Ytrain==1], Xval[Yval==0], Xval[Yval==1]],
                       outFile=outFolder+"/features.png", legendLabels=['Data Train', 'Higgs Train', 'Data Val', 'Higgs Val'],
                       colors=['blue', 'red', 'blue', 'red'], histtypes=[u'step', u'step', 'bar', 'bar'],
                       alphas=[1, 1, 0.4, 0.4], figsize=(20,40), autobins=False,
                       weights=[Wtrain[Ytrain==0], Wtrain[Ytrain==1], Wval[Yval==0], Wval[Yval==1]], error=True)
# %%

saveXYWrW(Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, Wtrain, Wval, Wtest,rWtrain, rWval, genMassTrain, genMassVal, genMassTest, inFolder=outFolder)

# %%
# scale with standard scalers and apply log to any pt and mass distributions
Xtrain = scale(Xtrain,featuresForTraining,  scalerName= outFolder + "/myScaler.pkl" ,fit=True, boosted=boosted, scaler='standard')
Xval  = scale(Xval, featuresForTraining, scalerName= outFolder + "/myScaler.pkl" ,fit=False, boosted=boosted, scaler='standard')

# %%
test_gaussianity_validation(Xtrain, Xval, featuresForTraining, outFolder)
# %%
if gpuFlag==False:
    plotNormalizedFeatures(data=[Xtrain[Ytrain==0], Xtrain[Ytrain==1], Xval[Yval==0], Xval[Yval==1]],
                       outFile=outFolder+"/featuresReweighted_scaled.png", legendLabels=['Data Train', 'Higgs Train', 'Data Val', 'Higgs Val'],
                       colors=['blue', 'red', 'blue', 'red'], histtypes=[u'step', u'step', 'bar', 'bar'],
                       alphas=[1, 1, 0.4, 0.4], figsize=(20,40), autobins=True,
                       weights=[Wtrain[Ytrain==0], Wtrain[Ytrain==1], Wval[Yval==0], Wval[Yval==1]], error=True)
# %%
