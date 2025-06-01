# %%
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import numpy as np
from sklearn.model_selection import train_test_split

import sys
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.loadData_Sampling import loadData_sampling
from plotFeatures import plotNormalizedFeatures
# PNN helpers
from helpers.getFeatures import getFeatures, getFeaturesHighPt
from helpers.getParams import getParams
from helpers.loadData_adversarial import loadData_adversarial
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
    boosted = 1
    dataTaking = '1D'
# %%
#if boosted>=1 and sampling==0:
#    assert False
if boosted==0 and sampling==1:
    assert False
print("Sampling ", sampling)
print("Boosted ", boosted)

outFolder = "/t3home/gcelotto/ggHbb/PNN/input/data_sampling" if sampling else "/t3home/gcelotto/ggHbb/PNN/input/data"
outFolder = outFolder+"_pt%d_%s"%(boosted, dataTaking)
if not os.path.exists(outFolder):
    os.makedirs(outFolder)

# Define features to read and to train the pNN (+parameter massHypo) and save the features for training in outfolder
if (boosted>=1):
    featuresForTraining, columnsToRead = getFeaturesHighPt(outFolder)
else:
    assert False
    featuresForTraining, columnsToRead = getFeatures(outFolder, massHypo=False, bin_center=False, boosted=boosted, jet3_btagWP=True)

# %%
# load data for the samples and preprocess the data(pT cut)
# fill the massHypo column
# reweight each sample to have total weight 1, shuffle and split in train and test
#if sampling:
data = loadData_sampling(nReal=-1, nMC=-1,
                            columnsToRead=columnsToRead, featuresForTraining=featuresForTraining, test_split=0.3,
                            boosted=boosted, dataTaking=dataTaking, sampling=sampling)
#else:
#    data = loadData_adversarial(nReal=-1, nMC=-1, size=5e6, outFolder=outFolder,
#                            columnsToRead=columnsToRead, featuresForTraining=featuresForTraining, test_split=0.1,
#                            boosted=boosted, dataTaking=dataTaking)

#%%
#XtrainVal, Xtest, YtrainVal, Ytest, WtrainVal, Wtest, genMassTrainVal, genMassTest = data
#Xtrain, Xval, Ytrain, Yval, Wtrain, Wval, genMassTrain, genMassVal = train_test_split(XtrainVal, YtrainVal, WtrainVal, genMassTrainVal, test_size=0.2/0.99, random_state=1999)
Xtrain, Xval, Ytrain, Yval, Wtrain, Wval, genMassTrain, genMassVal = data




# Check for nan values and equal weights in Signal and Background
nan_values_train = Xtrain.isna().sum().sum()
nan_values_val = Xval.isna().sum().sum()
print(f"Nan values in train: {nan_values_train}")
print(f"Nan values in validation: {nan_values_val}")
print(Xtrain.isna().sum())
assert nan_values_train==0, "no nan values in train"
assert nan_values_val==0, "no nan values in val"

Xtrain['dimuon_mass'] = np.where(Xtrain['dimuon_mass']==-999, 0.106, Xtrain['dimuon_mass'])
Xval['dimuon_mass'] = np.where(Xval['dimuon_mass']==-999, 0.106, Xval['dimuon_mass'])
#Xtest['dimuon_mass'] = np.where(Xtest['dimuon_mass']==-999, 0.106, Xtest['dimuon_mass'])
# Here it holds:
# Wval[Yval==1].sum() + Wtrain[Ytrain==1].sum() + Wtest[Ytest==1].sum() = 1
# Wval[Yval==0].sum() + Wtrain[Ytrain==0].sum() + Wtest[Ytest==0].sum() = 1
import math
pos_sum = Wval[Yval==1].sum() + Wtrain[Ytrain==1].sum()
neg_sum = Wval[Yval==0].sum() + Wtrain[Ytrain==0].sum()

print(f"Signal class weight sum: {pos_sum}")
print(f"Background class weight sum: {neg_sum}")
assert math.isclose(Wval[Yval==1].sum() + Wtrain[Ytrain==1].sum(), 1, rel_tol=1e-9),    "Sum of weights for positive class is not close to 1"
assert math.isclose(Wval[Yval==0].sum() + Wtrain[Ytrain==0].sum(), 1, rel_tol=1e-9),   "Sum of weights for negative class is not close to 1"
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
ax.hist(Xtrain.dijet_mass[genMassTrain==125], bins=bins, histtype='step', density=False, linewidth=3)
ax.hist(Xtrain.dijet_mass[genMassTrain==50], bins=bins, histtype='step', density=False, linewidth=3)
ax.hist(Xtrain.dijet_mass[genMassTrain==70], bins=bins, histtype='step', density=False, linewidth=3)
ax.hist(Xtrain.dijet_mass[genMassTrain==100], bins=bins, histtype='step', density=False, linewidth=3)
ax.hist(Xtrain.dijet_mass[genMassTrain==200], bins=bins, histtype='step', density=False, linewidth=3)
ax.hist(Xtrain.dijet_mass[genMassTrain==300], bins=bins, histtype='step', density=False, linewidth=3)
ax.set_xlabel("Dijet Mass [GeV]")
ax.set_ylabel("Unweighted Counts")
fig.savefig(outFolder+"/dijetMass.png", bbox_inches='tight')




# %%


saveXYWrW(Xtrain=Xtrain, Xval=Xval, Ytrain=Ytrain, Yval=Yval, Wtrain=Wtrain, Wval=Wval,rWtrain=rWtrain, rWval=rWval, genmassTrain=genMassTrain, genmassVal=genMassVal, inFolder=outFolder, isTest=False)

# %%
gpuFlag=False
if gpuFlag==False:
    plotNormalizedFeatures(data=[Xtrain[Ytrain==0][featuresForTraining], Xtrain[Ytrain==1][featuresForTraining], Xval[Yval==0][featuresForTraining], Xval[Yval==1][featuresForTraining]],
                       outFile=outFolder+"/featuresForTraining.png", legendLabels=['Data Train', 'Higgs Train', 'Data Val', 'Higgs Val'],
                       colors=['blue', 'red', 'blue', 'red'], histtypes=[u'step', u'step', 'bar', 'bar'],
                       alphas=[1, 1, 0.4, 0.4], figsize=(20,40), autobins=False,
                       weights=[Wtrain[Ytrain==0], Wtrain[Ytrain==1], Wval[Yval==0], Wval[Yval==1]], error=False)
    plotNormalizedFeatures(data=[Xtrain[Ytrain==0], Xtrain[Ytrain==1], Xval[Yval==0], Xval[Yval==1]],
                       outFile=outFolder+"/featuresReweighted.png", legendLabels=['Data Train', 'Higgs Train', 'Data Val', 'Higgs Val'],
                       colors=['blue', 'red', 'blue', 'red'], histtypes=[u'step', u'step', 'bar', 'bar'],
                       alphas=[1, 1, 0.4, 0.4], figsize=(20,40), autobins=False,
                       weights=[rWtrain[Ytrain==0], rWtrain[Ytrain==1], rWval[Yval==0], rWval[Yval==1]], error=False)
    #plotNormalizedFeatures(data=[Xtrain[Ytrain==0], Xtrain[Ytrain==1], Xval[Yval==0], Xval[Yval==1]],
    #                   outFile=outFolder+"/features.png", legendLabels=['Data Train', 'Higgs Train', 'Data Val', 'Higgs Val'],
    #                   colors=['blue', 'red', 'blue', 'red'], histtypes=[u'step', u'step', 'bar', 'bar'],
    #                   alphas=[1, 1, 0.4, 0.4], figsize=(20,40), autobins=False,
    #                   weights=[Wtrain[Ytrain==0], Wtrain[Ytrain==1], Wval[Yval==0], Wval[Yval==1]], error=True)




# %%
# scale with standard scalers and apply log to any pt and mass distributions

# %%
Xtrain = scale(Xtrain, featuresForTraining,  scalerName= outFolder + "/myScaler.pkl" ,fit=True, boosted=boosted, scaler='robust')
Xval  = scale(Xval, featuresForTraining, scalerName= outFolder + "/myScaler.pkl" ,fit=False, boosted=boosted, scaler='robust')

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
