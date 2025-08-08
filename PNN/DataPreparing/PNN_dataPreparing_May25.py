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
    parser.add_argument("-s", "--sampling", type=int, help="Enable sampling (default: False)", default=0)
    parser.add_argument("-b", "--boosted", type=int, default=3, help="Set boosted value (1 100-160) or 2 160-inf)")
    parser.add_argument("-dt", "--dataTaking", type=str, default='1D', help="1A or 1D")

    args = parser.parse_args()
    sampling = args.sampling
    boosted = args.boosted
    dataTaking = args.dataTaking
except:
    print("Error occurred")
    sampling = 0
    boosted = 0
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
    featuresForTraining, columnsToRead = getFeatures(outFolder)

# %%
# load data for the samples and preprocess the data(pT cut)
# fill the massHypo column
# reweight each sample to have total weight 1, shuffle and split in train and test
#if sampling:
data = loadData_sampling(nReal=100, nMC=-1,
                            columnsToRead=columnsToRead, featuresForTraining=featuresForTraining, test_split=0.2,
                            boosted=boosted, dataTaking=dataTaking, sampling=sampling, btagTight=False, mass_hypos=[50,70,100,200,300])
#else:
#    data = loadData_adversarial(nReal=-1, nMC=-1, size=5e6, outFolder=outFolder,
#                            columnsToRead=columnsToRead, featuresForTraining=featuresForTraining, test_split=0.1,
#                            boosted=boosted, dataTaking=dataTaking)
Xtrain, Xval, Ytrain, Yval, Wtrain, Wval, genMassTrain, genMassVal = data
# %%
fig, ax = plt.subplots(1, 1)
ax.hist(Xtrain[Ytrain==0].dijet_mass, bins=np.array([ 50.,61.47178677, 69.10711888, 75.73035104, 82.06945256
,88.4928578, 95.38378034,102.79587555,111.14177813,120.7728593
,132.47823225,147.52153342,168.38342721,202.85426985,300]), edgecolor='black')
ax.hist(Xtrain[Ytrain==1].dijet_mass, bins=np.array([ 50.,61.47178677, 69.10711888, 75.73035104, 82.06945256
,88.4928578, 95.38378034,102.79587555,111.14177813,120.7728593
,132.47823225,147.52153342,168.38342721,202.85426985,300]), histtype='step')
# Code for ParT
#tempdf = pd.read_csv("/t3home/gcelotto/ggHbb/outputs/counters/miniDf_process/miniDf_GluGluHToBBMINLO_tr.csv")
#tempdf2 = pd.read_csv("/t3home/gcelotto/ggHbb/outputs/counters/miniDf_process/miniDf_GluGluH_M125_ToBB_private.csv")
#mini = tempdf.genEventSumw2.sum() + tempdf2.genEventSumw2.sum()
#X = pd.concat([Xtrain, Xval])
#Y = np.concatenate([Ytrain, Yval])
#genMass = np.concatenate([genMassTrain, genMassVal])
#W = X.sf * X.PU_SF * X.btag_central * abs(X.genWeight) / mini
#X['Y'] = Y
#X['xsecWeight'] = W
#X['genMass'] = genMass
#lumi = 2000/5508 * 5.302 
#X.loc[genMass == 125, "xsecWeight"] = X.loc[genMass == 125, "xsecWeight"] * 28.61 * 1000 * lumi
#X.loc[genMass == 50, "xsecWeight"] = X.loc[genMass == 125, "xsecWeight"].sum()/len(X.loc[genMass == 50, "xsecWeight"])
#X.loc[genMass == 70, "xsecWeight"] = X.loc[genMass == 125, "xsecWeight"].sum()/len(X.loc[genMass == 70, "xsecWeight"])
#X.loc[genMass == 100, "xsecWeight"] = X.loc[genMass == 125, "xsecWeight"].sum()/len(X.loc[genMass == 100, "xsecWeight"])
#X.loc[genMass == 200, "xsecWeight"] = X.loc[genMass == 125, "xsecWeight"].sum()/len(X.loc[genMass == 200, "xsecWeight"])
#X.loc[genMass == 300, "xsecWeight"] = X.loc[genMass == 125, "xsecWeight"].sum()/len(X.loc[genMass == 300, "xsecWeight"])
#X.loc[genMass == 0, "xsecWeight"] = 1
#X.to_parquet("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/X_parT.parquet")
#%%



# Check for nan values and equal weights in Signal and Background
nan_values_train = Xtrain.isna().sum().sum()
nan_values_val = Xval.isna().sum().sum()
print(f"Nan values in train: {nan_values_train}")
print(f"Nan values in validation: {nan_values_val}")
print(Xtrain.isna().sum())
assert nan_values_train==0, "no nan values in train"
assert nan_values_val==0, "no nan values in val"

#Xtest['dimuon_mass'] = np.where(Xtest['dimuon_mass']==-999, 0.106, Xtest['dimuon_mass'])
# Here it holds:
# Wval[Yval==1].sum() + Wtrain[Ytrain==1].sum() + Wtest[Ytest==1].sum() = 1
# Wval[Yval==0].sum() + Wtrain[Ytrain==0].sum() + Wtest[Ytest==0].sum() = 1
import math
pos_sum = Wval[Yval==1].sum() + Wtrain[Ytrain==1].sum()
neg_sum = Wval[Yval==0].sum() + Wtrain[Ytrain==0].sum()

print(f"Signal class weight sum: {pos_sum}")
print(f"Background class weight sum: {neg_sum}")
assert math.isclose(Wval[Yval==1].sum() + Wtrain[Ytrain==1].sum(), 1, rel_tol=1e-4),    "Sum of weights for positive class is not close to 1 %.5f"%(Wval[Yval==1].sum() + Wtrain[Ytrain==1].sum())
assert math.isclose(Wval[Yval==0].sum() + Wtrain[Ytrain==0].sum(), 1, rel_tol=1e-9),   "Sum of weights for negative class is not close to 1"
# And also Wtrain[Ytrain==1].sum()/Wtrain[Ytrain==0].sum() for each train/val/test

# %%

# Higgs and Data have flat distribution in m_jj only for training and validation
rWtrain, rWval = flattenWeights(Xtrain, Xval, Ytrain, Yval, Wtrain, Wval, outFolder, outName=outFolder+ "/massReweighted.png",
                                xmin=int(Xtrain.dijet_mass.min()), nbins=201)

# To have typical numbers for lr and batch size set the mean weight to 1
rWtrain = rWtrain/np.mean(rWtrain)
rWval = rWval/np.mean(rWval)

Wtrain = Wtrain/np.mean(Wtrain)
Wval = Wval/np.mean(Wval)

# %%
fig, ax = plt.subplots(1, 1)
bins=np.linspace(int(Xtrain.dijet_mass.min()), 300, 401)
ax.hist(Xtrain.dijet_mass[genMassTrain==125], bins=bins, histtype='step', density=False, linewidth=3)
ax.hist(Xtrain.dijet_mass[genMassTrain==50], bins=bins, histtype='step', density=False, linewidth=3)
ax.hist(Xtrain.dijet_mass[genMassTrain==70], bins=bins, histtype='step', density=False, linewidth=3)
ax.hist(Xtrain.dijet_mass[genMassTrain==100], bins=bins, histtype='step', density=False, linewidth=3)
ax.hist(Xtrain.dijet_mass[genMassTrain==200], bins=bins, histtype='step', density=False, linewidth=3)
ax.hist(Xtrain.dijet_mass[genMassTrain==300], bins=bins, histtype='step', density=False, linewidth=3)
ax.set_xlabel("Dijet Mass [GeV]")
ax.set_ylabel("Unweighted Counts")
fig.savefig(outFolder+"/dijetMass.png", bbox_inches='tight')

fig, ax = plt.subplots(1, 1)
bins=np.linspace(int(Xtrain.dijet_mass.min()), 300, 401)
ax.hist(Xtrain.dijet_mass[genMassTrain==125], bins=bins, histtype='step', density=False, linewidth=3, weights=rWtrain[genMassTrain==125], label='M125')
ax.hist(Xtrain.dijet_mass[genMassTrain==50], bins=bins, histtype='step', density=False, linewidth=3, weights=rWtrain[genMassTrain==50], label='M50')
ax.hist(Xtrain.dijet_mass[genMassTrain==70], bins=bins, histtype='step', density=False, linewidth=3, weights=rWtrain[genMassTrain==70], label='M70')
ax.hist(Xtrain.dijet_mass[genMassTrain==100], bins=bins, histtype='step', density=False, linewidth=3, weights=rWtrain[genMassTrain==100], label='M100')
ax.hist(Xtrain.dijet_mass[genMassTrain==200], bins=bins, histtype='step', density=False, linewidth=3, weights=rWtrain[genMassTrain==200], label='M200')
ax.hist(Xtrain.dijet_mass[genMassTrain==300], bins=bins, histtype='step', density=False, linewidth=3, weights=rWtrain[genMassTrain==300], label='M300')

ax.hist(Xtrain.dijet_mass[genMassTrain>0], bins=bins, histtype='step', density=False, linewidth=3, weights=rWtrain[genMassTrain>0], label='Sum')
ax.legend()
ax.set_xlabel("Dijet Mass [GeV]")
ax.set_ylabel("Reweighted Counts")
fig.savefig(outFolder+"/dijetMass_reweighted.png", bbox_inches='tight')



fig, ax = plt.subplots(1, 1)
bins=np.linspace(int(Xval.dijet_mass.min()), 300, 101)
ax.hist(Xval.dijet_mass[genMassVal==125], bins=bins, histtype='step', density=False, linewidth=3, weights=rWval[genMassVal==125], label='M125')
ax.hist(Xval.dijet_mass[genMassVal==50], bins=bins, histtype='step', density=False, linewidth=3, weights=rWval[genMassVal==50], label='M50')
ax.hist(Xval.dijet_mass[genMassVal==70], bins=bins, histtype='step', density=False, linewidth=3, weights=rWval[genMassVal==70], label='M70')
ax.hist(Xval.dijet_mass[genMassVal==100], bins=bins, histtype='step', density=False, linewidth=3, weights=rWval[genMassVal==100], label='M100')
ax.hist(Xval.dijet_mass[genMassVal==200], bins=bins, histtype='step', density=False, linewidth=3, weights=rWval[genMassVal==200], label='M200')
ax.hist(Xval.dijet_mass[genMassVal==300], bins=bins, histtype='step', density=False, linewidth=3, weights=rWval[genMassVal==300], label='M300')

ax.hist(Xval.dijet_mass[genMassVal>0], bins=bins, histtype='step', density=False, linewidth=3, weights=rWval[genMassVal>0], label='Sum')
ax.legend()
ax.set_xlabel("Dijet Mass [GeV]")
ax.set_ylabel("Reweighted Counts")
fig.savefig(outFolder+"/dijetMassVal_reweighted.png", bbox_inches='tight')




# %%


saveXYWrW(Xtrain=Xtrain, Xval=Xval, Ytrain=Ytrain, Yval=Yval, Wtrain=Wtrain, Wval=Wval,rWtrain=rWtrain, rWval=rWval, genmassTrain=genMassTrain, genmassVal=genMassVal, inFolder=outFolder, isTest=False)
# %%
#part input
Xtrain['label']=Ytrain
Xval['label']=Yval
# %%

    #Without mass reweighiting
plotNormalizedFeatures(data=[Xtrain[Ytrain==0][featuresForTraining], Xtrain[Ytrain==1][featuresForTraining], Xval[Yval==0][featuresForTraining], Xval[Yval==1][featuresForTraining]],
                    outFile=outFolder+"/featuresForTraining.png", legendLabels=['Data Train', 'Higgs Train', 'Data Val', 'Higgs Val'],
                    colors=['blue', 'red', 'blue', 'red'], histtypes=[u'step', u'step', 'bar', 'bar'],
                    alphas=[1, 1, 0.4, 0.4], figsize=(20,40), autobins=False,
                    weights=[Wtrain[Ytrain==0], Wtrain[Ytrain==1], Wval[Yval==0], Wval[Yval==1]], error=False)
#With mass reweighiting
plotNormalizedFeatures(data=[Xtrain[Ytrain==0], Xtrain[Ytrain==1], Xval[Yval==0], Xval[Yval==1]],
                    outFile=outFolder+"/featuresReweighted.png", legendLabels=['Data Train', 'Higgs Train', 'Data Val', 'Higgs Val'],
                    colors=['blue', 'red', 'blue', 'red'], histtypes=[u'step', u'step', 'bar', 'bar'],
                    alphas=[1, 1, 0.4, 0.4], figsize=(20,40), autobins=False,
                    weights=[rWtrain[Ytrain==0], rWtrain[Ytrain==1], rWval[Yval==0], rWval[Yval==1]], error=False)
# With Mass Reweighiting per signal
plotNormalizedFeatures(data=[Xtrain[Ytrain==0], Xtrain[genMassTrain==125], Xval[Yval==0], Xval[genMassVal==125]],
                    outFile=outFolder+"/featuresReweighted_H125.png", legendLabels=['Data Train', 'H125 Train', 'Data Val', 'H125 Val'],
                    colors=['blue', 'red', 'blue', 'red'], histtypes=[u'step', u'step', 'bar', 'bar'],
                    alphas=[1, 1, 0.4, 0.4], figsize=(20,40), autobins=False,
                    weights=[rWtrain[Ytrain==0], rWtrain[genMassTrain==125], rWval[Yval==0], rWval[genMassVal==125]], error=False)
plotNormalizedFeatures(data=[Xtrain[Ytrain==0], Xtrain[genMassTrain==50], Xval[Yval==0], Xval[genMassVal==50]],
                    outFile=outFolder+"/featuresReweighted_H50.png", legendLabels=['Data Train', 'H50 Train', 'Data Val', 'H50 Val'],
                    colors=['blue', 'red', 'blue', 'red'], histtypes=[u'step', u'step', 'bar', 'bar'],
                    alphas=[1, 1, 0.4, 0.4], figsize=(20,40), autobins=False,
                    weights=[rWtrain[Ytrain==0], rWtrain[genMassTrain==50], rWval[Yval==0], rWval[genMassVal==50]], error=False)
plotNormalizedFeatures(data=[Xtrain[Ytrain==0], Xtrain[genMassTrain==70], Xval[Yval==0], Xval[genMassVal==70]],
                    outFile=outFolder+"/featuresReweighted_H70.png", legendLabels=['Data Train', 'H70 Train', 'Data Val', 'H70 Val'],
                    colors=['blue', 'red', 'blue', 'red'], histtypes=[u'step', u'step', 'bar', 'bar'],
                    alphas=[1, 1, 0.4, 0.4], figsize=(20,40), autobins=False,
                    weights=[rWtrain[Ytrain==0], rWtrain[genMassTrain==70], rWval[Yval==0], rWval[genMassVal==70]], error=False)
plotNormalizedFeatures(data=[Xtrain[Ytrain==0], Xtrain[genMassTrain==100], Xval[Yval==0], Xval[genMassVal==100]],
                    outFile=outFolder+"/featuresReweighted_H100.png", legendLabels=['Data Train', 'H100 Train', 'Data Val', 'H100 Val'],
                    colors=['blue', 'red', 'blue', 'red'], histtypes=[u'step', u'step', 'bar', 'bar'],
                    alphas=[1, 1, 0.4, 0.4], figsize=(20,40), autobins=False,
                    weights=[rWtrain[Ytrain==0], rWtrain[genMassTrain==100], rWval[Yval==0], rWval[genMassVal==100]], error=False)
plotNormalizedFeatures(data=[Xtrain[Ytrain==0], Xtrain[genMassTrain==200], Xval[Yval==0], Xval[genMassVal==200]],
                    outFile=outFolder+"/featuresReweighted_H200.png", legendLabels=['Data Train', 'H200 Train', 'Data Val', 'H200 Val'],
                    colors=['blue', 'red', 'blue', 'red'], histtypes=[u'step', u'step', 'bar', 'bar'],
                    alphas=[1, 1, 0.4, 0.4], figsize=(20,40), autobins=False,
                    weights=[rWtrain[Ytrain==0], rWtrain[genMassTrain==200], rWval[Yval==0], rWval[genMassVal==200]], error=False)

plotNormalizedFeatures(data=[Xtrain[Ytrain==0], Xtrain[genMassTrain==300], Xval[Yval==0], Xval[genMassVal==300]],
                       outFile=outFolder+"/featuresReweighted_H300.png", legendLabels=['Data Train', 'H300 Train', 'Data Val', 'H300 Val'],
                       colors=['blue', 'red', 'blue', 'red'], histtypes=[u'step', u'step', 'bar', 'bar'],
                       alphas=[1, 1, 0.4, 0.4], figsize=(20,40), autobins=False,
                       weights=[rWtrain[Ytrain==0], rWtrain[genMassTrain==300], rWval[Yval==0], rWval[genMassVal==300]], error=False)
    



# %%
# scale with standard scalers and apply log to any pt and mass distributions

# %%
Xtrain = scale(Xtrain, featuresForTraining,  scalerName= outFolder + "/myScaler.pkl" ,fit=True, boosted=boosted, scaler='robust')
Xval  = scale(Xval, featuresForTraining, scalerName= outFolder + "/myScaler.pkl" ,fit=False, boosted=boosted, scaler='robust')

# %%
test_gaussianity_validation(Xtrain, Xval, featuresForTraining, outFolder)
# %%

plotNormalizedFeatures(data=[Xtrain[Ytrain==0], Xtrain[Ytrain==1], Xval[Yval==0], Xval[Yval==1]],
                    outFile=outFolder+"/featuresReweighted_scaled.png", legendLabels=['Data Train', 'Higgs Train', 'Data Val', 'Higgs Val'],
                    colors=['blue', 'red', 'blue', 'red'], histtypes=[u'step', u'step', 'bar', 'bar'],
                    alphas=[1, 1, 0.4, 0.4], figsize=(20,40), autobins=True,
                    weights=[Wtrain[Ytrain==0], Wtrain[Ytrain==1], Wval[Yval==0], Wval[Yval==1]], error=True)
# %%
