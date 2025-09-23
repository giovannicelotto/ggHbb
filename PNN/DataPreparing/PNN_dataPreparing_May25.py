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

parser = argparse.ArgumentParser(description="Process some arguments.")
parser.add_argument("-s", "--sampling", type=int, help="Enable sampling (default: False)", default=0)
parser.add_argument("-b", "--boosted", type=int, default=3, help="Set boosted value (1 100-160) or 2 160-inf)")
parser.add_argument("-dt", "--dataTaking", type=str, default='1D', help="1A or 1D")

if hasattr(sys, 'ps1') or not sys.argv[1:]:
    # Interactive mode (REPL, Jupyter) OR no args provided → use defaults
    args = parser.parse_args([])
else:
    # Normal CLI usage
    args = parser.parse_args()
# %%
#if boosted>=1 and sampling==0:
#    assert False
if args.boosted==0 and args.sampling==1:
    assert False
print("Sampling ", args.sampling)
print("Boosted ", args.boosted)

outFolder = "/t3home/gcelotto/ggHbb/PNN/input/data_sampling" if args.sampling else "/t3home/gcelotto/ggHbb/PNN/input/data"
outFolder = outFolder+"_pt%d_%s"%(args.boosted, args.dataTaking)
if not os.path.exists(outFolder):
    os.makedirs(outFolder)

# Define features to read and to train the pNN (+parameter massHypo) and save the features for training in outfolder
if args.boosted==4:
    print("Here")
    featuresForTraining, columnsToRead = getFeatures(outFolder)
elif (args.boosted>=1):
    featuresForTraining, columnsToRead = getFeaturesHighPt(outFolder)

if args.boosted==4:
    mass_hypos = []
else:
    mass_hypos = [50,70,100,200,300]

# %%
# load data for the samples and preprocess the data(pT cut)
# fill the massHypo column
# reweight each sample to have total weight 1, shuffle and split in train and test
#if sampling:
data = loadData_sampling(nReal=-1, nMC=-1,
                            columnsToRead=columnsToRead, featuresForTraining=featuresForTraining, test_split=0.2,
                            boosted=args.boosted, dataTaking=args.dataTaking, sampling=args.sampling, btagTight=False, mass_hypos=mass_hypos)
#else:
#    data = loadData_adversarial(nReal=-1, nMC=-1, size=5e6, outFolder=outFolder,
#                            columnsToRead=columnsToRead, featuresForTraining=featuresForTraining, test_split=0.1,
#                            boosted=boosted, dataTaking=dataTaking)
Xtrain, Xval, Ytrain, Yval, Wtrain, Wval, genMassTrain, genMassVal = data

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

if args.boosted==4:
    rWtrain = Wtrain.copy()
    rWval = Wval.copy()
    for y in [0, 1]:  # loop over class (0=background, 1=signal)
        # total weight for this class
        total_train = Wtrain[Ytrain == y].sum()
        total_val   = Wval[Yval == y].sum()
        
        for j1 in [0, 1]:
            for j2 in [0, 1]:
                # mask for this quadrant
                mask_train = (Ytrain == y) & (Xtrain.jet1_btagTight == j1) & (Xtrain.jet2_btagTight == j2)
                mask_val   = (Yval == y)   & (Xval.jet1_btagTight == j1)   & (Xval.jet2_btagTight == j2)
                
                # sum of original weights in this quadrant
                sum_train = Wtrain[mask_train].sum()
                sum_val   = Wval[mask_val].sum()
                
                # target weight for this quadrant = 1/4 of total class weight
                target_train = 0.25 * total_train
                target_val   = 0.25 * total_val
                
                # rescale factor (only if quadrant has events)
                if sum_train > 0:
                    rWtrain[mask_train] = Wtrain[mask_train] * (target_train / sum_train)
                if sum_val > 0:
                    rWval[mask_val] = Wval[mask_val] * (target_val / sum_val)


    rWtrain = rWtrain/np.mean(rWtrain)
    rWval = rWval/np.mean(rWval)
else:
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
for m in (np.unique(genMassTrain)[np.unique(genMassTrain)!=0]):
    ax.hist(Xtrain.dijet_mass[genMassTrain==m], bins=bins, histtype='step', density=False, linewidth=3)
ax.set_xlabel("Dijet Mass [GeV]")
ax.set_ylabel("Unweighted Counts")
fig.savefig(outFolder+"/dijetMass.png", bbox_inches='tight')
# %%
fig, ax = plt.subplots(1, 1)
bins=np.linspace(int(Xtrain.dijet_mass.min()), 300, 401)
for m in (np.unique(genMassTrain)[np.unique(genMassTrain)!=0]):
    ax.hist(Xtrain.dijet_mass[genMassTrain==m], bins=bins, histtype='step', density=False, linewidth=3, label="M%d"%m)
ax.hist(Xtrain.dijet_mass[genMassTrain>0], bins=bins, histtype='step', density=False, linewidth=3, weights=rWtrain[genMassTrain>0], label='Sum')
ax.legend()
ax.set_xlabel("Dijet Mass [GeV]")
ax.set_ylabel("Reweighted Counts")
fig.savefig(outFolder+"/dijetMass_reweighted.png", bbox_inches='tight')
# %%


fig, ax = plt.subplots(1, 1)
bins=np.linspace(int(Xval.dijet_mass.min()), 300, 101)
for m in (np.unique(genMassTrain)[np.unique(genMassTrain)!=0]):
    ax.hist(Xtrain.dijet_mass[genMassTrain==m], bins=bins, histtype='step', density=False, linewidth=3, label="M%d"%m)
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
for m in (np.unique(genMassTrain)):
    if m==0:
        continue
    plotNormalizedFeatures(data=[Xtrain[Ytrain==0], Xtrain[genMassTrain==m], Xval[Yval==0], Xval[genMassVal==m]],
                    outFile=outFolder+"/featuresReweighted_H%d.png"%m, legendLabels=['Data Train', 'H%d Train'%m, 'Data Val', 'H%d Val'%m],
                    colors=['blue', 'red', 'blue', 'red'], histtypes=[u'step', u'step', 'bar', 'bar'],
                    alphas=[1, 1, 0.4, 0.4], figsize=(20,40), autobins=False,
                    weights=[rWtrain[Ytrain==0], rWtrain[genMassTrain==m], rWval[Yval==0], rWval[genMassVal==m]], error=False)




# %%
# scale with standard scalers and apply log to any pt and mass distributions

# %%
Xtrain = scale(Xtrain, featuresForTraining,  scalerName= outFolder + "/myScaler.pkl" ,fit=True, boosted=args.boosted, scaler='robust')
Xval  = scale(Xval, featuresForTraining, scalerName= outFolder + "/myScaler.pkl" ,fit=False, boosted=args.boosted, scaler='robust')

# %%
test_gaussianity_validation(Xtrain, Xval, featuresForTraining, outFolder)
# %%

plotNormalizedFeatures(data=[Xtrain[Ytrain==0], Xtrain[Ytrain==1], Xval[Yval==0], Xval[Yval==1]],
                    outFile=outFolder+"/featuresReweighted_scaled.png", legendLabels=['Data Train', 'Higgs Train', 'Data Val', 'Higgs Val'],
                    colors=['blue', 'red', 'blue', 'red'], histtypes=[u'step', u'step', 'bar', 'bar'],
                    alphas=[1, 1, 0.4, 0.4], figsize=(20,40), autobins=True,
                    weights=[Wtrain[Ytrain==0], Wtrain[Ytrain==1], Wval[Yval==0], Wval[Yval==1]], error=True)
# %%
