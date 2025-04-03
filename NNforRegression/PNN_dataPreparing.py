# %%
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import numpy as np
from sklearn.model_selection import train_test_split

import sys
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
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
import os
from helpers.scaleUnscale import test_gaussianity_validation


# %%
outFolder = "/t3home/gcelotto/ggHbb/NNforRegression/input" 

if not os.path.exists(outFolder):
    os.makedirs(outFolder)

# Define features to read and to train the NN
featuresForTraining = [
    "jet1_pt",    "jet1_eta",    "jet1_phi",    "jet1_mass", #"jet1_btagDeepFlavB",
    "jet1_btagTight", "jet1_dR_dijet","jet1_nMuons",
    "jet2_pt",    "jet2_eta",    "jet2_phi",    "jet2_mass", #"jet2_btagDeepFlavB",
    "jet2_btagTight", "jet2_dR_dijet","jet2_nMuons",
    "jet3_pt",    "jet3_eta",    "jet3_phi",    "jet3_mass", "jet3_dR_dijet","jet3_nMuons",
    "muon_pt",    "muon_eta",    "muon_dxySig",
    "dijet_pt",    "dijet_mass",
    "ht",    "nJets",
    #"genJetNu_pt_1",    "genJetNu_pt_2",    "genDijetNu_mass",    "target"
]
# %%
# load data for the samples and preprocess the data(pT cut)
# fill the massHypo column
# reweight each sample to have total weight 1, shuffle and split in train and test

from helpers.loadData_adversarial import uniform_sample
path =    "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/genMatched"
from functions import getDfProcesses_v2, getCommonFilters, cut
import glob
dfProcesses = getDfProcesses_v2()[0]
processes = dfProcesses.process.iloc[[43,38,39,40,41, 42]]
masses = [125, 50, 70, 100, 200, 300]
paths = [path+"/"+process for process in processes]
dfs=[]
for m, path in zip(masses,paths):
    fileNames = glob.glob(path+"/*.parquet")
    df = pd.read_parquet(fileNames, filters=getCommonFilters())
    df['genMass']=m
    dfs.append(df)
dfs = cut(dfs, 'dijet_mass', 45, 300)
dfs[1]=dfs[1][dfs[1].genDijetNu_mass<70]
dfs[2]=dfs[2][dfs[2].genDijetNu_mass<100]
dfs[4]=dfs[4][dfs[4].genDijetNu_mass>100]
dfs[5]=dfs[5][dfs[5].genDijetNu_mass>150]
# %%
lenghts = [len(df) for df in dfs]
print(lenghts)
minLenght = np.min(lenghts)
for idx, df in enumerate(dfs):
    dfs[idx]=df.iloc[:minLenght]
for df in dfs:
    print(len(df))

# %%
df = pd.concat(dfs)
df_sampled = uniform_sample(df, column='dijet_mass', num_bins=20)
# %%

Xtrain, Xtest, Ytrain, Ytest, genMassTrain, genMassTest = train_test_split(df_sampled, df_sampled['target'], df_sampled['genMass'], test_size=0.2, random_state=1999)
Xtrain, Xval, Ytrain, Yval, genMassTrain, genMassVal = train_test_split(Xtrain, Ytrain, genMassTrain, test_size=0.2, random_state=1999)

# %%
fig, ax = plt.subplots(1, 1)
bins=np.linspace(int(Xtrain.dijet_mass.min()), 300, 101)
ax.hist(Xtrain.dijet_mass[genMassTrain==125], bins=bins, histtype='step', density=False, label="M 125")
ax.hist(Xtrain.dijet_mass[genMassTrain==50], bins=bins, histtype='step', density=False, label="M 50")
ax.hist(Xtrain.dijet_mass[genMassTrain==70], bins=bins, histtype='step', density=False, label="M 70")
ax.hist(Xtrain.dijet_mass[genMassTrain==100], bins=bins, histtype='step', density=False, label="M 100")
ax.hist(Xtrain.dijet_mass[genMassTrain==200], bins=bins, histtype='step', density=False, label="M 200")
ax.hist(Xtrain.dijet_mass[genMassTrain==300], bins=bins, histtype='step', density=False, label="M 300")
ax.legend()
ax.set_xlabel("Dijet Mass [GeV]")
ax.set_ylabel("Unweighted Counts")
fig.savefig(outFolder+"/dijetMass.png", bbox_inches='tight')

# %%
Xtrain.to_parquet(outFolder +"/Xtrain.parquet")
Xval.to_parquet(outFolder +"/Xval.parquet")
Xtest.to_parquet(outFolder +"/Xtest.parquet")

np.save(outFolder +"/Ytrain.npy",    Ytrain)
np.save(outFolder +"/Yval.npy",      Yval)
np.save(outFolder +"/Ytest.npy",     Ytest)

np.save(outFolder +"/genMassTrain.npy",  genMassTrain)
np.save(outFolder +"/genMassVal.npy",    genMassVal)
np.save(outFolder +"/genMassTest.npy",   genMassTest)

# %%
