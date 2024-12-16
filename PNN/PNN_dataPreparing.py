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
from helpers.getFeatures import getFeatures
from helpers.getParams import getParams
from helpers.loadData_adversarial import loadData_adversarial
from helpers.getInfolderOutfolder import getInfolderOutfolder
from helpers.scaleUnscale import scale, unscale
from helpers.dcorLoss import *
from helpers.saveDataAndPredictions import saveXYW
from helpers.flattenWeights import flattenWeights



# %%

# Define folder of input and output. Create the folders if not existing
hp = getParams()
inFolder, outFolder = "/t3home/gcelotto/ggHbb/PNN/input/data", "/t3home/gcelotto/ggHbb/PNN/input/data"

# Define features to read and to train the pNN (+parameter massHypo) and save the features for training in outfolder
featuresForTraining, columnsToRead = getFeatures(outFolder, massHypo=True)


# load data for the samples and preprocess the data(pT cut)
# fill the massHypo column
# reweight each sample to have total weight 1, shuffle and split in train and test
data = loadData_adversarial(nReal=49, nMC=-1, size=1e6, outFolder=outFolder,
                            columnsToRead=columnsToRead, featuresForTraining=featuresForTraining, test_split=hp["test_split"])
XtrainVal, Xtest, YtrainVal, Ytest, advFeatureTrainVal, advFeatureTest, WtrainVal, Wtest, genMassTrainVal, genMassTest = data
Xtrain, Xval, Ytrain, Yval, advFeatureTrain, advFeatureVal, Wtrain, Wval, genMassTrain, genMassVal = train_test_split(XtrainVal, YtrainVal, advFeatureTrainVal, WtrainVal, genMassTrainVal, test_size=hp['val_split'], random_state=1999)

# %%

# Higgs and Data have flat distribution in m_jj only for training and validation
rWtrain, rWval = flattenWeights(Xtrain, Xval, Ytrain, Yval, Wtrain, Wval, inFolder, outName=outFolder+ "/massReweighted.png")
# To have typical numbers for lr and batch size set the mean weight to 1
rWtrain = rWtrain/np.mean(rWtrain)
rWval = rWval/np.mean(rWval)

Wtrain = Wtrain/np.mean(Wtrain)
Wval = Wval/np.mean(Wval)
Wtest = Wtest/np.mean(Wtest)
# %%
gpuFlag=False
if gpuFlag==False:
    plotNormalizedFeatures(data=[Xtrain[Ytrain==0], Xtrain[Ytrain==1], Xval[Yval==0], Xval[Yval==1]],
                       outFile=outFolder+"/featuresReweighted.png", legendLabels=['Data Train', 'Higgs Train', 'Data Val', 'Higgs Val'],
                       colors=['blue', 'red', 'blue', 'red'], histtypes=[u'step', u'step', 'bar', 'bar'],
                       alphas=[1, 1, 0.4, 0.4], figsize=(20,40), autobins=False,
                       weights=[rWtrain[Ytrain==0], rWtrain[Ytrain==1], rWval[Yval==0], rWval[Yval==1]], error=True)
# %%
# scale with standard scalers and apply log to any pt and mass distributions
import pandas as pd
saveXYW(Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, rWtrain, rWval, Wtest, genMassTrain, genMassVal, genMassTest, inFolder="/t3home/gcelotto/ggHbb/PNN/input/data")
np.save(inFolder+"/advFeatureTest.npy", advFeatureTest)
np.save(inFolder+"/advFeatureTrain.npy", advFeatureTrain)
np.save(inFolder+"/advFeatureVal.npy", advFeatureVal)

np.save(inFolder+"/Wtest.npy", Wtest)
np.save(inFolder+"/Wtrain.npy", Wtrain)
np.save(inFolder+"/Wval.npy", Wval)
# %%
Xtrain = scale(Xtrain,featuresForTraining,  scalerName= outFolder + "/myScaler.pkl" ,fit=True)
Xval  = scale(Xval, featuresForTraining, scalerName= outFolder + "/myScaler.pkl" ,fit=False)
advFeatureTrain = scale(pd.DataFrame(advFeatureTrain),['jet1_btagDeepFlavB'],  scalerName= outFolder + "/myScaler_adv.pkl" ,fit=True)
advFeatureVal  = scale(pd.DataFrame(advFeatureVal), ['jet1_btagDeepFlavB'], scalerName= outFolder + "/myScaler_adv.pkl" ,fit=False)
if gpuFlag==False:
    plotNormalizedFeatures(data=[Xtrain[Ytrain==0], Xtrain[Ytrain==1], Xval[Yval==0], Xval[Yval==1]],
                       outFile=outFolder+"/featuresReweighted_scaled.png", legendLabels=['Data Train', 'Higgs Train', 'Data Val', 'Higgs Val'],
                       colors=['blue', 'red', 'blue', 'red'], histtypes=[u'step', u'step', 'bar', 'bar'],
                       alphas=[1, 1, 0.4, 0.4], figsize=(20,40), autobins=True,
                       weights=[Wtrain[Ytrain==0], Wtrain[Ytrain==1], Wval[Yval==0], Wval[Yval==1]], error=True)

# %%
