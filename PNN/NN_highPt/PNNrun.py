# %%
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import sys
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
from plotFeatures import plotNormalizedFeatures
sys.path.append("/t3home/gcelotto/ggHbb/PNN/helpers")

from helpers.getFeatures import getFeatures
from helpers.getParams import getParams
from helpers.loadData import loadData
from helpers.getInfolderOutfolder import getInfolderOutfolder
from helpers.scaleUnscale import scale, unscale
from helpers.PNNClassifier import PNNClassifier
from helpers.saveDataAndPredictions import save
from helpers.doPlots import runPlots
from helpers.flattenWeights import flattenWeights
# %%

# Define folder of input and output. Create the folders if not existing
inFolder, outFolder = getInfolderOutfolder(name = "dec10")

# Define features to read and to train the pNN (+parameter massHypo) and save the features for training in outfolder
featuresForTraining, columnsToRead = getFeatures(outFolder)
# %%
# define the parameters for the nn
hp = getParams()

# number of files from real data and mc
nReal, nMC = 10, -1

# load data for the samples and preprocess the data(pT cut)
# fill the massHypo column
# cut the data to have same length in all the samples
# reweight each sample to have total weight 1, shuffle and split in train and test
data = loadData(nReal, nMC, outFolder, columnsToRead, featuresForTraining, hp)
Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest = data
featuresForTraining = featuresForTraining + ['massHypo']

# %%
# Higgs and Data have flat distribution in m_jj
rWtrain, rWtest = flattenWeights(Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, inFolder, outName=outFolder+ "/performance/massReweighted.png")
#rWtrain, rWtest = Wtrain.copy(), Wtest.copy()

plotNormalizedFeatures(data=[Xtrain[Ytrain==0], Xtrain[Ytrain==1], Xtest[Ytest==0], Xtest[Ytest==1]],
                       outFile=outFolder+"/performance/features.png", legendLabels=['Data Train', 'Higgs Train', 'Data Test', 'Higgs Test'],
                       colors=['blue', 'red', 'blue', 'red'], histtypes=[u'step', u'step', 'bar', 'bar'],
                       alphas=[1, 1, 0.4, 0.4], figsize=(20,40), autobins=False,
                       weights=[rWtrain[Ytrain==0], rWtrain[Ytrain==1], rWtest[Ytest==0], rWtest[Ytest==1]], error=True)
# %%
# scale with standard scalers and apply log to any pt and mass distributions
Xtrain = scale(Xtrain,featuresForTraining,  scalerName= outFolder + "/model/myScaler.pkl" ,fit=True)
Xtest  = scale(Xtest, featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)

plotNormalizedFeatures(data=[Xtrain[Ytrain==0], Xtrain[Ytrain==1], Xtest[Ytest==0], Xtest[Ytest==1]],
                       outFile=outFolder+"/performance/features_scaled.png", legendLabels=['Data Train', 'Higgs Train', 'Data Test', 'Higgs Test'],
                       colors=['blue', 'red', 'blue', 'red'], histtypes=[u'step', u'step', 'bar', 'bar'],
                       alphas=[1, 1, 0.4, 0.4], figsize=(20,40), autobins=True,
                       weights=[Wtrain[Ytrain==0], Wtrain[Ytrain==1], Wtest[Ytest==0], Wtest[Ytest==1]], error=True)

# define the model and fit and make predictions
YPredTrain, YPredTest, model, featuresForTraining = PNNClassifier(Xtrain=Xtrain, Xtest=Xtest, Ytrain=Ytrain, Ytest=Ytest, Wtrain=Wtrain, Wtest=Wtest,
                                                    rWtrain=rWtrain, rWtest=rWtest, featuresForTraining=featuresForTraining,
                                                    hp=hp, inFolder=inFolder, outFolder=outFolder)

# unscale in order to make plots 
Xtrain = unscale(Xtrain, featuresForTraining=featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl")
Xtest = unscale(Xtest, featuresForTraining=featuresForTraining,   scalerName =  outFolder + "/model/myScaler.pkl")

# save all the data in the inFolder
save(Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, YPredTrain, YPredTest, inFolder)
# %%
# Plots 
runPlots(Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, YPredTrain, YPredTest, featuresForTraining, model, inFolder, outFolder)
# %%
