# %%
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import sys
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
from plotFeatures import plotNormalizedFeatures
sys.path.append("/t3home/gcelotto/ggHbb/PNN/helpers")
import xgboost as xgb
from helpers.bdt_train_save import bdt_train_save

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
inFolder, outFolder = getInfolderOutfolder()

# Define features to read and to train the pNN (+parameter massHypo) and save the features for training in outfolder
featuresForTraining, columnsToRead = getFeatures(outFolder)

# define the parameters for the nn
hp = getParams()

# number of files from real data and mc
nReal, nMC = 1, -1

# load data for the samples and preprocess the data(pT cut)
# fill the massHypo column
# cut the data to have same length in all the samples
# reweight each sample to have total weight 1, shuffle and split in train and test
data = loadData(nReal, nMC, outFolder, columnsToRead, featuresForTraining, hp)
Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest = data
from sklearn.model_selection import train_test_split
Xtrain, Xval, Ytrain, Yval, Wtrain, Wval = train_test_split(Xtrain, Ytrain, Wtrain, test_size=0.2, random_state=42)
Ytrain = Ytrain.astype(int)
Yval = Yval.astype(int)
import numpy as np
Wtrain = Wtrain/np.mean(Wtrain)
Wval = Wval/np.mean(Wtrain)
# %%
# Convert these datasets to DMatrix format
dtrain = xgb.DMatrix(Xtrain[featuresForTraining], label=Ytrain,  weight=Wtrain)
dval = xgb.DMatrix(Xval[featuresForTraining], label=Yval, weight=Wval)
dtest = xgb.DMatrix(Xtest[featuresForTraining])

#featuresForTraining = featuresForTraining + ['massHypo']
# %%
# No need to scale BDT

bdt_train_save(dtrain=dtrain, dtest=dtest, dval=dval, y_test=Ytest, depth = 5,
               eta = 0.1, num_boost_round=1000, 
               min_child_weight=10,outName=outFolder + "/model/mymodel.model")
rWtrain, rWtest = Wtrain.copy(), Wtest.copy()


# define the model and fit and make predictions
bst_loaded = xgb.Booster()
bst_loaded.load_model(outFolder + "/model/mymodel.model")


#best_iteration = bst_loaded.best_iteration



YPredTrain, YPredTest = bst_loaded.predict(dtrain), bst_loaded.predict(dtest)

# %%
# Plots 
model = None
runPlots(Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, YPredTrain, YPredTest, featuresForTraining, model, inFolder, outFolder)
# %%
