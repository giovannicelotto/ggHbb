import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import sys
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
sys.path.append("/t3home/gcelotto/ggHbb/PNN/helpers")

from helpers.getFeatures import getFeatures
from helpers.getParams import getParams
from helpers.loadSaved import loadXYWSaved
from helpers.getInfolderOutfolder import getInfolderOutfolder
from helpers.scaleUnscale import scale, unscale

from helpers.doPlots import runPlots
from tensorflow.keras.models import load_model

# Define folder of input and output. Create the folders if not existing
inFolder, outFolder = getInfolderOutfolder()
# Define features to read and to train the pNN (+parameter massHypo) and save the features for training in outfolder
featuresForTraining, columnsToRead = getFeatures(outFolder)
# define the parameters for the nn
hp = getParams()


# load saved data
Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest = loadXYWSaved(inFolder=inFolder)
# load model
featuresForTraining = featuresForTraining + ['massHypo']
modelName = "myModel.h5"
model = load_model(outFolder +"/model/"+modelName)
print("Loaded model ", outFolder +"/model/"+modelName)

# scale input
Xtrain = scale(Xtrain,featuresForTraining,  scalerName= outFolder + "/model/myScaler.pkl" ,fit=True)
Xtest  = scale(Xtest, featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)

# Predictions
YPredTest = model.predict(Xtest[featuresForTraining])
YPredTrain = model.predict(Xtrain[featuresForTraining])

Xtrain = unscale(Xtrain, featuresForTraining=featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl")
Xtest = unscale(Xtest, featuresForTraining=featuresForTraining,   scalerName =  outFolder + "/model/myScaler.pkl")
# Plots 
runPlots(Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, YPredTrain, YPredTest, featuresForTraining, model, inFolder, outFolder)