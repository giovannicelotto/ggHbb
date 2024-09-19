import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import sys
sys.path.append("/t3home/gcelotto/ggHbb/PNN/helpers")
from helpers.getFeatures import getFeatures
from helpers.getParams import getParams
from helpers.getInfolderOutfolder import getInfolderOutfolder
from helpers.doPlots import runPlots
from helpers.loadSaved import loadSaved
from tensorflow.keras.models import load_model

inFolder, outFolder = getInfolderOutfolder()
modelName = "myModel.h5"
featuresForTraining, columnsToRead = getFeatures(outFolder)
featuresForTraining = featuresForTraining + ["massHypo"]
hp = getParams()

Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, YPredTrain, YPredTest = loadSaved(inFolder)
model = load_model(outFolder +"/model/"+modelName)
runPlots(Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, YPredTrain, YPredTest, featuresForTraining, model, inFolder, outFolder)