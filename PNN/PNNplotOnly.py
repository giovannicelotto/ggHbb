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
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
from plotFeatures import plotNormalizedFeatures

inFolder, outFolder = getInfolderOutfolder(name = "v3b_prova")
modelName = "myModel.h5"
featuresForTraining, columnsToRead = getFeatures(outFolder)
#featuresForTraining = featuresForTraining + ["massHypo"]
hp = getParams()

Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, YPredTrain, YPredTest = loadSaved(inFolder)

plotNormalizedFeatures(data=[Xtrain[Ytrain==0], Xtrain[Ytrain==1], Xtest[Ytest==0], Xtest[Ytest==1]],
                       outFile=outFolder+"/performance/features.png", legendLabels=['Data Train', 'Higgs Train', 'Data Test', 'Higgs Test'],
                       colors=['blue', 'red', 'blue', 'red'], histtypes=[u'step', u'step', 'bar', 'bar'],
                       alphas=[1, 1, 0.4, 0.4], figsize=(20,40), autobins=False,
                       weights=[Wtrain[Ytrain==0], Wtrain[Ytrain==1], Wtest[Ytest==0], Wtest[Ytest==1]], error=True)
model = load_model(outFolder +"/model/"+modelName)
runPlots(Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, YPredTrain, YPredTest, featuresForTraining, model, inFolder, outFolder)