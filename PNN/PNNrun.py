import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import sys
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

sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
from plotFeatures import plotNormalizedFeatures


inFolder, outFolder = getInfolderOutfolder()
featuresForTraining, columnsToRead = getFeatures(outFolder)
hp = getParams()
nReal = 10
nMC = -1

        
data = loadData(nReal, nMC, outFolder, columnsToRead, featuresForTraining, hp)
Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest = data
rWtrain, rWtest = flattenWeights(Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, inFolder, outName=outFolder+ "/performance/massReweighted.png")

plotNormalizedFeatures(data=[Xtrain[Ytrain==0], Xtrain[Ytrain==1], Xtest[Ytest==0], Xtest[Ytest==1]],
                       outFile=outFolder+"/performance/features.png", legendLabels=['Data Train', 'Higgs Train', 'Data Test', 'Higgs Test'],
                       colors=['blue', 'red', 'blue', 'red'], histtypes=[u'step', u'step', 'bar', 'bar'],
                       alphas=[1, 1, 0.4, 0.4], figsize=(10,30), autobins=False,
                       weights=[Wtrain[Ytrain==0], Wtrain[Ytrain==1], Wtest[Ytest==0], Wtest[Ytest==1]], error=True)
featuresForTraining = featuresForTraining + ['massHypo']
Xtrain = scale(Xtrain,featuresForTraining,  scalerName= outFolder + "/model/myScaler.pkl" ,fit=True)
print(Xtrain.jet1_eta.mean()), Xtrain.jet1_eta.std()
sys.exit()
Xtest  = scale(Xtest, featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)


YPredTrain, YPredTest, model, featuresForTraining = PNNClassifier(Xtrain=Xtrain, Xtest=Xtest, Ytrain=Ytrain, Ytest=Ytest, Wtrain=Wtrain, Wtest=Wtest,
                                                    rWtrain=rWtrain, rWtest=rWtest, featuresForTraining=featuresForTraining,
                                                    hp=hp, inFolder=inFolder, outFolder=outFolder)

Xtrain = unscale(Xtrain, featuresForTraining=featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl")
Xtest = unscale(Xtest, featuresForTraining=featuresForTraining,   scalerName =  outFolder + "/model/myScaler.pkl")

save(Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, YPredTrain, YPredTest, inFolder)

# Plots 
runPlots(Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, YPredTrain, YPredTest, featuresForTraining, model, inFolder, outFolder)