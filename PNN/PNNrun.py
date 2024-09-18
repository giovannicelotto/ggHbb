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

inFolder, outFolder = getInfolderOutfolder()
featuresForTraining, columnsToRead = getFeatures(inFolder)
hp = getParams()
nReal = 5
nMC = -1

        
data = loadData(nReal, nMC, outFolder, columnsToRead, featuresForTraining, hp)
Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest = data

    
Xtrain = scale(Xtrain, scalerName= inFolder + "/myScaler.pkl" ,fit=True)
Xtest  = scale(Xtest, scalerName= inFolder + "/myScaler.pkl" ,fit=False)
             
YPredTrain, YPredTest = PNNClassifier(Xtrain=Xtrain, Xtest=Xtest, Ytrain=Ytrain, Ytest=Ytest, Wtrain=Wtrain, Wtest=Wtest,
                                        hp=hp, inFolder=inFolder, outFolder=outFolder)

Xtrain = unscale(Xtrain,    scalerName= inFolder + "/myScaler.pkl")
Xtest = unscale(Xtest,      scalerName =  inFolder + "/myScaler.pkl")

save(Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, YPredTrain, YPredTest, inFolder)
