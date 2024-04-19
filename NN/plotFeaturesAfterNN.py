import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
from plotFeatures import plotNormalizedFeatures

Xtrain = pd.read_parquet("/t3home/gcelotto/ggHbb/NN/input/Xtrain.parquet")
Xtest = pd.read_parquet("/t3home/gcelotto/ggHbb/NN/input/Xtest.parquet")
Ytrain = pd.read_parquet("/t3home/gcelotto/ggHbb/NN/input/Ytrain.parquet")
Ytest = pd.read_parquet("/t3home/gcelotto/ggHbb/NN/input/Ytest.parquet")
yTrain_predict = np.load("/t3home/gcelotto/ggHbb/NN/input/yTrain_predict.npy")
y_predict = np.load("/t3home/gcelotto/ggHbb/NN/input/yTest_predict.npy")


#sys.exit("exit")



cut = 0.9
#Ytest = Ytest.reset_index(drop=True)
#Xtest = Xtest.reset_index(drop=True)
#Ytrain = Ytrain.reset_index(drop=True)
#Xtrain = Xtrain.reset_index(drop=True)
mSigTest= ((Ytest==1) & (y_predict>cut)).label
mSigTrain = ((Ytrain==1) & (yTrain_predict>cut)).label
#print(type(Xtrain))
#print(mSigTrain)
#print(Xtrain[mSigTrain])
#sys.exit("exit")
mBkgTest= ((Ytest==0) & (y_predict>cut)).label
mBkgTrain = ((Ytrain==0) & (yTrain_predict>cut)).label
plotNormalizedFeatures(data =   [Xtrain[mSigTrain],
                                 Xtrain[mBkgTrain],
                                 Xtest[mSigTest],  
                                 Xtest[mBkgTest]],
                                outFile = "/t3home/gcelotto/ggHbb/NN/output/features_afterCut_0p%d.png"%(cut*10),
                                legendLabels = ['Signal train ', 'BParking train', 'Signal test ', 'BParking test'] , colors = ['blue', 'red', 'blue', 'red'],
                                histtypes=[u'step', u'step', 'bar', 'bar'],
                                figsize=(20, 30),
                                alphas=[1, 1, 0.4, 0.4])
