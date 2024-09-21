# %%
import pandas as pd
import numpy as np

from helpers.getFeatures import getFeatures
from helpers.getInfolderOutfolder import getInfolderOutfolder
from helpers.loadSaved import loadSaved
from helpers.scaleUnscale import scale
from helpers.aucModelEvaluator import aucModelEvaluator


# %%




inFolder, outFolder = getInfolderOutfolder()
featuresForTraining, columnsToRead = getFeatures(massHypo=True)
Xtrain, Xtest, Ytrain, Ytest, Wtrain, Wtest, YPredTrain, YPredTest, rWtrain, rWtest = loadSaved(inFolder, rWeights=True)
Xtest  = scale(Xtest, featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)

#pairs = [(1e-5, 2),
#         (1e-3, 6),
#         (1e-4, 10)]
#max = 0
#lrMax, bsMax  = 0 ,0
#for p in pairs:
#    lr, bs = p
#
#    current = aucModelEvaluator(featuresForTraining=featuresForTraining, Xtrain=Xtrain, Xtest=Xtest,
#                  Ytrain=Ytrain, Ytest=Ytest, rWtrain=rWtrain, 
#                  lr=1e-5, bs=10)
#    if current > max:
#        max = current
#        lrMax = lr
#        bsMax = bs
#        print("new max")
#        print(lrMax, bsMax, max)
# %%
from bayes_opt import BayesianOptimization

max_lr, max_bs = -1, -1
    
pbounds = {
        'lr': (1e-6, 1e-3),
        'bs': (6, 14)}
optimizer = BayesianOptimization(
f=lambda lr, bs: aucModelEvaluator(featuresForTraining, Xtrain, Xtest, Ytrain, Ytest, rWtrain, lr, bs),  # lambda to pass the fixed args,
pbounds=pbounds,
verbose=1, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
random_state=1,
allow_duplicate_points=True
)
    
optimizer.maximize(
    init_points=10,
    n_iter=30,
)

max_lr=optimizer.max["params"]["lr"]
max_bs=optimizer.max["params"]["bs"]
print("auc : %.2f"%aucModelEvaluator(featuresForTraining, Xtrain, Xtest, Ytrain, Ytest, rWtrain, max_lr, max_bs))