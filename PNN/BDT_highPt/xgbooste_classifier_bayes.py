# %%
import numpy as np
import sys
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from datetime import datetime

# Get current month and day
current_date = datetime.now().strftime("%b%d")  # This gives the format like 'Dec12'

# PNN helpers
from helpers.getFeatures import getFeatures, getFeaturesHighPt
from helpers.getParams import getParams
from helpers.loadSaved import loadXYWrWSaved
from helpers.getInfolderOutfolder import getInfolderOutfolder
#from helpers.scaleUnscale import scale, unscale
from helpers.dcorLoss import *

import xgboost as xgb
import pandas as pd
from sklearn.metrics import log_loss
import glob
# %%
from bayes_opt import BayesianOptimization

gpuFlag=True if torch.cuda.is_available() else False

# Define folder of input and output. Create the folders if not existing
hp = getParams()

# Argument parsing (kept the same for flexibility)
import argparse

parser = argparse.ArgumentParser(description="Script.")
parser.add_argument("-v", "--version", type=float, help="version to characterize a model", default=None)
parser.add_argument("-s", "--size", type=int, help="Number of events to crop training dataset", default=1000000000)
parser.add_argument("-b", "--boosted", type=int, help="Boosted Class", default=1)
parser.add_argument("-eS", "--earlyStopping", type=int, help="patience early stop", default=None)

# %%
# Set default or override parameters based on the input arguments
args = parser.parse_args()
args_dict = vars(args)
for key, value in args_dict.items():
    if value is not None:
        hp[key] = value
# Loading data and preprocessing
sampling = False
inFolder, outFolder = getInfolderOutfolder(name="%s_%d_%s"%(current_date, args.boosted, str(hp["version"]).replace('.', 'p')), suffixResults='_BDT')
inFolder = "/t3home/gcelotto/ggHbb/PNN/input/data_sampling_pt%d_1D"%(args.boosted) if sampling else "/t3home/gcelotto/ggHbb/PNN/input/data_pt%d_1D"%(args.boosted)

# Define features to read and train the model
featuresForTraining, columnsToRead = getFeaturesHighPt(outFolder, jet3_btagWP=True)
assert 'dimuon_mass' in featuresForTraining, "No dimuon mass here"
# Load and preprocess data
Xtrain, Xval, Ytrain, Yval, Wtrain, Wval, rWtrain, rWval, genMassTrain, genMassVal = loadXYWrWSaved(inFolder=inFolder, isTest=False)
dijetMassTrain = np.array(Xtrain.dijet_mass.values)
dijetMassVal = np.array(Xval.dijet_mass.values)
print(len(Xtrain), " events in train dataset")
print(len(Xval), " events in val dataset")



# Cut data to the specified size
size = hp["size"]
Xtrain, Ytrain, Wtrain, rWtrain, genMassTrain, dijetMassTrain = Xtrain[:size], Ytrain[:size], Wtrain[:size], rWtrain[:size], genMassTrain[:size], dijetMassTrain[:size]
Xval, Yval, Wval, rWval, genMassVal, dijetMassVal = Xval[:size], Yval[:size], Wval[:size], rWval[:size], genMassVal[:size], dijetMassVal[:size]
print("Train Length after cutting", len(Xtrain))
print("Val Length after cutting", len(Xval))
# %%
# Prepare data for XGBoost
X_train = Xtrain[featuresForTraining].values
X_val = Xval[featuresForTraining].values



# Create DMatrix for XGBoost (the data structure used by XGBoost)
dtrain = xgb.DMatrix(X_train, label=Ytrain, weight=rWtrain)
dval = xgb.DMatrix(X_val, label=Yval, weight=rWval)




def xgb_eval(learning_rate, max_depth, reg_lambda, batch_size, subsample, colsample_bytree):
    #fl = focal_loss(alpha=alpha, gamma=gamma)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": learning_rate,
        "batch_size": int(batch_size),
        "max_depth": int(max_depth),
        "lambda": reg_lambda,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "tree_method": 'auto',
        "verbosity": 0,
        "seed": 1999,
    }
    bst = xgb.train(
        params,
        dtrain,
        #obj=fl,
        num_boost_round=hp["epochs"],
        evals=[(dtrain, 'train'), (dval, 'eval')],
        early_stopping_rounds=hp["patienceES"],
        verbose_eval=False
    )

    best_iteration = bst.best_iteration
    y_pred = bst.predict(dval, iteration_range=(0, best_iteration + 1))
    log_loss_value = log_loss(Yval, y_pred, sample_weight=Wval)
    # KS constraint
    signal_mask = genMassVal == 125
    bkg_mask = Yval == 0

    signal_predictions = y_pred[signal_mask]
    bkg_predictions = y_pred[bkg_mask]

    signalTrain_predictions = bst.predict(dtrain, iteration_range=(0, best_iteration + 1))[genMassTrain == 125]
    bkgTrain_predictions = bst.predict(dtrain, iteration_range=(0, best_iteration + 1))[Ytrain == 0]


    pval_sig = stats.kstest(signalTrain_predictions, signal_predictions)[1]
    pval_bkg = stats.kstest(bkgTrain_predictions, bkg_predictions)[1]
    # Penalize overfit
    penalty = 0.0
    if (pval_sig < 0.05) | (pval_bkg < 0.05):
        penalty = 10.0  # Add penalty to avoid overfit

    global best_model, best_logloss
    if (log_loss_value + penalty) < best_logloss:
        best_logloss = log_loss_value + penalty
        best_model = bst

    return -(log_loss_value + penalty)  # Maximize score (BayesOpt maximizes)











from itertools import product
from scipy import stats
import os


results = []
best_model = None
best_logloss = float('inf')
best_params = None

# Bounds of hyperparameters
pbounds = {
    'learning_rate': (0.005, 0.1),
    'max_depth': (2, 5),
    'reg_lambda': (1, 1000),
    'batch_size': (4096, 4096),
    'subsample': (0.6, 0.9),
    'colsample_bytree': (0.6, 0.9),
    #'alpha': (0.25, 1),
    #'gamma': (0,2),
}

optimizer = BayesianOptimization(
    f=xgb_eval,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(
    init_points=10,     # Random initial points
    n_iter=30          # Guided optimization steps
)
print("Best Result:", optimizer.max)
best_model.save_model(outFolder + "/model/xgboost_model.json")
print("Model saved")
#best_params = optimizer.max['params']
#best_params['max_depth'] = int(best_params['max_depth'])
#best_params['batch_size'] = int(best_params['batch_size'])
#best_params["eval_metric"]= "logloss"
#best_params["seed"]= 1999
#best_params["subsample"]= 0.7
#best_params["colsample_bytree"]= 0.7
#best_params["tree_method"]= "auto"
#print("Best Params chosen" , best_params)
## Retrain final model
#final_model = xgb.train(
#    best_params,
#    dtrain,
#    num_boost_round=hp["epochs"],
#    evals=[(dtrain, 'train'), (dval, 'eval')],
#    #early_stopping_rounds=hp["patienceES"],
#    verbose_eval=True
#)
#