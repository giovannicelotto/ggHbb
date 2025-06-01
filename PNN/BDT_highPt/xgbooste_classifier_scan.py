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

gpuFlag=True if torch.cuda.is_available() else False

# Define folder of input and output. Create the folders if not existing
hp = getParams()

# Argument parsing (kept the same for flexibility)
import argparse

parser = argparse.ArgumentParser(description="Script.")
parser.add_argument("-v", "--version", type=float, help="version to characterize a model", default=None)
parser.add_argument("-e", "--epochs", type=int, help="number of epochs", default=1500)
parser.add_argument("-s", "--size", type=int, help="Number of events to crop training dataset", default=1000000000)
parser.add_argument("-b", "--boosted", type=int, help="Boosted Class", default=1)
parser.add_argument("-lr", "--learningRate", type=float, help="learning rate", default=None)
parser.add_argument("-bs", "--batch_size", type=int, help="Number of events perBatch", default=None)
parser.add_argument("-eS", "--earlyStopping", type=int, help="patience early stop", default=None)

# %%
# Set default or override parameters based on the input arguments
try:
    args = parser.parse_args()
    if args.version is not None:
        hp["version"] = args.version 
        print("version changed to ", hp["version"])
    if args.epochs is not None:
        hp["epochs"] = args.epochs 
        print("N epochs to ", hp["epochs"])
    if args.batch_size is not None:
        hp["batch_size"] = int(args.batch_size )
        print("N batch_size to ", hp["batch_size"])
    if args.earlyStopping is not None:
        hp["patienceES"] = args.earlyStopping
    if args.learningRate is not None:
        hp["learning_rate"] = args.learningRate
    if args.size != int(1e9):
        hp["size"] = args.size
    if args.boosted != int(1e9):
        boosted = args.boosted
    print("After parameters")
    print(hp)
except:
    print("-"*40)
    print("Error in passing the arguments!")
    hp["version"] = 0. 
    hp["epochs"] = 10000
    hp["size"] = 1000000
    hp["batch_size"]=128
    hp['patienceES'] = 300
    hp["learning_rate"]=0.01
    print(hp)
    boosted =22
# Loading data and preprocessing
sampling = True
inFolder, outFolder = getInfolderOutfolder(name="%s_%d_%s"%(current_date, boosted, str(hp["version"]).replace('.', 'p')), suffixResults='_BDT')
inFolder = "/t3home/gcelotto/ggHbb/PNN/input/data_sampling_pt%d_1D"%(boosted) if sampling else "/t3home/gcelotto/ggHbb/PNN/input/data_highPt"

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
dtrain = xgb.DMatrix(X_train, label=Ytrain, weight=Wtrain)
dval = xgb.DMatrix(X_val, label=Yval, weight=Wval)
from itertools import product
from scipy import stats
import os

# Define parameter grid
learning_rates = [0.0005, 0.001, 0.002]
max_depths = [2, 3, 4]
lambdas = [10, 100, 10000]

results = []
best_model = None
best_logloss = float('inf')
best_params = None

print("Starting hyperparameter scan...")
for lr, depth, lmbda in product(learning_rates, max_depths, lambdas):
    print(f"Trying: lr={lr}, depth={depth}, lambda={lmbda}")

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": lr,
        "max_depth": depth,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "verbosity": 1,
        "tree_method": 'auto',
        "lambda": lmbda
    }

    # Train with early stopping
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=hp["epochs"],
        evals=[(dtrain, 'train'), (dval, 'eval')],
        early_stopping_rounds=hp["patienceES"],
        verbose_eval=False
    )

    # Final training on best iteration
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=bst.best_iteration
    )

    # Predictions for log loss
    y_pred = bst.predict(dval)
    val_loss = log_loss(Yval, y_pred)

    # Predictions for KS test
    signal_mask = genMassVal == 125
    bkg_mask = Yval == 0
    signal_predictions = y_pred[signal_mask]
    realData_predictions = y_pred[bkg_mask]

    signalTrain_predictions = bst.predict(dtrain)[genMassTrain == 125]
    realDataTrain_predictions = bst.predict(dtrain)[Ytrain == 0]

    # KS tests
    pval_sig = stats.kstest(signalTrain_predictions, signal_predictions)[1]
    pval_bkg = stats.kstest(realDataTrain_predictions, realData_predictions)[1]

    results.append((val_loss, pval_sig, pval_bkg, lr, depth, lmbda))
    print(f"→ logloss={val_loss:.4f}, p_sig={pval_sig:.3f}, p_bkg={pval_bkg:.3f}")

    if val_loss < best_logloss and pval_sig > 0.05 and pval_bkg > 0.05:
        print("✅ New best model found.")
        best_logloss = val_loss
        best_model = bst
        best_params = (lr, depth, lmbda)
        # Save model immediately
        model_dir = os.path.join(outFolder, "model_scan")
        os.makedirs(model_dir, exist_ok=True)
        best_model.save_model(f"{model_dir}/best_model.json")

# Save results
results_df = pd.DataFrame(results, columns=["logloss", "pval_sig", "pval_bkg", "lr", "max_depth", "lambda"])
results_df.to_csv(outFolder + "/model_scan/hyperparam_scan_results.csv", index=False)

print("=" * 40)
if best_model:
    print(f"Best Model: lr={best_params[0]}, depth={best_params[1]}, lambda={best_params[2]}")
    print(f"Validation Log Loss: {best_logloss:.4f}")
else:
    print("No valid model passed the KS tests.")
