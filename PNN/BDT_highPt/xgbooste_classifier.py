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
parser.add_argument("-l2", "--L2", type=float, help="L2 penalty", default=None)
parser.add_argument("-md", "--maxdepth", type=int, help="max depth of trees", default=None)
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

# XGBoost parameters
params = {
    "objective": "binary:logistic",   # binary classification problem
    "eval_metric": "logloss",          # evaluation metric (logarithmic loss)
    "learning_rate": hp["learning_rate"],
    "max_depth": args.maxdepth,
    "batch_size": int(hp["batch_size"]),
    "subsample": .7,
    "colsample_bytree": 0.7,
    "verbose": 2,
    "tree_method": 'auto',
    "lambda": args.L2,
    "seed": 1999,
    #"alpha": 100  # L1 regularization (default is 0)
}
# %%
# Early stopping parameters
early_stopping_rounds = hp["patienceES"]
evals = [(dtrain, 'train'), (dval, 'eval')]

# Training XGBoost model
print("Training XGBoost model...")
bst = xgb.train(
    params, 
    dtrain, 
    num_boost_round=hp["epochs"], 
    evals=evals, 
    early_stopping_rounds=early_stopping_rounds
)
best_iteration = bst.best_iteration
bst = xgb.train(
    params, 
    dtrain, 
    num_boost_round=best_iteration,  # Use the best iteration
    evals=evals
)
# Save the trained model
bst.save_model(outFolder + "/model/xgboost_model.json")
print("Model saved")

# Evaluation
print("Evaluating model...")
y_pred = bst.predict(dval)
val_loss = log_loss(Yval, y_pred)
print(f"Validation Log Loss: {val_loss:.4f}")

# Save the log loss history (optional, can be used for analysis)
np.save(outFolder + "/model/val_loss_history.npy", val_loss)

# Save model and training configuration
with open(outFolder + "/model/training.txt", "w") as file:
    for key, value in hp.items():
        file.write(f"{key} : {value}\n")

print("Training and evaluation complete.")

# %%
