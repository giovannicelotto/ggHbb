# %%
import numpy as np
import sys
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
parser.add_argument("-v", "--version", type=float, help="version of model", default=None)
parser.add_argument("-e", "--epochs", type=int, help="number of epochs", default=1500)
parser.add_argument("-s", "--size", type=int, help="Number of events to crop training dataset", default=1000000000)
parser.add_argument("-b", "--boosted", type=int, help="Boosted Class", default=1)
parser.add_argument("-lr", "--learningRate", type=float, help="learning rate", default=None)
parser.add_argument("-bs", "--batch_size", type=int, help="Number of events perBatch", default=None)
parser.add_argument("-eS", "--earlyStopping", type=int, help="patience early stop", default=None)
#parser.add_argument("-n", "--nodes",type=lambda s: [int(item) for item in s.split(',')], help="List of nodes per layer",default=None)

# %%
# Set default or override parameters based on the input arguments

args = parser.parse_args()
# Loading data and preprocessing
sampling = True
inFolder, outFolder = getInfolderOutfolder(name="%s_%d_%s"%(current_date, args.boosted, str(args.version).replace('.', 'p')), suffixResults='_mjjDisco')
inFolder ="/t3home/gcelotto/ggHbb/PNN/input/data_pt0_1D"

# Define features to read and train the model
featuresForTraining, columnsToRead = getFeatures(outFolder, massHypo=False, bin_center=False, boosted=args.boosted, jet3_btagWP=True)

# Load and preprocess data
Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, Wtrain, Wval, Wtest, rWtrain, rWval, genMassTrain, genMassVal, genMassTest = loadXYWrWSaved(inFolder=inFolder)
Xtrain, Ytrain, Wtrain, rWtrain, genMassTrain = Xtrain[(genMassTrain==125) | (genMassTrain==0)], Ytrain[(genMassTrain==125) | (genMassTrain==0)], Wtrain[(genMassTrain==125) | (genMassTrain==0)], rWtrain[(genMassTrain==125) | (genMassTrain==0)], genMassTrain[(genMassTrain==125) | (genMassTrain==0)] 
Xval, Yval, Wval, rWval, genMassVal = Xval[(genMassVal==125) | (genMassVal==0)], Yval[(genMassVal==125) | (genMassVal==0)], Wval[(genMassVal==125) | (genMassVal==0)], rWval[(genMassVal==125) | (genMassVal==0)], genMassVal[(genMassVal==125) | (genMassVal==0)] 
Xtest, Ytest, Wtest, genMassTest = Xtest[(genMassTest==125) | (genMassTest==0)], Ytest[(genMassTest==125) | (genMassTest==0)], Wtest[(genMassTest==125) | (genMassTest==0)], genMassTest[(genMassTest==125) | (genMassTest==0)] 
Wtrain[genMassTrain==125]=Wtrain[genMassTrain==125]*5
Wval[genMassVal==125]=Wval[genMassVal==125]*5
Wtest[genMassTest==125]=Wtest[genMassTest==125]*5
dijetMassTrain = np.array(Xtrain.dijet_mass.values)
dijetMassVal = np.array(Xval.dijet_mass.values)
print(len(Xtrain), " events in train dataset")
print(len(Xval), " events in val dataset")

#rawFiles = pd.read_parquet(glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/training/*.parquet")[:10])
#rawFiles = rawFiles[~((rawFiles.jet1_btagTight>0.5) & (rawFiles.jet2_btagTight>0.5))]
#rawFiles = rawFiles[rawFiles.dijet_pt>100]

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
#X_rawFiles = rawFiles[featuresForTraining].values


# Create DMatrix for XGBoost (the data structure used by XGBoost)
dtrain = xgb.DMatrix(X_train, label=Ytrain, weight=Wtrain)
dval = xgb.DMatrix(X_val, label=Yval, weight=Wval)
#draw = xgb.DMatrix(X_rawFiles, label=np.zeros(len(X_rawFiles)), weight=np.ones(len(X_rawFiles)))

# XGBoost parameters
params = {
    "objective": "binary:logistic",   # binary classification problem
    "eval_metric": "logloss",          # evaluation metric (logarithmic loss)
    "learning_rate": args.learningRate,
    "max_depth": 4,
    "subsample": .7,
    "colsample_bytree": 0.7,
    "silent": 1,
    "tree_method": 'auto',
    "lambda": 1,  # L2 regularization (default is 1)
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
    num_boost_round=args.epochs, 
    evals=evals, 
    early_stopping_rounds=early_stopping_rounds
)
best_iteration = bst.best_iteration

print("Finished")
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
