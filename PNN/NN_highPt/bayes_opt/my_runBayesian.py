from skopt import gp_minimize
from skopt.space import Real,  Integer
from skopt.utils import use_named_args
from skopt.callbacks import CheckpointSaver
import joblib
import os
import sys
import numpy as np
import argparse
import time
from datetime import timedelta
#Helpers for ML
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.scaleUnscale import scale, unscale
from helpers.getFeatures import getFeatures, getFeaturesHighPt
from helpers.loadSaved import loadXYWrWSaved
from helpers.getParams import getParams
from helpers.dcorLoss import *
from scipy import stats
from sklearn.metrics import log_loss
from skopt import load
# Torch
import torch    
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
torch.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader, TensorDataset
gpuFlag=True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import and GPU definitions finished

parser = argparse.ArgumentParser(description="Script.")
#### Define arguments
parser.add_argument("-e", "--epochs",           type=int, help="number of epochs", default=1500)
parser.add_argument("-s", "--size",             type=int, help="Number of events to crop training dataset", default=1000000000)
parser.add_argument("-b", "--boosted",          type=int, help="Boosted Class", default=1)
parser.add_argument("-eS", "--earlyStopping",   type=int, help="patience early stop", default=None)
args = parser.parse_args()

featuresForTraining, columnsToRead = getFeaturesHighPt(outFolder=None)
print(featuresForTraining)
inFolder = "/t3home/gcelotto/ggHbb/PNN/input/data_pt%d_1D"%(args.boosted)
outFolderForScaler = "/t3home/gcelotto/ggHbb/PNN/NN_highPt/bayes_opt"

Xtrain, Xval, Ytrain, Yval, Wtrain, Wval, rWtrain, rWval, genMassTrain, genMassVal = loadXYWrWSaved(inFolder=inFolder, isTest=False)
print(len(Xtrain), " events in train dataset")
print(len(Xval), " events in val dataset")
# %%
# %%
# scale with standard scalers and apply log to any pt and mass distributions

Xtrain = scale(Xtrain,featuresForTraining,  scalerName= outFolderForScaler + "/model_b%d/myScaler.pkl"%args.boosted ,fit=True)
Xval  = scale(Xval, featuresForTraining, scalerName= outFolderForScaler + "/model_b%d/myScaler.pkl"%args.boosted ,fit=False)



# %%

Xtrain, Ytrain, Wtrain, rWtrain, genMassTrain = Xtrain[:args.size], Ytrain[:args.size], Wtrain[:args.size], rWtrain[:args.size], genMassTrain[:args.size]
Xval, Yval, Wval, rWval, genMassVal = Xval[:args.size], Yval[:args.size], Wval[:args.size], rWval[:args.size], genMassVal[:args.size]
print("Train Lenght after cutting", len(Xtrain))
print("Val Lenght after cutting", len(Xval))

XtrainTensor = torch.tensor(Xtrain[featuresForTraining].values, dtype=torch.float32, device=device)
YtrainTensor = torch.tensor(Ytrain, dtype=torch.float, device=device).unsqueeze(1)
WtrainTensor = torch.tensor(rWtrain, dtype=torch.float32, device=device).unsqueeze(1)


Xval_tensor = torch.tensor(Xval[featuresForTraining].values, dtype=torch.float32, device=device)
Yval_tensor = torch.tensor(Yval, dtype=torch.float, device=device).unsqueeze(1)
Wval_tensor = torch.tensor(rWval, dtype=torch.float32, device=device).unsqueeze(1)



hp = getParams()

traindataset = TensorDataset(
    XtrainTensor.to(device),
    YtrainTensor.to(device),
    WtrainTensor.to(device),
)
val_dataset = TensorDataset(
    Xval_tensor.to(device),
    Yval_tensor.to(device),
    Wval_tensor.to(device),
)
# Drop last to drop the last (if incomplete size) batch
hp["batch_size"] = hp["batch_size"] if hp["batch_size"]<len(Xtrain) else len(Xtrain)
train_dataloader = DataLoader(traindataset, batch_size=hp["batch_size"], shuffle=True, drop_last=True)

hp["val_batch_size"] = hp["batch_size"] if hp["batch_size"]<len(Xval) else len(Xval)
val_dataloader = DataLoader(val_dataset, batch_size=hp["val_batch_size"], shuffle=False, drop_last=True)




def train_and_evaluate(hp):
    # Create model and optimizer with current hyperparams
    model = Classifier_HighPt(input_dim=Xtrain[featuresForTraining].shape[1], nNodes=hp["nNodes"], dropout_prob=hp["dropout"])
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=hp["learning_rate"])
    criterion = nn.BCELoss(reduction='none')

    # Dataloaders (batch size may have changed)
    train_dataloader = DataLoader(traindataset, batch_size=min(hp["batch_size"], len(traindataset)), shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=min(hp["batch_size"], len(val_dataset)), shuffle=False, drop_last=True)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_weights = None

    for epoch in range(hp["epochs"]):
        
        model.train()
        total_trainloss = 0.0
        for batch in train_dataloader:
            X_batch, Y_batch, W_batch = batch
            optimizer.zero_grad()
            predictions = model(X_batch)
            raw_loss = criterion(predictions, Y_batch)
            classifier_loss = (raw_loss * W_batch).mean()
            classifier_loss.backward()
            optimizer.step()
            total_trainloss += classifier_loss.item()

        avg_train_loss = total_trainloss / len(train_dataloader)
        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                X_batch, Y_batch, W_batch = batch
                predictions = model(X_batch)
                raw_loss = criterion(predictions, Y_batch)
                classifier_loss = (raw_loss * W_batch).mean()
                total_val_loss += classifier_loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        if epoch%20==0:
            print(f"{epoch}/{hp['epochs']} : {avg_train_loss:.4f} | {avg_val_loss:.4f}", flush=True)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= hp["patienceES"]:
            break

    if best_model_weights:
        model.load_state_dict(best_model_weights)
        torch.save(model, outFolderForScaler+"/model_b1/model.pth")




# Penalize overfit
    y_pred = model(Xval_tensor).detach().numpy()
    log_loss_value = log_loss(Yval, y_pred, sample_weight=Wval)

    signal_mask = genMassVal == 125
    bkg_mask = Yval == 0

    signal_predictions = y_pred[signal_mask]
    bkg_predictions = y_pred[bkg_mask]

    signalTrain_predictions = model(XtrainTensor).detach().numpy()[genMassTrain == 125]
    bkgTrain_predictions = model(XtrainTensor).detach().numpy()[Ytrain == 0]


    pval_sig = stats.kstest(signalTrain_predictions, signal_predictions)[1]
    pval_bkg = stats.kstest(bkgTrain_predictions, bkg_predictions)[1]
    # Penalize overfit
    penalty = 0.0

    if (pval_sig < 0.01) or (pval_bkg < 0.01):
        penalty = 1.0
    elif (pval_sig < 0.02) or (pval_bkg < 0.02):
        penalty = 0.8
    elif (pval_sig < 0.03) or (pval_bkg < 0.03):
        penalty = 0.7
    elif (pval_sig < 0.04) or (pval_bkg < 0.04):
        penalty = 0.6
    elif (pval_sig < 0.05) or (pval_bkg < 0.05):
        penalty = 0.5

    global best_model, best_logloss
    if (log_loss_value + penalty) < best_logloss:
        best_logloss = log_loss_value + penalty

    return -(log_loss_value + penalty)  # Maximize score (BayesOpt maximizes)



# Define skopt parameter space
space = [
    #Real(1e-5, 5e-3, name='learning_rate'),
    #Integer(8, 12, name='batch_size_log2'),
    Real(0.0, 0.5, name='dropout'),
    Integer(5, 11, name='n1_log2'),
    Integer(2, 9, name='n2_log2'),
    Integer(0, 7, name='n3_log2'),
]

# Global tracking
best_logloss = float('inf')

# Redefine objective to match skopt's expectations
@use_named_args(space) # To map the list of arguments in a dictionary
def train_and_evaluate_skopt(dropout, n1_log2, n2_log2, n3_log2):
    learning_rate, batch_size_log2 = 9e-4, 16384
    global best_logloss
    start_time = time.time()
    batch_size = int(2 ** int(batch_size_log2))
    MIN_LAYER_SIZE_LOG2 = 2
    if n3_log2 < MIN_LAYER_SIZE_LOG2:
        nodes = [int(2 ** int(n1_log2)), int(2 ** int(n2_log2))]
    else:
        nodes = [int(2 ** int(n1_log2)), int(2 ** int(n2_log2)), int(2 ** int(n3_log2))]

    hp = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "dropout": dropout,
        "nNodes": nodes,
        "patienceES": args.earlyStopping,
        "epochs": args.epochs,
    }

    val_loss = train_and_evaluate(hp)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Trial completed in: {timedelta(seconds=elapsed)} ({elapsed:.2f} seconds)", flush=True)

    return -val_loss  # skopt minimizes

# Save checkpoint after each trial
log_path = f"/t3home/gcelotto/ggHbb/PNN/NN_highPt/bayes_opt/model_b{args.boosted}/skopt_checkpoint.pkl"

# Restart from previous attempts if any
if os.path.exists(log_path):
    print("Found previous points")
    previous = load(log_path)
    print(previous.x_iters)
    print(previous.func_vals)
    checkpoint_saver = CheckpointSaver(log_path, compress=9)

    result = gp_minimize(
        func=train_and_evaluate_skopt,
        dimensions=space,
        acq_func='EI',       # Expected Improvement
        n_calls=20 + len(previous.x_iters) ,          #n_calls = previous + random + bayes
        n_initial_points=10,
        initial_point_generator='random',
        x0=previous.x_iters,   # pre-evaluate known good configs
        y0=previous.func_vals,   # pre-evaluate known good configs
        callback=[checkpoint_saver],
        random_state=42,
        verbose=True,
    )

# Start a new search
else:
    checkpoint_saver = CheckpointSaver(log_path, compress=9)
    initial_params = [
        [0.35, 5, 4, 2],  # known good config

    ]

    print("Starting skopt optimization...")
    result = gp_minimize(
        func=train_and_evaluate_skopt,
        dimensions=space,
        acq_func='EI',       # Expected Improvement
        n_calls=40,          
        n_initial_points=30,
        initial_point_generator='random',
        x0=initial_params,   # pre-evaluate known good configs
        callback=[checkpoint_saver],
        random_state=42,
        verbose=True,
    )


# Show best result
print("\nBest loss:", result.fun)
print("Best params:")
for dim, val in zip(space, result.x):
    print(f"  {dim.name}: {val:.6f}")

# Optionally save final result
joblib.dump(result, log_path.replace('checkpoint.pkl', 'final_result.pkl'))
