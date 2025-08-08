from bayes_opt import BayesianOptimization
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
#parser.add_argument("-v", "--version",          type=float, help="version of the model", default=0)
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
        if epoch%10==0:
            print(epoch)
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


def train_and_evaluate_bayes(learning_rate, batch_size_log2, dropout, n1_log2, n2_log2, n3_log2):
    start_time = time.time()
    batch_size = int(2 ** int(batch_size_log2))
    MIN_LAYER_SIZE_LOG2 = 2  # Equivalent to 4 nodes
    # thirs layer can have 4 nodes or more.
    # If less  remove the third layer
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



    

    return val_loss  # bayes_opt maximizes, so we minimize val_loss




# Here the real Bayes optimization is running


from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
import os
print("Import of bayesian libraries done...")
best_logloss = float('inf')
pbounds = {
    'learning_rate': (1e-5, 5e-3),
    'batch_size_log2': (8, 16),
    'dropout': (0.0, 0.5),
    'n1_log2': (5, 12),              # 2^5 = 32 to 2^9 = 512
    'n2_log2': (4, 10),
    'n3_log2': (0, 7),
}

optimizer = BayesianOptimization(
    f=train_and_evaluate_bayes,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

# Set up logger
print("Set up logger...")
print("Check ", np.average(Xtrain.dijet_mass[Ytrain==0], weights=rWtrain[Ytrain==0]))
print("Check ", np.average(Xtrain.dijet_mass[Ytrain==1], weights=rWtrain[Ytrain==1]))
log_path = "/t3home/gcelotto/ggHbb/PNN/NN_highPt/bayes_opt/model_b%d/logs_bayes.json"%args.boosted
## Load previous logs if available
if os.path.exists(log_path):
    print("Path was loaded", flush=True)
    load_logs(optimizer, logs=[log_path])
logger = JSONLogger(path=log_path)
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
#



# ---- INSERT custom known good trial(s) here ----
# Example of known config you want to test:
known_config = {
    'learning_rate': 1e-3,
    'batch_size_log2': 10,
    'dropout': 0.2,
    'n1_log2': 10,
    'n2_log2': 8,
    'n3_log2': 0,
}
#Evaluate and register manually
print("Evaluate first point...")
target = train_and_evaluate_bayes(**known_config)
optimizer.register(params=known_config, target=target)


optimizer.maximize(init_points=50, n_iter=15)