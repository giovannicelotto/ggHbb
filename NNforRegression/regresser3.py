# %%
import numpy as np
import sys
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
sys.path.append("/t3home/gcelotto/ggHbb/PNN/helpers")
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from datetime import datetime

# Get current month and day
current_date = datetime.now().strftime("%b%d")  # This gives the format like 'Dec12'

# PNN helpers
from helpers.getFeatures import getFeatures, getFeaturesHighPt
from helpers.getParams import getParams
from helpers.loadSaved import loadXYWrWSaved
from helpers.getInfolderOutfolder import getInfolderOutfolder
from helpers.scaleUnscale import scale, unscale
from helpers.dcorLoss import *
import torch
import torch.nn as nn
import torch.nn.init as init
# Torch
import torch    
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
torch.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader, TensorDataset
import argparse
# ------------------------
# First Stage Model: Predicts genJetNu1_pt and genJetNu2_pt
# ------------------------
class FirstStageNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=[16, 4]):
        super(FirstStageNN, self).__init__()
        layers = []
        current_dim = input_dim
        for h in hidden_dim:
            layers.append(nn.Linear(current_dim, h))
            layers.append(nn.ELU())
            layers.append(nn.BatchNorm1d(h))
            current_dim = h
        layers.append(nn.Linear(current_dim, 2))  # Predicts genJetNu1_pt and genJetNu2_pt
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ------------------------
# Second Stage Model: Predicts genDijetNu_mass using original features + predicted genJetNu1_pt and genJetNu2_pt
# ------------------------
class SecondStageNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=[16, 4]):
        super(SecondStageNN, self).__init__()
        layers = []
        current_dim = input_dim + 2  # +2 because we add the two intermediate outputs
        for h in hidden_dim:
            layers.append(nn.Linear(current_dim, h))
            layers.append(nn.ELU())
            layers.append(nn.BatchNorm1d(h))
            current_dim = h
        layers.append(nn.Linear(current_dim, 1))  # Predicts genDijetNu_mass
        self.model = nn.Sequential(*layers)

    def forward(self, x, y1_y2_pred):
        x = torch.cat([x, y1_y2_pred], dim=1)  # Concatenate original inputs with predicted Y1 and Y2
        return self.model(x)

# Theory
# initializing all weights to zero in neural networks is largely inadvisable due to the critical issue of symmetry breaking.
# However, zero initialization for biases is generally acceptable as it does not contribute to the symmetry problem in the same way weights do. Biases can start from zero since their primary role is to provide an adjustable threshold for neuron activation rather than to diversify learning paths.
#                The batch normalization layer has two learnable parameters:
#
#weight (also called gamma): This is the scaling factor applied to the normalized output.
#bias (also called beta): This is the shifting factor added after normalization.
#Typically, the initialization for these parameters is:
#
#weight (gamma): Initialized to 1 (to preserve the identity transformation initially).
#bias (beta): Initialized to 0 (no shift initially).
#PyTorch's nn.BatchNorm1d layers are automatically initialized to these defaults, so you don't necessarily need to manually initialize them unless you have a specific reason to do so.

    # this is autocalled for predictions. apply the model (self) to x
# %%
gpuFlag=True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
sampling = True
inFolder, outFolder = "/t3home/gcelotto/ggHbb/NNforRegression/input", "/t3home/gcelotto/ggHbb/NNforRegression/results"
import pandas as pd

Xtrain = pd.read_parquet(inFolder+"/Xtrain.parquet")
Xval = pd.read_parquet(inFolder+"/Xval.parquet")
Xtest = pd.read_parquet(inFolder+"/Xtest.parquet")
#Ytrain = np.load(inFolder+"/Ytrain.npy")
#Yval = np.load(inFolder+"/Yval.npy")
#Ytest = np.load(inFolder+"/Ytest.npy")
Ytrain = Xtrain.genDijetNu_mass.values
Yval = Xval.genDijetNu_mass.values
Ytest = Xtest.genDijetNu_mass.values
# %%
genJetNu1_train = Xtrain.genJetNu_pt_1.values
genJetNu1_val = Xval.genJetNu_pt_1.values
genJetNu1_test = Xtest.genJetNu_pt_1.values


genJetNu2_train = Xtrain.genJetNu_pt_2.values
genJetNu2_val = Xval.genJetNu_pt_2.values
genJetNu2_test = Xtest.genJetNu_pt_2.values

# %%
genMassTrain = np.load(inFolder+"/genMassTrain.npy")
genMassVal = np.load(inFolder+"/genMassVal.npy")
genMassTest = np.load(inFolder+"/genMassTest.npy")

# scale with standard scalers and apply log to any pt and mass distributions

Xtrain = scale(Xtrain,Xtrain.columns,  scalerName= outFolder + "/myScaler.pkl" ,fit=True, log=False, scaler='robust')
Xval  = scale(Xval, Xval.columns, scalerName= outFolder + "/myScaler.pkl" ,fit=False, log=False, scaler='robust')
Xtest  = scale(Xtest, Xtest.columns, scalerName= outFolder + "/myScaler.pkl" ,fit=False, log=False, scaler='robust')

# %%
featuresForTraining = [
    "jet1_pt",    "jet1_eta",    "jet1_phi",    "jet1_mass", #"jet1_btagDeepFlavB",
    "jet1_btagTight", "jet1_dR_dijet","jet1_nMuons",
    "jet2_pt",    "jet2_eta",    "jet2_phi",    "jet2_mass", #"jet2_btagDeepFlavB",
    "jet2_btagTight", "jet2_dR_dijet","jet2_nMuons",
    "jet3_pt",    "jet3_eta",    "jet3_phi",    "jet3_mass", "jet3_dR_dijet","jet3_nMuons",
    "muon_pt",    "muon_eta",    "muon_dxySig",
    "dijet_pt",    "dijet_mass",
    "ht",    "nJets",
    #"genJetNu_pt_1",    "genJetNu_pt_2",    "genDijetNu_mass",    "target"
]
# ------------------------
# Data Preparation
# ------------------------
XtrainTensor = torch.tensor(Xtrain[featuresForTraining].values, dtype=torch.float32, device=device)
Y1trainTensor = torch.tensor(genJetNu1_train, dtype=torch.float32, device=device).unsqueeze(1)
Y2trainTensor = torch.tensor(genJetNu2_train, dtype=torch.float32, device=device).unsqueeze(1)
YtrainTensor = torch.tensor(Ytrain, dtype=torch.float32, device=device).unsqueeze(1)

XvalTensor = torch.tensor(Xval[featuresForTraining].values, dtype=torch.float32, device=device)
Y1valTensor = torch.tensor(genJetNu1_val, dtype=torch.float32, device=device).unsqueeze(1)
Y2valTensor = torch.tensor(genJetNu2_val, dtype=torch.float32, device=device).unsqueeze(1)
YvalTensor = torch.tensor(Yval, dtype=torch.float32, device=device).unsqueeze(1)

train_dataset = TensorDataset(XtrainTensor, Y1trainTensor, Y2trainTensor, YtrainTensor)
val_dataset = TensorDataset(XvalTensor, Y1valTensor, Y2valTensor, YvalTensor)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
# %%
# ------------------------
# Training Parameters
# ------------------------
epochs = 300
criterion = nn.MSELoss()
lr = 1e-3
early_stopping_patience = 30

# ------------------------
# First Stage Training (Predicts genJetNu1_pt and genJetNu2_pt)
# ------------------------
first_stage = FirstStageNN(input_dim=Xtrain[featuresForTraining].shape[1]).to(device)
optimizer1 = optim.Adam(first_stage.parameters(), lr=lr)

best_val_loss = float('inf')
patience_counter = 0
print("Training First Stage Model...")

for epoch in range(epochs):
    first_stage.train()
    total_train_loss = 0.0

    for X_batch, Y1_batch, Y2_batch, _ in train_loader:
        optimizer1.zero_grad()
        Y1_Y2_pred = first_stage(X_batch)
        loss = criterion(Y1_Y2_pred[:, 0], Y1_batch.squeeze()) + criterion(Y1_Y2_pred[:, 1], Y2_batch.squeeze())
        loss.backward()
        optimizer1.step()
        total_train_loss += loss.item()

    # Validation
    first_stage.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for X_batch, Y1_batch, Y2_batch, _ in val_loader:
            Y1_Y2_pred = first_stage(X_batch)
            loss = criterion(Y1_Y2_pred[:, 0], Y1_batch.squeeze()) + criterion(Y1_Y2_pred[:, 1], Y2_batch.squeeze())
            total_val_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_first_stage_weights = first_stage.state_dict()
    else:
        patience_counter += 1

    if patience_counter >= early_stopping_patience:
        print("First Stage Early Stopping Triggered!")
        break

first_stage.load_state_dict(best_first_stage_weights)
torch.save(first_stage.state_dict(), "first_stage.pth")
print("First Stage Model Saved!")
# %%x
# ------------------------
# Second Stage Training (Predicts genDijetNu_mass)
# ------------------------
second_stage = SecondStageNN(input_dim=Xtrain[featuresForTraining].shape[1]).to(device)
optimizer2 = optim.Adam(second_stage.parameters(), lr=lr)

best_val_loss = float('inf')
patience_counter = 0
print("Training Second Stage Model...")

for epoch in range(epochs):
    second_stage.train()
    total_train_loss = 0.0

    for X_batch, _, _, Y_batch in train_loader:
        with torch.no_grad():
            Y1_Y2_pred = first_stage(X_batch)

        optimizer2.zero_grad()
        Y_pred = second_stage(X_batch, Y1_Y2_pred.detach())  # Detach to avoid backprop to first stage
        loss = criterion(Y_pred.squeeze(), Y_batch.squeeze())
        loss.backward()
        optimizer2.step()
        total_train_loss += loss.item()

    # Validation
    second_stage.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for X_batch, _, _, Y_batch in val_loader:
            Y1_Y2_pred = first_stage(X_batch)
            Y_pred = second_stage(X_batch, Y1_Y2_pred.detach())
            loss = criterion(Y_pred.squeeze(), Y_batch.squeeze())
            total_val_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_second_stage_weights = second_stage.state_dict()
    else:
        patience_counter += 1

    if patience_counter >= early_stopping_patience:
        print("Second Stage Early Stopping Triggered!")
        break

second_stage.load_state_dict(best_second_stage_weights)
torch.save(second_stage.state_dict(), "second_stage.pth")
print("Second Stage Model Saved!")

# %%
Xtrain = unscale(Xtrain, Xtrain.columns, scalerName= outFolder + "/myScaler.pkl", log=False)
Xtest = unscale(Xtest, Xtest.columns, scalerName= outFolder + "/myScaler.pkl", log=False)
Xval = unscale(Xval, Xval.columns, scalerName= outFolder + "/myScaler.pkl", log=False)
# %%
import matplotlib.pyplot as plt

Y1_Y2_pred = first_stage(XvalTensor)
predictions_val = second_stage(XvalTensor, Y1_Y2_pred.detach()).detach().squeeze().numpy()

#predictions_val = (second_stage(tensor)).detach().squeeze().numpy()
bins = np.linspace(40, 300, 81)
fig, ax = plt.subplots(1, 1)
ax.hist(Xval.dijet_mass[genMassVal==125], bins=bins, histtype='step', label='Reco')
ax.hist(Xval.dijet_mass[genMassVal==125]*predictions_val[genMassVal==125], bins=bins, histtype='step', label='regressed')
ax.hist(Xval.genDijetNu_mass[genMassVal==125], bins=bins, histtype='step', label='Gen')
ax.legend()
print(np.std(np.clip(Xval.dijet_mass[genMassVal==125], 60, 180)))
print(np.std(np.clip(Xval.dijet_mass[genMassVal==125]*predictions_val[genMassVal==125], 60, 180)))

# %%
fig, ax = plt.subplots(1, 1)
m = 125
ax.hist(Xval.dijet_mass[genMassVal==m], bins=bins, histtype='step', label='Reco')
ax.hist(predictions_val[genMassVal==m], bins=bins, histtype='step', label='regressed')
ax.hist(Xval.dijet_mass_2018[genMassVal==m], bins=bins, histtype='step', label='2018')
ax.hist(Xval.genDijetNu_mass[genMassVal==m], bins=bins, histtype='step', label='Gen')
ax.legend()
print(np.std(np.clip(Xval.dijet_mass[genMassVal==m], 60, 180)))
print(np.std(np.clip(predictions_val[genMassVal==m], 60, 180)))
# %%
