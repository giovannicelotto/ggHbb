# %%
import numpy as np
import sys
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
sys.path.append("/t3home/gcelotto/ggHbb/PNN/helpers")
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from datetime import datetime
import matplotlib.pyplot as plt
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
class MultiStageNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=[4, 2], hidden_dim2=[32,16]):
        super(MultiStageNN, self).__init__()
        layers = []
        current_dim = input_dim
        
        for h in hidden_dim:
            layers.append(nn.Linear(current_dim, h))
            layers.append(nn.ELU())
            layers.append(nn.BatchNorm1d(h))
            current_dim = h
        
        self.shared_layers = nn.Sequential(*layers)
        
        # First stage outputs (genJetNu1_pt and genJetNu2_pt)
        self.head_jet1 = nn.Linear(current_dim, 1)
        self.head_jet2 = nn.Linear(current_dim, 1)
        
        # Second stage (uses original input + first stage outputs)
        layers=[]
        current_dim = input_dim+2
        layers.append(nn.BatchNorm1d(current_dim))
        for h in hidden_dim2:
            layers.append(nn.Linear(current_dim, h))
            layers.append(nn.ELU())
            layers.append(nn.BatchNorm1d(h))
            current_dim = h
        
        self.secondStage_layers = nn.Sequential(*layers)
        self.final_layer = nn.Linear(h, 1)
    
    def forward(self, x):
        shared_output = self.shared_layers(x)
        jet1_pred = self.head_jet1(shared_output)
        jet2_pred = self.head_jet2(shared_output)
        
        # Use original input X instead of shared_output
        extended_input = torch.cat([x, jet1_pred, jet2_pred], dim=1)
        secondStage_output = self.secondStage_layers(extended_input)
        mass_pred = self.final_layer(secondStage_output)
        
        return jet1_pred, jet2_pred, mass_pred

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



genJetNu1_train = Xtrain.genJetNu_pt_1.values
genJetNu1_val = Xval.genJetNu_pt_1.values
genJetNu1_test = Xtest.genJetNu_pt_1.values


genJetNu2_train = Xtrain.genJetNu_pt_2.values
genJetNu2_val = Xval.genJetNu_pt_2.values
genJetNu2_test = Xtest.genJetNu_pt_2.values

Ytrain = Xtrain.genDijetNu_mass.values
Yval = Xval.genDijetNu_mass.values
Ytest = Xtest.genDijetNu_mass.values
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

XtestTensor = torch.tensor(Xtest[featuresForTraining].values, dtype=torch.float32, device=device)
Y1testTensor = torch.tensor(genJetNu1_test, dtype=torch.float32, device=device).unsqueeze(1)
Y2testTensor = torch.tensor(genJetNu2_test, dtype=torch.float32, device=device).unsqueeze(1)
YtestTensor = torch.tensor(Ytest, dtype=torch.float32, device=device).unsqueeze(1)


train_dataset = TensorDataset(XtrainTensor, Y1trainTensor, Y2trainTensor, YtrainTensor)
val_dataset = TensorDataset(XvalTensor, Y1valTensor, Y2valTensor, YvalTensor)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

# %%
# Loss function
criterion = nn.MSELoss()


# Training loop
model = MultiStageNN(input_dim=Xtrain[featuresForTraining].shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


best_val_loss = float('inf')
patience_counter = 0

# %%
pretraining = 100
midtraining = 100
training = 100
early_stopping_patience = 30

train_loss = []
val_loss = []

for epoch in range(pretraining+midtraining+training):
    # Gradually adjust lambda_mass and lambda_jet
    if epoch<pretraining:
        lambda_mass = 0
        lambda_jet = 1
    elif (epoch>=pretraining) & (epoch<(pretraining+midtraining)):
        lambda_mass = min(1.0, (epoch-pretraining) / (midtraining))  # Linearly increase to 1.0
        lambda_jet = max(0.0, 1.0 - ((epoch-pretraining) / (midtraining)))  # Linearly decrease to 0.0
    elif (epoch>=pretraining+midtraining) & (epoch<pretraining+midtraining+training):
        lambda_mass = 1
        lambda_jet = 0
        
    else:
        pass
    
    def multitask_loss(jet1_pred, jet1_true, jet2_pred, jet2_true, mass_pred, mass_true):
        loss_jet1 = criterion(jet1_pred.squeeze(), jet1_true.squeeze())
        loss_jet2 = criterion(jet2_pred.squeeze(), jet2_true.squeeze())
        loss_mass = criterion(mass_pred.squeeze(), mass_true.squeeze())
        return lambda_mass * loss_mass + lambda_jet * (loss_jet1 + loss_jet2)
    
    model.train()
    total_train_loss = 0.0

    for X_batch, Y1_batch, Y2_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        jet1_pred, jet2_pred, mass_pred = model(X_batch)
        loss = multitask_loss(jet1_pred, Y1_batch, jet2_pred, Y2_batch, mass_pred, Y_batch)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    # Validation
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for X_batch, Y1_batch, Y2_batch, Y_batch in val_loader:
            jet1_pred, jet2_pred, mass_pred = model(X_batch)
            loss = multitask_loss(jet1_pred, Y1_batch, jet2_pred, Y2_batch, mass_pred, Y_batch)
            total_val_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)

    train_loss.append(avg_train_loss)
    val_loss.append(avg_val_loss)
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Lambda_mass: {lambda_mass:.2f}, Lambda_jet: {lambda_jet:.2f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_weights = model.state_dict()
    else:
        patience_counter += 1

    if (patience_counter >= early_stopping_patience) & (epoch>midtraining+training+pretraining+early_stopping_patience):
        print("Early Stopping Triggered!")
        break

model.load_state_dict(best_model_weights)
torch.save(model.state_dict(), "multistage_model.pth")
print("Model Saved!")
#%%

fig, ax = plt.subplots(1,1)
ax.plot(train_loss)
ax.plot(val_loss)
ax.set_xlim(pretraining-20, ax.get_xlim()[1])
# %%
# %%
Xtrain = unscale(Xtrain, Xtrain.columns, scalerName= outFolder + "/myScaler.pkl", log=False)
Xtest = unscale(Xtest, Xtest.columns, scalerName= outFolder + "/myScaler.pkl", log=False)
Xval = unscale(Xval, Xval.columns, scalerName= outFolder + "/myScaler.pkl", log=False)
# %%

predictions_test = model(XtestTensor)[2].detach().squeeze().numpy()
predictions_val = model(XvalTensor)[2].detach().squeeze().numpy()
df_pred=Xval
df_pred['genDijetNu_mass']=predictions_val
df_pred = unscale(df_pred, df_pred.columns,scalerName= outFolder + "/myScaler.pkl", log=False)
predictions_val=df_pred.genDijetNu_mass
# %%
bins = np.linspace(40, 300, 91)
fig, ax = plt.subplots(1, 1)
m=125
ax.hist(Xval.dijet_mass[genMassVal==m], bins=bins, histtype='step', label='Reco')
ax.hist(predictions_val[genMassVal==m], bins=bins, histtype='step', label='regressed')
ax.hist(Xval.genDijetNu_mass[genMassVal==m], bins=bins, histtype='step', label='Gen')
ax.legend()


reco = np.clip(Xval.dijet_mass[genMassVal==m], 50, 180)
regressed = np.clip(predictions_val[genMassVal==m], 50, 180)
print(np.std(reco)/np.mean(reco))
print(np.std(regressed)/np.mean(regressed))

# %%
