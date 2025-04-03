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
class Classifier(nn.Module):
    def __init__(self, input_dim, nNodes=[32, 16]):
        """
        Initialize the classifier model.
        
        Args:
            input_dim (int): Number of input features.
            nNodes (list or tuple): Number of nodes in each hidden layer.
                                     Example: [128, 64, 32]
        """
        super(Classifier, self).__init__()
        
        layers = []
        current_dim = input_dim  # Start with input dimension
        n = nNodes[0]
        layers.append(nn.Linear(current_dim, n))
        layers.append(nn.ELU())
        layers.append(nn.BatchNorm1d(n))
        current_dim = n
        #layers.append(nn.Dropout(dropout_prob))
        for n in nNodes[1:]:
            layers.append(nn.Linear(current_dim, n))
            layers.append(nn.ELU())
            layers.append(nn.BatchNorm1d(n))
            # Update current dimension to the output of this layer
            current_dim = n  
        
        # Add the final output layer
        layers.append(nn.Linear(current_dim, 1))

        # Combine all layers into a Sequential module
        self.fc = nn.Sequential(*layers)

        # Apply weight initialization
        self.apply(self._initialize_weights)
        
    
    
    def _initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            #init.xavier_normal_(layer.weight)
            # Kaiming Initialization for layers with ReLU activations
            init.xavier_normal_(layer.weight)
            # Initialize biases to 0
            init.constant_(layer.bias, 0)

            # Special treatment for the output layer (Sigmoid output layer)
            if layer.out_features == 1:
                init.xavier_normal_(layer.weight)  
                init.constant_(layer.bias, 0)      

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
    def forward(self, x):
        return self.fc(x)
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
XtrainTensor = torch.tensor(Xtrain[featuresForTraining].values, dtype=torch.float32, device=device)
YtrainTensor = torch.tensor(Ytrain, dtype=torch.float, device=device).unsqueeze(1)


Xval_tensor = torch.tensor(Xval[featuresForTraining].values, dtype=torch.float32, device=device)
Yval_tensor = torch.tensor(Yval, dtype=torch.float, device=device).unsqueeze(1)

Xtest_tensor = torch.tensor(Xtest[featuresForTraining].values, dtype=torch.float32, device=device)
Ytest_tensor = torch.tensor(Ytest, dtype=torch.float, device=device).unsqueeze(1)


traindataset = TensorDataset(
    XtrainTensor.to(device),
    YtrainTensor.to(device),
)
val_dataset = TensorDataset(
    Xval_tensor.to(device),
    Yval_tensor.to(device),
)

# Drop last to drop the last (if incomplete size) batch
train_dataloader = DataLoader(traindataset, batch_size=512, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=False, drop_last=True)
# %%
# Model, loss, optimizer
model = Classifier(input_dim=Xtrain[featuresForTraining].shape[1], nNodes=[128, 64])
model.to(device)
epochs = 300
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

early_stopping_patience = 30
best_val_loss = float('inf')
patience_counter = 0
print("Train start")

# ______________________________________________________________________________
# ______________________________________________________________________________
# ______________________  TRAINING PHASE  ______________________________________
# ______________________________________________________________________________
# ______________________________________________________________________________



train_loss_history = []

val_loss_history = []
best_model_weights = None # weights saved for RestoreBestWeights
best_epoch = None

# %%

for epoch in range(epochs):
    model.train()
    total_trainloss = 0.0
    # Training phase
    for batch in train_dataloader:
        X_batch, Y_batch = batch

        
        # Reset the gradients of all optimized torch.Tensor
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, Y_batch)
        loss.backward()
        
        optimizer.step()

        total_trainloss += loss.item()


# ______________________________________________________________________________
# ______________________________________________________________________________
# ______________________  VALIDATION PHASE  ____________________________________
# ______________________________________________________________________________
# ______________________________________________________________________________



    
    if epoch % 1 == 0: 
        model.eval()
        total_val_loss = 0.0
        total_val_classifier_loss = 0.0


        with torch.no_grad():
            for batch in val_dataloader:
                X_batch, Y_batch = batch
                predictions = model(X_batch)

                loss = criterion(predictions, Y_batch)
                total_val_loss += loss.item()


    # Calculate average losses (average over batches)
    avg_trainloss = total_trainloss / len(train_dataloader)
    avg_val_loss = total_val_loss / len(val_dataloader)

    train_loss_history.append(avg_trainloss)
    val_loss_history.append(avg_val_loss)

    # Print losses
    print(f"Epoch [{epoch+1}/{epochs}], "
          f"Train Loss: {avg_trainloss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}",
          flush=(epoch % 50 == 0))


    # Early Stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0  # reset patience if validation loss improves
        best_model_weights = model.state_dict() # save the learnable model weights
        best_epoch= epoch
    else:
        patience_counter += 1

    if (patience_counter >= early_stopping_patience) & (epoch > early_stopping_patience):
        print("Early stopping triggered.")
        break

if best_model_weights is not None:
    model.load_state_dict(best_model_weights)
    print("Restored the best model weights.")



# %%
np.save(outFolder + "/train_loss_history.npy", train_loss_history)
np.save(outFolder + "/val_loss_history.npy", val_loss_history)

# %%
torch.save(model, outFolder+"/model.pth")
print("Model saved")
#with open(outFolder + "/training.txt", "w") as file:
#    for key, value in hp.items():
#        file.write(f"{key} : {value}\n")

# %%
Xtrain = unscale(Xtrain, Xtrain.columns, scalerName= outFolder + "/myScaler.pkl", log=False)
Xtest = unscale(Xtest, Xtest.columns, scalerName= outFolder + "/myScaler.pkl", log=False)
Xval = unscale(Xval, Xval.columns, scalerName= outFolder + "/myScaler.pkl", log=False)
# %%
import matplotlib.pyplot as plt
predictions_test = (model(Xtest_tensor)).detach().squeeze().numpy()
predictions_val = (model(Xval_tensor)).detach().squeeze().numpy()
bins = np.linspace(40, 200, 41)
fig, ax = plt.subplots(1, 1)
ax.hist(Xval.dijet_mass[genMassVal==125], bins=bins, histtype='step', label='Reco')
ax.hist(Xval.dijet_mass[genMassVal==125]*predictions_val[genMassVal==125], bins=bins, histtype='step', label='regressed')
ax.hist(Xval.genDijetNu_mass[genMassVal==125], bins=bins, histtype='step', label='Gen')
ax.legend()
print(np.std(np.clip(Xval.dijet_mass[genMassVal==125], 60, 180)))
print(np.std(np.clip(Xval.dijet_mass[genMassVal==125]*predictions_val[genMassVal==125], 60, 180)))

# %%
fig, ax = plt.subplots(1, 1)
ax.hist(Xval.dijet_mass[genMassVal==125], bins=bins, histtype='step', label='Reco')
ax.hist(predictions_val[genMassVal==125], bins=bins, histtype='step', label='regressed')
ax.hist(Xval.genDijetNu_mass[genMassVal==125], bins=bins, histtype='step', label='Gen')
ax.legend()
print(np.std(np.clip(Xval.dijet_mass[genMassVal==125], 60, 180)))
print(np.std(np.clip(predictions_val[genMassVal==125], 60, 180)))

# %%
