# %%
# add dPhi, dEta as edge features
# add btagWP as node feature
# understand GINECONV
# start preparing p4s 
import numpy as np
import pandas as pd
import sys
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.dcorLoss import distance_corr
from helpers.scaleUnscale import scale, unscale
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch
import os
import argparse
from GNN_class import GNN, JetGraphDataset, GNN_3j1m, GNN_3j1m_hetero, GNN_3j1m_hetero_hetero, myGAT
#%%
from torch_geometric.data import InMemoryDataset, Data, DataLoader
torch.backends.cudnn.benchmark = True
gpuFlag=True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is ", device)
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.device_count())

num_cores = os.cpu_count()
print(f"Number of CPU cores available: {num_cores}")
parser = argparse.ArgumentParser(description="Script.")
#### Define arguments
parser.add_argument("-btagWP", "--btagWP", type=str, help="btagWP", default="M")
parser.add_argument("-v", "--modelVersion", type=int, help="modelVersion", default=1)
parser.add_argument("-l", "--lambda_disco", type=int, help="lambda_disco", default=20)
parser.add_argument("-c", "--coderDimension", type=int, help="coderDimension", default=8)
if 'ipykernel' in sys.modules:
    args = parser.parse_args(args=[])  # Avoids errors when running interactively
else:
    args = parser.parse_args()



folder = f"/work/gcelotto/GNN/model_{args.lambda_disco}_v{args.modelVersion}"
while os.path.exists(folder):
    args.modelVersion += 1
    folder = f"/work/gcelotto/GNN/model_{args.lambda_disco}_v{args.modelVersion}"
print("Creating folder: ", folder)
os.makedirs(folder)
os.makedirs(folder+"/losses_history")
os.makedirs(folder+"/plots")
os.makedirs(folder+"/GNN_weights")
args.modelVersion = os.path.basename(folder)


# %%
N = 1024*1024*8
dataset_train = torch.load("/work/gcelotto/GNN/graphs_train_hetero_hetero.pt")[:N]
dataset_val = torch.load("/work/gcelotto/GNN/graphs_val_hetero_hetero.pt")[:N]
Ytrain = np.load("/work/gcelotto/ggHbb_work/input_NN/data_pt3_1D/Ytrain_M.npy")[:N]
genMassTrain = np.load("/work/gcelotto/ggHbb_work/input_NN/data_pt3_1D/genMassTrain_M.npy")[:N]
rWtrain = np.load("/work/gcelotto/ggHbb_work/input_NN/data_pt3_1D/rWtrain_M.npy")[:N]
rWval = np.load("/work/gcelotto/ggHbb_work/input_NN/data_pt3_1D/rWval_M.npy")[:N]
Xtrain = pd.read_parquet("/work/gcelotto/ggHbb_work/input_NN/data_pt3_1D/Xtrain_M.parquet").iloc[:N,:]
Xval = pd.read_parquet("/work/gcelotto/ggHbb_work/input_NN/data_pt3_1D/Xval_M.parquet").iloc[:N,:]
Wtrain = np.load("/work/gcelotto/ggHbb_work/input_NN/data_pt3_1D/Wtrain_M.npy")[:N]
Wval = np.load("/work/gcelotto/ggHbb_work/input_NN/data_pt3_1D/Wval_M.npy")[:N]
genMassVal = np.load("/work/gcelotto/ggHbb_work/input_NN/data_pt3_1D/genMassVal_M.npy")[:N]
Yval = np.load("/work/gcelotto/ggHbb_work/input_NN/data_pt3_1D/Yval_M.npy")[:N]
print("Loaded datasets with %d training and %d validation graphs" % (len(dataset_train), len(dataset_val)))
for i, data in enumerate(dataset_train):
    data.y = torch.tensor(Ytrain[i], dtype=torch.long)
    data.w = torch.tensor(rWtrain[i], dtype=torch.float)
    #data.mjj = torch.tensor(Xtrain.dijet_mass.values[i], dtype=torch.float)

for i, data in enumerate(dataset_val):
    data.y = torch.tensor(Yval[i], dtype=torch.long)
    data.w = torch.tensor(rWval[i], dtype=torch.float)
    #data.mjj = torch.tensor(Xval[i].dijet_mass.values[i], dtype=torch.float)
    
# %%
print("Preparing datasets...")
loader = DataLoader(dataset_train, batch_size=8192, shuffle=True, num_workers=4, pin_memory=True)
loader_val = DataLoader(dataset_val, batch_size=8192, shuffle=True, num_workers=4, pin_memory=True)
# %%

# %%
#device = "cuda" if torch.cuda.is_available() else "cpu"
#print("Moving model to device...")
model = myGAT(coderDimension=args.coderDimension)#.to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total trainable parameters:", total_params, flush=True)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
criterion = nn.BCEWithLogitsLoss(reduction='none')
loss_total_history_train = []
loss_classifier_history_train = []
loss_disco_history_train = []
loss_history_val = []
loss_total_history_val = []
loss_classifier_history_val = []
loss_disco_history_val = []
lambda_disco = args.lambda_disco
# %%

num_epochs=200
best_val_loss = float('inf')
best_model_weights = None
patience_counter = 0
early_stopping_patience = 150
print("Training Start")
#import time
for epoch in range(num_epochs):
    model.train()
    epoch_train_total_loss = 0
    epoch_train_classifier_loss = 0.0
    epoch_train_disco_loss = 0.0
    
    
    num_batches = 0

    #start = time.time()
    for batch in loader:
        #batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch).reshape(-1)

        y = batch.y 
        bkg_mask = (y == 0)

        raw_loss = criterion(out, batch.y)
        classifier_loss = (raw_loss * batch.weight).mean()

        if lambda_disco>=1e-1:
            W_batch = torch.ones([len(y)])
            dCorr = distance_corr(out[bkg_mask], batch.mjj[bkg_mask], W_batch[bkg_mask])
            loss = classifier_loss + lambda_disco*dCorr
        else:
            loss = classifier_loss
        loss.backward()
        optimizer.step()
        epoch_train_classifier_loss += classifier_loss.item()
        if lambda_disco>=1e-1:
            epoch_train_disco_loss += lambda_disco*dCorr.item()

        epoch_train_total_loss += loss.item()  # sum losses
        num_batches += 1
        #stop = time.time()
        #print("Training batch %d took %.2f seconds" % (idx, stop - start), flush=True)
    #print("Train done. Validation start.")
    avg_train_loss = epoch_train_total_loss / num_batches  # mean over batches
    avg_train_classifier_loss = epoch_train_classifier_loss / num_batches  # mean over batches
    avg_train_disco_loss = epoch_train_disco_loss / num_batches  # mean over batches
    loss_total_history_train.append(avg_train_loss)
    loss_classifier_history_train.append(avg_train_classifier_loss)
    loss_disco_history_train.append(avg_train_disco_loss)
    
    
    epoch_val_total_loss = 0
    epoch_val_classifier_loss = 0.0
    epoch_val_disco_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader_val:
            #batch = batch.to(device)
            y = batch.y 
            bkg_mask = (y == 0)
            out = model(batch).reshape(-1)

            raw_loss = criterion(out, batch.y)
            classifier_loss = (raw_loss * batch.weight).mean()

            if lambda_disco>=1e-1:
                W_batch = torch.ones([len(y)])
                dCorr = distance_corr(out[bkg_mask], batch.mjj[bkg_mask], W_batch[bkg_mask])
                loss = classifier_loss + lambda_disco*dCorr
            else:
                loss = classifier_loss

            epoch_val_classifier_loss += classifier_loss.item()
            if lambda_disco>=1e-1:
                epoch_val_disco_loss += lambda_disco*dCorr.item()
            epoch_val_total_loss += loss.item()  # sum losses
            num_batches += 1

    avg_val_loss = epoch_val_total_loss / num_batches  # mean over batches
    avg_val_classifier_loss = epoch_val_classifier_loss / num_batches  # mean over batches
    avg_val_disco_loss = epoch_val_disco_loss / num_batches  # mean over batches
    loss_total_history_val.append(avg_val_loss)
    loss_classifier_history_val.append(avg_val_classifier_loss)
    loss_disco_history_val.append(avg_val_disco_loss)
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0  # reset patience if validation loss improves
        best_model_weights = model.state_dict()
        torch.save(model.state_dict(), folder+"/GNN_weights/best_model_e%d.pt"%(epoch))
    else:
        patience_counter += 1

    if (patience_counter >= early_stopping_patience) & (epoch > early_stopping_patience):
        print("Early stopping triggered.")
        break
    flush = True if epoch % 1 == 0 else False
    print(f"Epoch {epoch+1} | Train: {avg_train_loss:.3f} Class: {avg_train_classifier_loss:3f} DisCo: {avg_train_disco_loss:3f}| Val: {avg_val_loss:3f} Class: {avg_val_classifier_loss:3f} DisCo: {avg_val_disco_loss:3f}", flush=flush)
if best_model_weights is not None:
    print("Restoring best mode weights from epoch with val loss: ", best_val_loss)
    model.load_state_dict(best_model_weights)


import os
save_path = os.path.join(folder, f"GNN_weights/gnn.pt")
torch.save(model.state_dict(), save_path)

print(f"Saved model to {save_path}")


# %%

np.save(folder+"/losses_history/loss_total_train", loss_total_history_train)
np.save(folder+"/losses_history/loss_classifier_train", loss_classifier_history_train)
np.save(folder+"/losses_history/loss_disco_train", loss_disco_history_train)
np.save(folder+"/losses_history/loss_total_val", loss_total_history_val)
np.save(folder+"/losses_history/loss_classifier_val", loss_classifier_history_val)
np.save(folder+"/losses_history/loss_disco_val", loss_disco_history_val)

# %%
