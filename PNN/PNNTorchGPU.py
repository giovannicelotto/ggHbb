# %%
# what about splitting in advance the batches in advance so that each batch is a bin of dijet mass
# 50, 10, 20, 5, 0, 1, 2 are done in one bin
# they are also done with gradient clipping (clipping was removed for lambda = 50)
# torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2) to be put after backward of the loss before the step taken
# 50k events for train and val.
# 25k events per batch
# lr set to 1e-4


# 51, 11, 21, 5.1, 0.01, 1.01, 2.01 are done in 3 bins
# gradient clipping was removed
# 50k events for train and val.
# 50k events per batch (single batch)
# lr set to 5e-4
import numpy as np
import logging
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
sys.path.append("/t3home/gcelotto/ggHbb/PNN/helpers")
from datetime import datetime

# Get current month and day
current_date = datetime.now().strftime("%b%d")  # This gives the format like 'Dec12'

# PNN helpers
from helpers.getFeatures import getFeatures
from helpers.getParams import getParams
from helpers.loadSaved import loadXYrWSaved
from helpers.getInfolderOutfolder import getInfolderOutfolder
from helpers.scaleUnscale import scale, unscale
from helpers.dcorLoss import *

# Torch
import torch    
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
torch.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader, TensorDataset
import argparse

# %%
gpuFlag=True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define folder of input and output. Create the folders if not existing
hp = getParams()
try:
    parser = argparse.ArgumentParser(description="Script.")
    #### Define arguments
    parser.add_argument("-l", "--lambdaCor", type=float, help="lambda for penalty term", default=None)
    parser.add_argument("-e", "--epochs", type=int, help="number of epochs", default=None)
    args = parser.parse_args()
    if args.lambdaCor is not None:
        hp["lambda_reg"] = args.lambdaCor 
        print("lambda_reg changed to ", hp["lambda_reg"])
    if args.epochs is not None:
        hp["epochs"] = args.epochs 
        print("N epochs to ", hp["epochs"])
except:
    print("-"*40)
    print("No arguments provided for lambda!")
    print("lambda_reg changed to ", hp["lambda_reg"])
    hp["lambda_reg"] = 5. 
    print(hp)
    print("interactive mode")
    print("-"*40)
# %%
inFolder, outFolder = getInfolderOutfolder(name = "%s_%s"%(current_date, str(hp["lambda_reg"]).replace('.', 'p')))
# Define features to read and to train the pNN (+parameter massHypo) and save the features for training in outfolder
featuresForTraining, columnsToRead = getFeatures(outFolder, massHypo=True)

# %%
# define the parameters for the nn

print("Before loading data")
# load data for the samples and preprocess the data(pT cut)
# fill the massHypo column
# cut the data to have same length in all the samples
# reweight each sample to have total weight 1, shuffle and split in train and test
Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, rWtrain, rWval, Wtest, genMassTrain, genMassVal, genMassTest = loadXYrWSaved(inFolder=inFolder+"/data")
Wtrain, Wval = np.load(inFolder+"/data/Wtrain.npy"), np.load(inFolder+"/data/Wval.npy")
dijetMassTrain = np.array(Xtrain.dijet_mass.values)
dijetMassVal = np.array(Xval.dijet_mass.values)
advFeatureTrain = np.load(inFolder+"/data/advFeatureTrain.npy")     
advFeatureVal   = np.load(inFolder+"/data/advFeatureVal.npy")
print(len(Xtrain), " events in train dataset")
print(len(Xval), " events in val dataset")
# %%
#import dcor
#dcor_results = {}
#for column in Xtrain[featuresForTraining].columns:
#    print(column)
#    dcor_value = dcor.distance_correlation(advFeatureTrain[Ytrain==0], Xtrain[Ytrain==0][column])
#    dcor_results[column] = dcor_value
#
## Print the results
#print(dcor_results)
# %%
# scale with standard scalers and apply log to any pt and mass distributions

Xtrain = scale(Xtrain,featuresForTraining,  scalerName= outFolder + "/model/myScaler.pkl" ,fit=True)
Xval  = scale(Xval, featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)
#advFeatureTrain = scale(pd.DataFrame(advFeatureTrain, columns=['jet1_btagDeepFlavB']),['jet1_btagDeepFlavB'],  scalerName= outFolder + "/model/myScaler_adv.pkl" ,fit=True)
#advFeatureVal  = scale(pd.DataFrame(advFeatureVal,columns=['jet1_btagDeepFlavB']), ['jet1_btagDeepFlavB'], scalerName= outFolder + "/model/myScaler_adv.pkl" ,fit=False)
# %%
size = 25000
Xtrain, Ytrain, Wtrain, rWtrain, genMassTrain, advFeatureTrain, dijetMassTrain = Xtrain[:size], Ytrain[:size], Wtrain[:size], rWtrain[:size], genMassTrain[:size], advFeatureTrain[:size], dijetMassTrain[:size]
Xval, Yval, Wval, rWval, genMassVal, advFeatureVal, dijetMassVal = Xval[:size], Yval[:size], Wval[:size], rWval[:size], genMassVal[:size], advFeatureVal[:size], dijetMassVal[:size]
# %%
#import matplotlib.pyplot as plt
#from hist import Hist
#fig,ax = plt.subplots(1, 1)
#bins = np.linspace(40, 300, 41)
#hD = Hist.new.Reg(41, 40, 300, name="mjj", label="mjj").Weight() 
#hS = Hist.new.Reg(41, 40, 300, name="mjj", label="mjj").Weight() 
#
#hD.fill(Xtrain[Ytrain==0].dijet_mass, weight=rWtrain[Ytrain==0])
#hS.fill(Xtrain[Ytrain==1].dijet_mass, weight=rWtrain[Ytrain==1])
##hS.fill(Xtrain[Ytrain==1].dijet_mass, weight=Wtrain[Ytrain==1])
#
#hD.plot(ax=ax, label='Data ')
#hS.plot(ax=ax, linewidth=1, label='Signal')
#ax.legend()
#ax.text(x=0.9, y=0.2, ha='right', s="data W: %d"%(Wtrain[Ytrain==0].sum()), transform=ax.transAxes)
#ax.text(x=0.9, y=0.1, ha='right', s="signal W: %d"%(Wtrain[Ytrain==1].sum()), transform=ax.transAxes)
#print(Wtrain.mean())

# %%
# Comment if want to use flat in mjj
rWtrain, rWval = Wtrain.copy(), Wval.copy()
# %%

XtrainTensor = torch.tensor(Xtrain[featuresForTraining].values, dtype=torch.float32, device=device)
YtrainTensor = torch.tensor(Ytrain, dtype=torch.float, device=device).unsqueeze(1)
rWtrainTensor = torch.tensor(rWtrain, dtype=torch.float32, device=device).unsqueeze(1)
advFeatureTrain_tensor = torch.tensor(advFeatureTrain, dtype=torch.float32, device=device)
dijetMassTrain_tensor = torch.tensor(dijetMassTrain, dtype=torch.float32, device=device)


Xval_tensor = torch.tensor(Xval[featuresForTraining].values, dtype=torch.float32, device=device)
Yval_tensor = torch.tensor(Yval, dtype=torch.float, device=device).unsqueeze(1)
rWval_tensor = torch.tensor(rWval, dtype=torch.float32, device=device).unsqueeze(1)
advFeatureVal_tensor = torch.tensor(advFeatureVal, dtype=torch.float32, device=device)
dijetMassVal_tensor = torch.tensor(dijetMassVal, dtype=torch.float32, device=device)

#Xtest_tensor = torch.tensor(np.float32(Xtest[featuresForTraining].values)).float()

train_masks = (YtrainTensor < 0.5).to(device)
val_masks = (Yval_tensor < 0.5).to(device)


#traindataset = TensorDataset(XtrainTensor, YtrainTensor, rWtrainTensor, advFeatureTrain_tensor)
#val_dataset = TensorDataset(Xval_tensor, Yval_tensor, rWval_tensor, advFeatureVal_tensor)
traindataset = TensorDataset(
    XtrainTensor.to(device),
    YtrainTensor.to(device),
    rWtrainTensor.to(device),
    advFeatureTrain_tensor.to(device),
    dijetMassTrain_tensor.to(device),
    train_masks.to(device)
)
val_dataset = TensorDataset(
    Xval_tensor.to(device),
    Yval_tensor.to(device),
    rWval_tensor.to(device),
    advFeatureVal_tensor.to(device),
    dijetMassVal_tensor.to(device),
    val_masks.to(device)
)
# Drop last to drop the last (if incomplete size) batch
traindataloader = DataLoader(traindataset, batch_size=hp["batch_size"], shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=hp["batch_size"], shuffle=False, drop_last=True)
# %%
# Model, loss, optimizer
model = Classifier(input_dim=Xtrain[featuresForTraining].shape[1], nNodes=hp["nNodes"])
model.to(device)
epochs = hp["epochs"]
criterion = nn.BCELoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=hp["learning_rate"])

early_stopping_patience = hp["patienceES"]
best_val_loss = float('inf')
patience_counter = 0
print("Train start")

# ______________________________________________________________________________
# ______________________________________________________________________________
# ______________________  TRAINING PHASE  ______________________________________
# ______________________________________________________________________________
# ______________________________________________________________________________



trainloss_history = []
trainclassifier_loss_history = []
traindcor_loss_history = []

val_loss_history = []
val_classifier_loss_history = []
val_dcor_loss_history = []
best_model_weights = None # weights saved for RestoreBestWeights
best_epoch = None

mass_bins = np.linspace(40, 300, 4)

for epoch in range(hp["epochs"]):
    model.train()
    total_trainloss = 0.0
    total_trainclassifier_loss = 0.0
    total_traindcor_loss = 0.0
    # Training phase
    for batch in traindataloader:
        X_batch, Y_batch, W_batch, advFeature_batch, dijetMass_batch, mask_batch = batch

        
        # Reset the gradients of all optimized torch.Tensor
        optimizer.zero_grad()
        predictions = model(X_batch).squeeze().unsqueeze(1)
        raw_loss = criterion(predictions, Y_batch)
        # Apply weights manually
        classifier_loss = (raw_loss * W_batch.unsqueeze(1)).mean()


        W_batch = torch.ones(len(W_batch), device=device).unsqueeze(1)
        dCorr_total = 0.
        for low, high in zip(mass_bins[:-1], mass_bins[1:]):
            # Mask for the current mass bin
            bin_mask = (dijetMass_batch.unsqueeze(1) >= low) & (dijetMass_batch.unsqueeze(1) < high) & (mask_batch)

            if (bin_mask.sum().item()==0):
                print("No elements in this bin!")
                continue
            bin_predictions = predictions[bin_mask]
            bin_advFeature = advFeature_batch[bin_mask.squeeze()]
            bin_weights = W_batch[bin_mask]

            # Skip if there are no examples in the bin
            if bin_predictions.numel() == 0:
                #print("skipping")
                continue
            
            # Compute dCorr for the current mass bin
            dCorr_bin = distance_corr(bin_predictions, bin_advFeature, bin_weights)

            dCorr_total += dCorr_bin 
        
        
        # Combined loss
        loss = classifier_loss + hp["lambda_reg"] * dCorr_total/(len(mass_bins) -1)
        loss.backward()
        
        optimizer.step()

        total_trainloss += loss.item()
        total_trainclassifier_loss += classifier_loss.item()
        total_traindcor_loss += dCorr_total.item()/(len(mass_bins) -1)


# ______________________________________________________________________________
# ______________________________________________________________________________
# ______________________  VALIDATION PHASE  ____________________________________
# ______________________________________________________________________________
# ______________________________________________________________________________



    
    if epoch % 1 == 0: 
        model.eval()
        total_val_loss = 0.0
        total_val_classifier_loss = 0.0
        total_val_dcor_loss = 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                X_batch, Y_batch, W_batch, advFeature_batch, dijetMass_batch, mask_batch = batch
                predictions = model(X_batch).squeeze().unsqueeze(1)

                raw_loss = criterion(predictions, Y_batch)
            # Apply weights manually
                classifier_loss = (raw_loss * W_batch.unsqueeze(1)).mean()

                # If there are any remaining entries after filtering, calculate dcor
                W_batch = torch.ones(len(W_batch), device=device).unsqueeze(1)
                dCorr_total = 0.
                for low, high in zip(mass_bins[:-1], mass_bins[1:]):
                    # Mask for the current mass bin
                    bin_mask = (dijetMass_batch.unsqueeze(1) >= low) & (dijetMass_batch.unsqueeze(1) < high) & (mask_batch)

                    # Apply bin-specific mask
                    bin_predictions = predictions[bin_mask]
                    bin_advFeature = advFeature_batch[bin_mask.squeeze()]
                    bin_weights = W_batch[bin_mask]

                    # Skip if there are no examples in the bin
                    if bin_predictions.numel() == 0:
                        continue
                    
                    # Compute dCorr for the current mass bin
                    dCorr_bin = distance_corr(bin_predictions, bin_advFeature, bin_weights)
                    dCorr_total += dCorr_bin
                # Combined loss
                loss = classifier_loss + hp["lambda_reg"] * dCorr_total/(len(mass_bins) -1)
                total_val_loss += loss.item()
                total_val_classifier_loss += classifier_loss.item()
                total_val_dcor_loss += dCorr_total.item()/(len(mass_bins) -1)


    # Calculate average losses (average over batches)
    avg_trainloss = total_trainloss / len(traindataloader)
    avg_trainclassifier_loss = total_trainclassifier_loss / len(traindataloader)
    avg_traindcor_loss = total_traindcor_loss / len(traindataloader)

    avg_val_loss = total_val_loss / len(val_dataloader)
    avg_val_classifier_loss = total_val_classifier_loss / len(val_dataloader)
    avg_val_dcor_loss = total_val_dcor_loss / len(val_dataloader)

    trainloss_history.append(avg_trainloss)
    trainclassifier_loss_history.append(avg_trainclassifier_loss)
    traindcor_loss_history.append(avg_traindcor_loss)
    val_loss_history.append(avg_val_loss)
    val_classifier_loss_history.append(avg_val_classifier_loss)
    val_dcor_loss_history.append(avg_val_dcor_loss)

    # Print losses
    print(f"Epoch [{epoch+1}/{epochs}], "
          f"Train Loss: {avg_trainloss:.4f}, Classifier Loss: {avg_trainclassifier_loss:.4f}, dCor Loss: {avg_traindcor_loss:.8f}, "
          f"Val Loss: {avg_val_loss:.4f}, Val Classifier Loss: {avg_val_classifier_loss:.4f}, Val dCor Loss: {avg_val_dcor_loss:.8f}",
          flush=(epoch % 50 == 0))

    # Early Stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0  # reset patience if validation loss improves
        best_model_weights = model.state_dict()
        best_epoch= epoch
    else:
        patience_counter += 1

    if patience_counter >= early_stopping_patience:
        print("Early stopping triggered.")
        break

if best_model_weights is not None:
    model.load_state_dict(best_model_weights)
    print("Restored the best model weights.")



# %%
np.save(outFolder + "/model/trainloss_history.npy", trainloss_history)
np.save(outFolder + "/model/val_loss_history.npy", val_loss_history)
np.save(outFolder + "/model/trainclassifier_loss_history.npy", trainclassifier_loss_history)
np.save(outFolder + "/model/val_classifier_loss_history.npy", val_classifier_loss_history)
np.save(outFolder + "/model/traindcor_loss_history.npy", traindcor_loss_history)
np.save(outFolder + "/model/val_dcor_loss_history.npy", val_dcor_loss_history)

# %%
torch.save(model, outFolder+"/model/model.pth")
print("Model saved")
# %%
#for param in model.parameters():
        #    print("Gradient norm:", param.grad.norm().item())