# %%
import numpy as np
import sys
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
sys.path.append("/t3home/gcelotto/ggHbb/PNN/helpers")
from datetime import datetime
import argparse

# PNN helpers
from helpers.getFeatures import getFeatures
from helpers.getParams import getParams
from helpers.loadSaved import loadXYrWSaved
from helpers.getInfolderOutfolder import getInfolderOutfolder
from helpers.scaleUnscale import scale
from helpers.dcorLoss import *

# Torch
import torch    
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
torch.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader, TensorDataset


# Get current month and day
current_date = datetime.now().strftime("%b%d")  # This gives the format like 'Dec12'
# %%
gpuFlag=True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define folder of input and output. Create the folders if not existing
hp = getParams()
try:
    parser = argparse.ArgumentParser(description="Script.")
    #### Define arguments
    parser.add_argument("-l", "--lambda_dcor", type=float, help="lambda for penalty term", default=None)
    parser.add_argument("-e", "--epochs", type=int, help="number of epochs", default=None)
    parser.add_argument("-s", "--size", type=int, help="Number of events to crop training dataset", default=int(1e9))
    parser.add_argument("-bs", "--batch_size", type=int, help="Number of eventsper batch size", default=int(25e3))
    parser.add_argument("-node", "--node", type=int, help="nodes of single layer in case of one layer for simple nn", default=None)
    args = parser.parse_args()
    if args.lambda_dcor is not None:
        hp["lambda_dcor"] = args.lambda_dcor 
        print("lambda_dcor changed to ", hp["lambda_dcor"])
    if args.epochs is not None:
        hp["epochs"] = args.epochs 
        print("N epochs to ", hp["epochs"])
    if args.batch_size is not None:
        hp["batch_size"] = int(args.batch_size )
        print("N batch_size to ", hp["batch_size"])
    if args.node is not None:
        hp["nNodes"] = [args.node]
except:
    print("-"*40)
    print("No arguments provided for lambda!")
    print("lambda_dcor changed to ", hp["lambda_dcor"])
    hp["lambda_dcor"] = 5. 
    print(hp)
    print("interactive mode")
    print("-"*40)
hp["size"]=args.size
size = args.size
print("Size is ", size)
# %%
bin_center_bool = True
inFolder, outFolder = getInfolderOutfolder(name = "%s_%s"%(current_date, str(hp["lambda_dcor"]).replace('.', 'p')), suffixResults="TripleDisco")
# Define features to read and to train the pNN (+parameter massHypo) and save the features for training in outfolder
featuresForTraining, columnsToRead = getFeatures(outFolder, massHypo=True, bin_center=bin_center_bool)

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
#print("Cutting dijet mass > 70")
#mTrain = Xtrain.dijet_mass>70
#mVal = Xval.dijet_mass>70
#
#Xtrain=Xtrain[mTrain]
#Ytrain=Ytrain[mTrain]
#rWtrain=rWtrain[mTrain]
#genMassTrain=genMassTrain[mTrain]
#Wtrain=Wtrain[mTrain]
#advFeatureTrain=advFeatureTrain[mTrain]
#dijetMassTrain=dijetMassTrain[mTrain]
#
#Xval=Xval[mVal]
#Yval=Yval[mVal]
#rWval=rWval[mVal]
#genMassVal=genMassVal[mVal]
#Wval=Wval[mVal]
#advFeatureVal=advFeatureVal[mVal]
#dijetMassVal=dijetMassVal[mVal]



print(len(Xtrain), " events in train dataset")
print(len(Xval), " events in val dataset")
# %%
mass_bins = np.quantile(Xtrain[Ytrain==0].dijet_mass.values, np.linspace(0, 1, 15))
mass_bins[0], mass_bins[-1] = 40., 300.
np.save(outFolder+"/mass_bins.npy", mass_bins)
if bin_center_bool:
    bin_centers = [(mass_bins[i] + mass_bins[i+1]) / 2 for i in range(len(mass_bins) - 1)]


    bin_indices = np.digitize(Xtrain['dijet_mass'].values, mass_bins) - 1
    Xtrain['bin_center'] = np.where(
        (bin_indices >= 0) & (bin_indices < len(bin_centers)),  # Ensure valid indices
        np.array(bin_centers)[bin_indices],
        np.nan  # Assign NaN for out-of-range dijet_mass
    )
    bin_indices = np.digitize(Xval['dijet_mass'].values, mass_bins) - 1
    Xval['bin_center'] = np.where(
        (bin_indices >= 0) & (bin_indices < len(bin_centers)),  # Ensure valid indices
        np.array(bin_centers)[bin_indices],
        np.nan  # Assign NaN for out-of-range dijet_mass
    )
# %%
# scale with standard scalers and apply log to any pt and mass distributions

Xtrain = scale(Xtrain,featuresForTraining,  scalerName= outFolder + "/model/myScaler.pkl" ,fit=True)
Xval  = scale(Xval, featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)
#advFeatureTrain = scale(pd.DataFrame(advFeatureTrain, columns=['jet1_btagDeepFlavB']),['jet1_btagDeepFlavB'],  scalerName= outFolder + "/model/myScaler_adv.pkl" ,fit=True)
#advFeatureVal  = scale(pd.DataFrame(advFeatureVal,columns=['jet1_btagDeepFlavB']), ['jet1_btagDeepFlavB'], scalerName= outFolder + "/model/myScaler_adv.pkl" ,fit=False)
# %%
#size = int(6e6)
Xtrain, Ytrain, Wtrain, rWtrain, genMassTrain, advFeatureTrain, dijetMassTrain = Xtrain[:size], Ytrain[:size], Wtrain[:size], rWtrain[:size], genMassTrain[:size], advFeatureTrain[:size], dijetMassTrain[:size]
Xval, Yval, Wval, rWval, genMassVal, advFeatureVal, dijetMassVal = Xval[:size], Yval[:size], Wval[:size], rWval[:size], genMassVal[:size], advFeatureVal[:size], dijetMassVal[:size]


# %%
# Comment if want to use flat in mjj
rWtrain, rWval = Wtrain.copy(), Wval.copy()
# %%

XtrainTensor = torch.tensor(Xtrain[featuresForTraining].values, dtype=torch.float32, device=device)
YtrainTensor = torch.tensor(Ytrain, dtype=torch.float, device=device).unsqueeze(1)
rWtrainTensor = torch.tensor(rWtrain, dtype=torch.float32, device=device).unsqueeze(1)
advFeatureTrain_tensor = torch.tensor(advFeatureTrain, dtype=torch.float32, device=device).unsqueeze(1)
dijetMassTrain_tensor = torch.tensor(dijetMassTrain, dtype=torch.float32, device=device).unsqueeze(1)


Xval_tensor = torch.tensor(Xval[featuresForTraining].values, dtype=torch.float32, device=device)
Yval_tensor = torch.tensor(Yval, dtype=torch.float, device=device).unsqueeze(1)
rWval_tensor = torch.tensor(rWval, dtype=torch.float32, device=device).unsqueeze(1)
advFeatureVal_tensor = torch.tensor(advFeatureVal, dtype=torch.float32, device=device).unsqueeze(1)
dijetMassVal_tensor = torch.tensor(dijetMassVal, dtype=torch.float32, device=device).unsqueeze(1)

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
hp["batch_size"] = hp["batch_size"] if hp["batch_size"]<size else size
#hp['bimod'] = 0
traindataloader = DataLoader(traindataset, batch_size=hp["batch_size"], shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=hp["batch_size"], shuffle=False, drop_last=True)
# %%
# Model, loss, optimizer
nn1 = Classifier(input_dim=Xtrain[featuresForTraining].shape[1], nNodes=hp["nNodes"])
nn2 = Classifier(input_dim=Xtrain[featuresForTraining].shape[1], nNodes=hp["nNodes"])
nn3 = Classifier(input_dim=Xtrain[featuresForTraining].shape[1], nNodes=hp["nNodes"])
nn1.to(device)
nn2.to(device)
nn3.to(device)

epochs = hp["epochs"]
criterion = nn.BCELoss(reduction='none')
optimizer1 = optim.Adam(nn1.parameters(), lr=hp["learning_rate"])
optimizer2 = optim.Adam(nn2.parameters(), lr=hp["learning_rate"])
optimizer3 = optim.Adam(nn3.parameters(), lr=hp["learning_rate"])

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


# %%


for epoch in range(hp["epochs"]):
    nn1.train()
    nn2.train()
    nn3.train()
    
    total_trainloss = 0.0
    total_trainclassifier_loss = 0.0
    total_traindcor_loss = 0.0
    # Training phase
    for batch in traindataloader:
        X_batch, Y_batch, W_batch, advFeature_batch, dijetMass_batch, mask_batch = batch

        
        # Reset the gradients of all optimized torch.Tensor
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()

        predictions1 = nn1(X_batch)
        predictions2 = nn2(X_batch)
        predictions3 = nn2(X_batch)

        raw_loss1 = criterion(predictions1, Y_batch)
        raw_loss2 = criterion(predictions2, Y_batch)
        raw_loss3 = criterion(predictions2, Y_batch)
        
        classifier_loss1 = (raw_loss1 * W_batch).mean()
        classifier_loss2 = (raw_loss2 * W_batch).mean()
        classifier_loss3 = (raw_loss2 * W_batch).mean()


        W_batch = torch.ones([len(W_batch), 1], device=device)
        dCorr_total = 0.
        #dCorr_list = [] 
        for low, high in zip(mass_bins[:-1], mass_bins[1:]):
            # Mask for the current mass bin
            bin_mask = (dijetMass_batch >= low) & (dijetMass_batch < high) & (mask_batch)

            if (bin_mask.sum().item()==0):
                print("No elements in this bin!")
                continue
            bin_predictions1 = predictions1[bin_mask]
            bin_predictions2 = predictions2[bin_mask]
            bin_predictions3 = predictions3[bin_mask]
            
            bin_advFeature = advFeature_batch[bin_mask]
            bin_weights = W_batch[bin_mask]

            # Skip if there are no examples in the bin
            #if bin_predictions1.numel() == 0:
            #    #print("skipping")
            #    continue
            
            # Compute dCorr for the current mass bin
            dCorr_bin_12 = distance_corr(bin_predictions1, bin_predictions2, bin_weights)
            dCorr_bin_13 = distance_corr(bin_predictions1, bin_predictions3, bin_weights)
            dCorr_bin_23 = distance_corr(bin_predictions2, bin_predictions3, bin_weights)

            dCorr_total += dCorr_bin_12
            dCorr_total += dCorr_bin_13
            dCorr_total += dCorr_bin_23
            #dCorr_list.append(dCorr_bin)
        #if len(dCorr_list) > 1:  # Ensure there is more than one bin with values
        #    dCorr_tensor = torch.tensor(dCorr_list, dtype=torch.float32, device=device)
        #    variance_dCorr = torch.var(dCorr_tensor)
        #else:
        #    variance_dCorr = torch.tensor(0.0, dtype=torch.float32, device=device)  # Handle edge case
        
        #bimod_loss = bimodality(predictions1[mask_batch]) + bimodality(predictions2[mask_batch])
        # Combined loss
        loss = classifier_loss1 +classifier_loss2 + classifier_loss3 + hp["lambda_dcor"] * dCorr_total/(len(mass_bins) -1) #+ hp["lambda_var"] * variance_dCorr#+ hp['bimod']*bimod_loss
        loss.backward()
        
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()

        total_trainloss += loss.item()
        total_trainclassifier_loss += classifier_loss1.item() + classifier_loss2.item() + classifier_loss1.item()
        total_traindcor_loss += dCorr_total.item()/(len(mass_bins) -1)


# ______________________________________________________________________________
# ______________________________________________________________________________
# ______________________  VALIDATION PHASE  ____________________________________
# ______________________________________________________________________________
# ______________________________________________________________________________



    
    if epoch % 1 == 0: 
        nn1.eval()
        nn2.eval()
        nn3.eval()
        total_val_loss = 0.0
        total_val_classifier_loss = 0.0
        total_val_dcor_loss = 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                X_batch, Y_batch, W_batch, advFeature_batch, dijetMass_batch, mask_batch = batch
                predictions1 = nn1(X_batch)
                predictions2 = nn2(X_batch)
                predictions3 = nn3(X_batch)

                raw_loss1 = criterion(predictions1, Y_batch)
                raw_loss2 = criterion(predictions2, Y_batch)
                raw_loss3 = criterion(predictions3, Y_batch)

            # Apply weights manually
                classifier_loss1 = (raw_loss1 * W_batch).mean()
                classifier_loss2 = (raw_loss2 * W_batch).mean()
                classifier_loss3 = (raw_loss3 * W_batch).mean()

                # If there are any remaining entries after filtering, calculate dcor
                W_batch = torch.ones([len(W_batch), 1], device=device)
                dCorr_total = 0.
                #dCorr_list = []
                for low, high in zip(mass_bins[:-1], mass_bins[1:]):
                    # Mask for the current mass bin
                    bin_mask = (dijetMass_batch >= low) & (dijetMass_batch < high) & (mask_batch)

                    # Apply bin-specific mask
                    bin_predictions1 = predictions1[bin_mask]
                    bin_predictions2 = predictions2[bin_mask]
                    bin_predictions3 = predictions3[bin_mask]

                    bin_advFeature = advFeature_batch[bin_mask]
                    bin_weights = W_batch[bin_mask]

                    # Skip if there are no examples in the bin
                    #if bin_predictions.numel() == 0:
                    #    continue
                    
                    # Compute dCorr for the current mass bin
                    dCorr_bin_12 = distance_corr(bin_predictions1, bin_predictions2, bin_weights)
                    dCorr_bin_13 = distance_corr(bin_predictions1, bin_predictions3, bin_weights)
                    dCorr_bin_23 = distance_corr(bin_predictions2, bin_predictions3, bin_weights)
                    
                    dCorr_total += dCorr_bin_12
                    dCorr_total += dCorr_bin_13
                    dCorr_total += dCorr_bin_23
                    #dCorr_list.append(dCorr_bin)
                # Combined loss
                #if len(dCorr_list) > 1:  # Ensure there is more than one bin with values
                #    dCorr_tensor = torch.tensor(dCorr_list, dtype=torch.float32, device=device)
                #    variance_dCorr = torch.var(dCorr_tensor)
                #else:
                #    variance_dCorr = torch.tensor(0.0, dtype=torch.float32, device=device)  # Handle edge case

                #bimod_loss = bimodality(predictions1[mask_batch]) + bimodality(predictions2[mask_batch])
                loss = classifier_loss1 + classifier_loss2 + classifier_loss3 +  hp["lambda_dcor"] * dCorr_total/(len(mass_bins) -1) #+ hp["lambda_var"] * variance_dCorr#+ bimod_loss * hp['bimod']
                total_val_loss += loss.item()
                total_val_classifier_loss += classifier_loss1.item() + classifier_loss2.item()
                total_val_dcor_loss += dCorr_total.item()/(len(mass_bins) -1)


    # Calculate average losses (average over batches)
    avg_trainloss = total_trainloss / len(traindataloader)
    avg_train_classifier_loss = total_trainclassifier_loss / len(traindataloader)
    avg_traindcor_loss = total_traindcor_loss / len(traindataloader)

    avg_val_loss = total_val_loss / len(val_dataloader)
    avg_val_classifier_loss = total_val_classifier_loss / len(val_dataloader)
    avg_val_dcor_loss = total_val_dcor_loss / len(val_dataloader)

    trainloss_history.append(avg_trainloss)
    trainclassifier_loss_history.append(avg_train_classifier_loss)
    traindcor_loss_history.append(avg_traindcor_loss)
    val_loss_history.append(avg_val_loss)
    val_classifier_loss_history.append(avg_val_classifier_loss)
    val_dcor_loss_history.append(avg_val_dcor_loss)

    # Print losses
    print(f"Epoch [{epoch+1}/{epochs}], "
          f"Train Loss: {avg_trainloss:.4f}, Classifier Loss: {avg_train_classifier_loss:.4f}, dCor Loss: {avg_traindcor_loss:.8f}, "
          f"Val Loss: {avg_val_loss:.4f}, Val Classifier Loss: {avg_val_classifier_loss:.4f}, Val dCor Loss: {avg_val_dcor_loss:.8f}",
          flush=(epoch % 50 == 0))

    # Early Stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0  # reset patience if validation loss improves
        best_model_weights1 = nn1.state_dict()
        best_model_weights2 = nn2.state_dict()
        best_model_weights3 = nn3.state_dict()
        best_epoch= epoch
    else:
        patience_counter += 1

    if patience_counter >= early_stopping_patience:
        print("Early stopping triggered.")
        break

if best_model_weights is not None:
    nn1.load_state_dict(best_model_weights1)
    nn2.load_state_dict(best_model_weights2)
    nn3.load_state_dict(best_model_weights2)
    print("Restored the best model weights.")



# %%
np.save(outFolder + "/model/trainloss_history.npy", trainloss_history)
np.save(outFolder + "/model/val_loss_history.npy", val_loss_history)
np.save(outFolder + "/model/trainclassifier_loss_history.npy", trainclassifier_loss_history)
np.save(outFolder + "/model/val_classifier_loss_history.npy", val_classifier_loss_history)
np.save(outFolder + "/model/traindcor_loss_history.npy", traindcor_loss_history)
np.save(outFolder + "/model/val_dcor_loss_history.npy", val_dcor_loss_history)

# %%
torch.save(nn1, outFolder+"/model/nn1.pth")
torch.save(nn2, outFolder+"/model/nn2.pth")
torch.save(nn3, outFolder+"/model/nn3.pth")
with open(outFolder + "/model/training.txt", "w") as file:
    for key, value in hp.items():
        file.write(f"{key} : {value}\n")

print("Model saved")
# %%
#for param in model.parameters():
        #    print("Gradient norm:", param.grad.norm().item())