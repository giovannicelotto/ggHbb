# %%
import numpy as np
import sys
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from datetime import datetime
import yaml
# Get current month and day
current_date = datetime.now().strftime("%b%d")  # This gives the format like 'Dec12'

# PNN helpers
#from helpers.getFeatures import getFeatures, getFeaturesHighPt
from helpers.getParams import getParams
from helpers.loadSaved import loadXYWrWSaved
from helpers.getInfolderOutfolder import getInfolderOutfolder
from helpers.scaleUnscale import scale, unscale
from helpers.dcorLoss import *
import torch
import torch.nn.functional as F


# Torch
import torch    
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
torch.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader, TensorDataset
import argparse

# %%

torch.backends.cudnn.benchmark = True
gpuFlag=True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is ", device)
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.device_count())
import os

num_cores = os.cpu_count()
print(f"Number of CPU cores available: {num_cores}")
# Define folder of input and output. Create the folders if not existing
# %%

parser = argparse.ArgumentParser(description="Script.")
#### Define arguments
parser.add_argument("-l", "--lambda_disco", type=float, help="lambda value for disco", default=20.)
parser.add_argument("-e", "--epochs", type=int, help="number of epochs", default=500)
parser.add_argument("-s", "--size", type=int, help="Number of events to crop training dataset", default=9999999999)
parser.add_argument("-b", "--boosted", type=int, help="Boosted Class", default=12)
parser.add_argument("-btagWP", "--btagWP", type=str, help="btagWP", default="M")
parser.add_argument("-d", "--dropout", type=float, help="dropout prob", default=0.)
parser.add_argument("-lr", "--learningRate", type=float, help="learning rate", default=1e-3)
parser.add_argument("-bs", "--batch_size", type=int, help="Number of events perBatch", default=8192)
parser.add_argument("-eS", "--earlyStopping", type=int, help="patience early stop", default=20)
parser.add_argument("-nSV", "--nodesSV", type=int, help="nodes fro mlp used for sv", default=8)
parser.add_argument("-n", "--nodes",type=lambda s: [int(item) for item in s.split(',')],  # Convert comma-separated string to list of ints
                        help="List of nodes per layer (e.g., 128,64,32 for a 3-layer NN)",default="64,32"#""
)

if 'ipykernel' in sys.modules:
    args = parser.parse_args(args=[])  # Avoids errors when running interactively
else:
    args = parser.parse_args()

# %%

inFolder, outFolder = getInfolderOutfolder(name = "%s_%d_%s"%(current_date, args.boosted, str(args.lambda_disco).replace('.', 'p') if args.lambda_disco>0 else str(args.lambda_shape).replace('.', 'p')), suffixResults='_mjjDisco/gateSV_jet3_muon2')
inFolder = f"/work/gcelotto/ggHbb_work/input_NN/data_pt{args.boosted}_1D"
# Define features to read and to train the pNN (+parameter massHypo) and save the features for training in outfolder
#featuresForTraining, columnsToRead = getFeaturesHighPt(outFolder)
if args.boosted>20:
    with open("/t3home/gcelotto/ggHbb/PNN/config/featuresToRead_20plus.yaml") as f:
        feature_cfg = yaml.safe_load(f)
else:
    with open("/t3home/gcelotto/ggHbb/PNN/config/featuresToRead.yaml") as f:
        feature_cfg = yaml.safe_load(f)
featuresForTraining = feature_cfg['featuresForTraining']
columnsToRead = featuresForTraining+feature_cfg['genFeatures']
np.save(outFolder + "/model/featuresForTraining.npy", featuresForTraining)
# %%
# define the parameters for the nn

print("Before loading data")
# load data for the samples and preprocess the data(pT cut)
# fill the massHypo column
# cut the data to have same length in all the samples
# reweight each sample to have total weight 1, shuffle and split in train and test
Xtrain, Xval, Ytrain, Yval, Wtrain, Wval, rWtrain, rWval, genMassTrain, genMassVal = loadXYWrWSaved(inFolder=inFolder, isTest=False, btagWP=args.btagWP)
print(Xtrain.isna().sum())
dijetMassTrain = np.array(Xtrain.dijet_mass.values)
dijetMassVal = np.array(Xval.dijet_mass.values)
print(len(Xtrain), " events in train dataset")
print(len(Xval), " events in val dataset")




# %%

Xtrain = scale(Xtrain,featuresForTraining,  scalerName= outFolder + "/model/myScaler.pkl" ,fit=True, features_to_exclude=['jet1_has_sv', 'jet2_has_sv', 'has_jet3', 'jet2_has_muon'])
Xval  = scale(Xval, featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False, features_to_exclude=['jet1_has_sv', 'jet2_has_sv', 'has_jet3', 'jet2_has_muon'])
print(Xtrain.isna().sum())
# %%
Xtrain, Ytrain, Wtrain, rWtrain, genMassTrain, dijetMassTrain = Xtrain[:args.size], Ytrain[:args.size], Wtrain[:args.size], rWtrain[:args.size], genMassTrain[:args.size], dijetMassTrain[:args.size]
Xval, Yval, Wval, rWval, genMassVal, dijetMassVal = Xval[:args.size], Yval[:args.size], Wval[:args.size], rWval[:args.size], genMassVal[:args.size], dijetMassVal[:args.size]
print("Train Lenght after cutting", len(Xtrain))
print("Val Lenght after cutting", len(Xval))

# %%
# Comment if want to use flat in mjj
# rWtrain, rWval = Wtrain.copy(), Wval.copy()


# %%
sv1_features = [
    'jet1_sv_pt_prime',
    'jet1_sv_mass_prime',
    'jet1_sv_Ntrk',
    'jet1_has_sv'
]

sv2_features = [
    'jet2_sv_pt_prime',
    'jet2_sv_mass_prime',
    'jet2_sv_Ntrk',
    'jet2_has_sv'
]

jet3_features = [
    'jet3_pt_prime',
    'jet3_mass_prime',
    'jet3_eta_prime',
    'jet3_phi_prime',
    'jet3_btagWP',
    'has_jet3'
]

jet2_muon_features = [
    'jet2_muon_pt_prime',
    'jet2_muon_eta_prime',
    'jet2_muon_phi_prime',
    'jet2_muon_dxySig',
    'jet2_has_muon',
]

main_features = [f for f in featuresForTraining if f not in sv1_features + sv2_features + jet3_features + jet2_muon_features]
main_features+=['jet1_has_sv', 'jet2_has_sv', 'has_jet3', 'jet2_has_muon'] # add the gate features to the main features

Xtrain_main = torch.tensor(Xtrain[main_features].values, dtype=torch.float32)
Xtrain_sv1  = torch.tensor(Xtrain[sv1_features].values, dtype=torch.float32)
Xtrain_sv2  = torch.tensor(Xtrain[sv2_features].values, dtype=torch.float32)
Xtrain_jet3  = torch.tensor(Xtrain[jet3_features].values, dtype=torch.float32)
Xtrain_jet2_muon  = torch.tensor(Xtrain[jet2_muon_features].values, dtype=torch.float32)

Xval_main = torch.tensor(Xval[main_features].values, dtype=torch.float32)
Xval_sv1  = torch.tensor(Xval[sv1_features].values, dtype=torch.float32)
Xval_sv2  = torch.tensor(Xval[sv2_features].values, dtype=torch.float32)
Xval_jet3  = torch.tensor(Xval[jet3_features].values, dtype=torch.float32)
Xval_jet2_muon  = torch.tensor(Xval[jet2_muon_features].values, dtype=torch.float32)




#XtrainTensor = torch.tensor(Xtrain[featuresForTraining].values, dtype=torch.float32, ) #device=device
YtrainTensor = torch.tensor(Ytrain, dtype=torch.float, ).unsqueeze(1) #device=device
WtrainTensor = torch.tensor(rWtrain, dtype=torch.float32, ).unsqueeze(1) #device=device
dijetMassTrain_tensor = torch.tensor(dijetMassTrain, dtype=torch.float32, ).unsqueeze(1) #device=device


#Xval_tensor = torch.tensor(Xval[featuresForTraining].values, dtype=torch.float32, ) #device=device
Yval_tensor = torch.tensor(Yval, dtype=torch.float, ).unsqueeze(1) #device=device
Wval_tensor = torch.tensor(rWval, dtype=torch.float32, ).unsqueeze(1) #device=device
dijetMassVal_tensor = torch.tensor(dijetMassVal, dtype=torch.float32, ).unsqueeze(1) #device=device

#Xtest_tensor = torch.tensor(np.float32(Xtest[featuresForTraining].values)).float()

#train_masks = ((YtrainTensor < 2) & (dijetMassTrain_tensor < 90)).to(device)
#val_masks = ((Yval_tensor < 2) & (dijetMassVal_tensor < 90)).to(device)
train_masks = (YtrainTensor < 0.5) # to(device)
val_masks = (Yval_tensor < 0.5) # to(device)


traindataset = TensorDataset(
    Xtrain_main,
    Xtrain_sv1,
    Xtrain_sv2,
    Xtrain_jet3,
    Xtrain_jet2_muon,
    YtrainTensor,
    WtrainTensor,
    dijetMassTrain_tensor,
    train_masks
)
val_dataset = TensorDataset(
    Xval_main,
    Xval_sv1,
    Xval_sv2,
    Xval_jet3,
    Xval_jet2_muon,
    Yval_tensor,
    Wval_tensor,
    dijetMassVal_tensor,
    val_masks
)
# Drop last to drop the last (if incomplete size) batch

train_dataloader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=4)
# %%
# Model, loss, optimizer
model = ClassifierWithSV_jet3_muon2_update(
    input_dim_main=(len(main_features)),
    nNodes_main=args.nodes,
    nNodes_sv=(args.nodesSV,),
    nNodes_jet3=(args.nodesSV,),
    nNodes_jet2_muon=(args.nodesSV,)
)
#model.to(device)
criterion = nn.BCELoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=args.learningRate)

early_stopping_patience = args.earlyStopping
best_val_loss = float('inf')
patience_counter = 0
print("Train start")

# ______________________________________________________________________________
# ______________________________________________________________________________
# ______________________  TRAINING PHASE  ______________________________________
# ______________________________________________________________________________
# ______________________________________________________________________________



train_loss_history = []
train_classifier_loss_history = []
train_disco_loss_history = []
train_shape_loss_history = []

val_loss_history = []
val_classifier_loss_history = []
val_disco_loss_history = []
val_shape_loss_history = []
best_model_weights = None # weights saved for RestoreBestWeights
best_epoch = None
save_epoch = 0

# %%
#import time
#start =time.time()
for epoch in range(args.epochs):
    model.train()
    total_trainloss = 0.0
    total_train_classifier_loss = 0.0
    total_train_disco_loss = 0.0
    total_train_shape_loss = 0.0
    # Training phase
    for batch in train_dataloader:

        #X_batch, Y_batch, W_batch, dijetMass_batch, mask_batch = batch
        x_main, jet1_sv, jet2_sv, jet3_batch,jet2_muon_batch,  Y_batch, W_batch, dijetMass_batch, mask_batch = batch
        
        # Reset the gradients of all optimized torch.Tensor
        optimizer.zero_grad()
        #x_main = x_main.to(device)
        #jet1_sv = jet1_sv.to(device)
        #jet2_sv = jet2_sv.to(device)
        #Y_batch = Y_batch.to(device)
    
        predictions = model(x_main, jet1_sv, jet2_sv, jet3_batch, jet2_muon_batch)

        raw_loss = criterion(predictions, Y_batch)
        # Apply weights manually
        classifier_loss = (raw_loss * W_batch).mean()
        loss = classifier_loss.clone()
        # Combined loss
        if args.lambda_disco>=1e-1:
            W_batch = torch.ones([len(W_batch), 1]) #device=device
            dCorr = distance_corr(predictions[mask_batch], dijetMass_batch[mask_batch], W_batch[mask_batch])
            loss += args.lambda_disco*dCorr



        loss.backward()
        optimizer.step()
        
        # Sum over batches
        total_trainloss += loss.item()
        total_train_classifier_loss += classifier_loss.item()
        #total_train_shape_loss += args.lambda_shape*shape_loss.item() if args.lambda_shape>=1e-5 else 0.0
        if args.lambda_disco>=1e-1:
            total_train_disco_loss += args.lambda_disco*dCorr.item()
        #stop =time.time()
        #print("Epoch %d: Training batch %d took %.2f seconds" % (epoch, idx, stop - start), flush=True)

# 44batches in 131 s
# ______________________________________________________________________________
# ______________________________________________________________________________
# ______________________  VALIDATION PHASE  ____________________________________
# ______________________________________________________________________________
# ______________________________________________________________________________



    
    if epoch % 1 == 0: 
        model.eval()
        total_val_loss = 0.0
        total_val_classifier_loss = 0.0
        total_val_disco_loss = 0.0
        total_val_shape_loss = 0.0


        with torch.no_grad():
            for batch in val_dataloader:
                x_main, jet1_sv, jet2_sv, jet3_batch, jet2_muon_batch, Y_batch, W_batch, dijetMass_batch, mask_batch = batch
        
                # Reset the gradients of all optimized torch.Tensor
                optimizer.zero_grad()
                #x_main = x_main.to(device)
                #jet1_sv = jet1_sv.to(device)
                #jet2_sv = jet2_sv.to(device)
                #Y_batch = Y_batch.to(device)

                predictions = model(x_main, jet1_sv, jet2_sv, jet3_batch, jet2_muon_batch)

                raw_loss = criterion(predictions, Y_batch)
                # Apply weights manually
                classifier_loss = (raw_loss * W_batch).mean()
                loss = classifier_loss.clone()
                # Combined loss
                if args.lambda_disco>=1e-1:
                    W_batch = torch.ones([len(W_batch), 1]) #device=device
                    dCorr = distance_corr(predictions[mask_batch], dijetMass_batch[mask_batch], W_batch[mask_batch])
                    loss += args.lambda_disco*dCorr

                        # Sum over batches
                total_val_loss += loss.item()
                #total_val_shape_loss += args.lambda_shape*shape_loss.item() if args.lambda_shape>=1e-5 else 0.0
                total_val_classifier_loss += classifier_loss.item()
                if args.lambda_disco>=1e-1:
                    total_val_disco_loss += args.lambda_disco*dCorr.item()


    # Calculate average losses (average over batches)
    avg_train_loss = total_trainloss / len(train_dataloader)
    avg_train_classifier_loss = total_train_classifier_loss / len(train_dataloader)
    avg_train_shape_loss = total_train_shape_loss / len(train_dataloader)
    avg_train_disco_loss = total_train_disco_loss / len(train_dataloader)

    avg_val_loss = total_val_loss / len(val_dataloader)
    avg_val_classifier_loss = total_val_classifier_loss / len(val_dataloader)
    avg_val_shape_loss = total_val_shape_loss / len(val_dataloader)
    avg_val_disco_loss = total_val_disco_loss / len(val_dataloader)

    train_loss_history.append(avg_train_loss)
    train_classifier_loss_history.append(avg_train_classifier_loss)
    train_shape_loss_history.append(avg_train_shape_loss)
    train_disco_loss_history.append(avg_train_disco_loss)

    val_loss_history.append(avg_val_loss)
    val_classifier_loss_history.append(avg_val_classifier_loss)
    val_shape_loss_history.append(avg_val_shape_loss)
    val_disco_loss_history.append(avg_val_disco_loss)

    # Print losses
    print(f"Epoch [{epoch+1}/{args.epochs}], "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Train Classifier Loss: {avg_train_classifier_loss:.4f}, "
            f"Train Disco Loss: {avg_train_disco_loss:.4f}, "
            #f"Train Shape Loss: {avg_train_shape_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f} "
            f"Val Classifier Loss: {avg_val_classifier_loss:.4f}, "
            f"Val Disco Loss: {avg_val_disco_loss:.4f}, ",
            #f"Val Shape Loss: {avg_val_shape_loss:.4f}, ",
          flush=(epoch % 1 == 0))


    # Early Stopping check
    if avg_val_loss < best_val_loss:
        if save_epoch is not None:
            if (epoch-save_epoch>=10):
                torch.save(model, outFolder+"/model/model_e%d.pth"%(epoch))
                save_epoch = epoch
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
np.save(outFolder + "/model/train_loss_history.npy", train_loss_history)
np.save(outFolder + "/model/val_loss_history.npy", val_loss_history)

np.save(outFolder + "/model/train_classifier_loss_history.npy", train_classifier_loss_history)
np.save(outFolder + "/model/val_classifier_loss_history.npy", val_classifier_loss_history)

np.save(outFolder + "/model/train_disco_loss_history.npy", train_disco_loss_history)
np.save(outFolder + "/model/val_disco_loss_history.npy", val_disco_loss_history)

#np.save(outFolder + "/model/train_shape_loss_history.npy", train_shape_loss_history)
#np.save(outFolder + "/model/val_shape_loss_history.npy", val_shape_loss_history)

# %%
torch.save(model, outFolder+"/model/model.pth")
print("Model saved")
with open(outFolder + "/model/training.txt", "w") as file:
    for key, value in vars(args).items():  # convert Namespace -> dict
        file.write(f"{key} : {value}\n")

# %%
#sys.path.append("/t3home/gcelotto/ggHbb/scripts/plotScripts/plotFeatures.py")
#from plotFeatures import plotNormalizedFeatures
#plotNormalizedFeatures(data=[Xtrain[featuresForTraining][Ytrain==0],Xtrain[featuresForTraining][Ytrain==1]], outFile="/t3home/gcelotto/ggHbb/GNN/NN_featuresMLP.png", legendLabels=["bkg", "sig"], colors=['blue', 'red'], error=False, autobins=True)
#plotNormalizedFeatures(data=[Xtrain[featuresForTraining][Ytrain==0],Xtrain[featuresForTraining][Ytrain==1]], outFile="/t3home/gcelotto/ggHbb/GNN/NN_featuresMLP_rW.png", legendLabels=["bkg", "sig"], colors=['blue', 'red'], error=False, autobins=True, weights=[rWtrain[Ytrain==0], rWtrain[Ytrain==1]])
# %%
