# %%
import numpy as np
import sys
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
sys.path.append("/t3home/gcelotto/ggHbb/PNN/helpers")
from datetime import datetime

# Get current month and day
current_date = datetime.now().strftime("%b%d")  # This gives the format like 'Dec12'

# PNN helpers
from helpers.getFeatures import getFeatures
from helpers.getParams import getParams
from helpers.loadSaved import loadXYWrWSaved
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
# %%

parser = argparse.ArgumentParser(description="Script.")
#### Define arguments
parser.add_argument("-l", "--lambda_dcor", type=float, help="lambda for penalty term", default=None)
parser.add_argument("-e", "--epochs", type=int, help="number of epochs", default=1500)
parser.add_argument("-s", "--size", type=int, help="Number of events to crop training dataset", default=1000000000)
parser.add_argument("-lr", "--learningRate", type=float, help="learning rate", default=None)
parser.add_argument("-bs", "--batch_size", type=int, help="Number of events perBatch", default=None)
parser.add_argument("-eS", "--earlyStopping", type=int, help="patience early stop", default=None)
parser.add_argument("-n", "--nodes",type=lambda s: [int(item) for item in s.split(',')],  # Convert comma-separated string to list of ints
                        help="List of nodes per layer (e.g., 128,64,32 for a 3-layer NN)",default=None
)
args = parser.parse_args()

# %%
try:
    if args.lambda_dcor is not None:
        hp["lambda_dcor"] = args.lambda_dcor 
        print("lambda_dcor changed to ", hp["lambda_dcor"])
    if args.epochs is not None:
        hp["epochs"] = args.epochs 
        print("N epochs to ", hp["epochs"])
    if args.batch_size is not None:
        hp["batch_size"] = int(args.batch_size )
        print("N batch_size to ", hp["batch_size"])
    if args.nodes is not None:
        hp["nNodes"] = args.nodes
    if args.earlyStopping is not None:
        hp["patienceES"] = args.earlyStopping
    if args.learningRate is not None:
        hp["learning_rate"] = args.learningRate
    if args.size != int(1e9):
        hp["size"] = args.size
    print("After parameters")
    print(hp)
except:
    print("-"*40)
    print("Error in passing the arguments!")
    print("lambda_dcor changed to ", hp["lambda_dcor"])
    hp["lambda_dcor"] = 0. 
    print(hp)
    print("interactive mode")
    print("-"*40)
    hp["size"] = 20000
    hp["batch_size"]=1000
    hp['nodes'] = [4, 2]
    hp['patienceES'] = 40
# %%
sampling = True
boosted = 1
inFolder, outFolder = getInfolderOutfolder(name = "%s_%d_%s"%(current_date, boosted, str(hp["lambda_dcor"]).replace('.', 'p')), suffixResults='_mjjDisco')
inFolder = "/t3home/gcelotto/ggHbb/PNN/input/data_sampling_pt%d"%(boosted) if sampling else "/t3home/gcelotto/ggHbb/PNN/input/data_highPt"

# Define features to read and to train the pNN (+parameter massHypo) and save the features for training in outfolder
featuresForTraining, columnsToRead = getFeatures(outFolder, massHypo=False)
featuresForTraining = featuresForTraining + ['dijet_mass']
#if 'jet2_btagDeepFlavB' in featuresForTraining:
#    featuresForTraining.remove('jet2_btagDeepFlavB')
#if 'jet1_btagDeepFlavB' in featuresForTraining:
#    featuresForTraining.remove('jet1_btagDeepFlavB')

# %%
# define the parameters for the nn

print("Before loading data")
# load data for the samples and preprocess the data(pT cut)
# fill the massHypo column
# cut the data to have same length in all the samples
# reweight each sample to have total weight 1, shuffle and split in train and test
Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, Wtrain, Wval,Wtest, rWtrain, rWval, genMassTrain, genMassVal, genMassTest = loadXYWrWSaved(inFolder=inFolder)
dijetMassTrain = np.array(Xtrain.dijet_mass.values)
dijetMassVal = np.array(Xval.dijet_mass.values)
print(len(Xtrain), " events in train dataset")
print(len(Xval), " events in val dataset")
# %%
# %%
# scale with standard scalers and apply log to any pt and mass distributions

Xtrain = scale(Xtrain,featuresForTraining,  scalerName= outFolder + "/model/myScaler.pkl" ,fit=True)
Xval  = scale(Xval, featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)
# %%
size = hp["size"]
Xtrain, Ytrain, Wtrain, rWtrain, genMassTrain, dijetMassTrain = Xtrain[:size], Ytrain[:size], Wtrain[:size], rWtrain[:size], genMassTrain[:size], dijetMassTrain[:size]
Xval, Yval, Wval, rWval, genMassVal, dijetMassVal = Xval[:size], Yval[:size], Wval[:size], rWval[:size], genMassVal[:size], dijetMassVal[:size]
print("Train Lenght after cutting", len(Xtrain))
print("Val Lenght after cutting", len(Xval))

# %%
# Comment if want to use flat in mjj
# rWtrain, rWval = Wtrain.copy(), Wval.copy()
# %%

XtrainTensor = torch.tensor(Xtrain[featuresForTraining].values, dtype=torch.float32, device=device)
YtrainTensor = torch.tensor(Ytrain, dtype=torch.float, device=device).unsqueeze(1)
WtrainTensor = torch.tensor(Wtrain, dtype=torch.float32, device=device).unsqueeze(1)
dijetMassTrain_tensor = torch.tensor(dijetMassTrain, dtype=torch.float32, device=device).unsqueeze(1)


Xval_tensor = torch.tensor(Xval[featuresForTraining].values, dtype=torch.float32, device=device)
Yval_tensor = torch.tensor(Yval, dtype=torch.float, device=device).unsqueeze(1)
Wval_tensor = torch.tensor(Wval, dtype=torch.float32, device=device).unsqueeze(1)
dijetMassVal_tensor = torch.tensor(dijetMassVal, dtype=torch.float32, device=device).unsqueeze(1)

#Xtest_tensor = torch.tensor(np.float32(Xtest[featuresForTraining].values)).float()

train_masks = (YtrainTensor < 0.5).to(device)
val_masks = (Yval_tensor < 0.5).to(device)


traindataset = TensorDataset(
    XtrainTensor.to(device),
    YtrainTensor.to(device),
    WtrainTensor.to(device),
    dijetMassTrain_tensor.to(device),
    train_masks.to(device)
)
val_dataset = TensorDataset(
    Xval_tensor.to(device),
    Yval_tensor.to(device),
    Wval_tensor.to(device),
    dijetMassVal_tensor.to(device),
    val_masks.to(device)
)
# Drop last to drop the last (if incomplete size) batch
hp["batch_size"] = hp["batch_size"] if hp["batch_size"]<len(Xtrain) else len(Xtrain)
train_dataloader = DataLoader(traindataset, batch_size=hp["batch_size"], shuffle=True, drop_last=True)

hp["val_batch_size"] = hp["batch_size"] if hp["batch_size"]<len(Xval) else len(Xval)
val_dataloader = DataLoader(val_dataset, batch_size=hp["val_batch_size"], shuffle=False, drop_last=True)
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



train_loss_history = []

val_loss_history = []
best_model_weights = None # weights saved for RestoreBestWeights
best_epoch = None

# %%

for epoch in range(hp["epochs"]):
    model.train()
    total_trainloss = 0.0
    # Training phase
    for batch in train_dataloader:
        X_batch, Y_batch, W_batch, dijetMass_batch, mask_batch = batch

        
        # Reset the gradients of all optimized torch.Tensor
        optimizer.zero_grad()
        predictions = model(X_batch)
        raw_loss = criterion(predictions, Y_batch)
        # Apply weights manually
        classifier_loss = (raw_loss * W_batch).mean()
        # Combined loss
        loss = classifier_loss
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
                X_batch, Y_batch, W_batch, dijetMass_batch, mask_batch = batch
                predictions = model(X_batch)

                raw_loss = criterion(predictions, Y_batch)
                # Apply weights manually
                classifier_loss = (raw_loss * W_batch).mean()
                # Combined loss
                loss = classifier_loss 
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
np.save(outFolder + "/model/train_loss_history.npy", train_loss_history)
np.save(outFolder + "/model/val_loss_history.npy", val_loss_history)

# %%
torch.save(model, outFolder+"/model/model.pth")
print("Model saved")
with open(outFolder + "/model/training.txt", "w") as file:
    for key, value in hp.items():
        file.write(f"{key} : {value}\n")

# %%
