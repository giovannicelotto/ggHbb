# %%
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import numpy as np
import dcor
import logging
from sklearn.model_selection import train_test_split
from checkOrthogonality import checkOrthogonality, checkOrthogonalityInMassBins
import pandas as pd
import sys
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
sys.path.append("/t3home/gcelotto/ggHbb/PNN/helpers")
from plotFeatures import plotNormalizedFeatures
# PNN helpers
from helpers.getFeatures import getFeatures
from helpers.getParams import getParams
from helpers.loadSaved import loadXYrWSaved
from helpers.getInfolderOutfolder import getInfolderOutfolder
from helpers.scaleUnscale import scale, unscale
from helpers.dcorLoss import *
from helpers.saveDataAndPredictions import saveXYWP
from helpers.flattenWeights import flattenWeights
from helpers.doPlots import runPlotsTorch, plot_lossTorch

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse

# %%

gpuFlag=True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define folder of input and output. Create the folders if not existing
hp = getParams()
parser = argparse.ArgumentParser(description="Script.")
# Define arguments
parser.add_argument("-l", "--lambdaCor", type=float, help="lambda for penalty term", default=None)
args = parser.parse_args()
if args.lambdaCor is not None:
    hp["lambda_reg"] = args.lambdaCor 
# %%
inFolder, outFolder = getInfolderOutfolder(name = "dec12_%s"%(str(hp["lambda_reg"]).replace('.', 'p')))
# Define features to read and to train the pNN (+parameter massHypo) and save the features for training in outfolder
featuresForTraining = np.load(inFolder + "/data/featuresForTraining.npy")
# %%
# define the parameters for the nn


# load data for the samples and preprocess the data(pT cut)
# fill the massHypo column
# cut the data to have same length in all the samples
# reweight each sample to have total weight 1, shuffle and split in train and test
Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, rWtrain, rWval, Wtest, genMassTrain, genMassVal, genMassTest = loadXYrWSaved(inFolder=inFolder+"/data")
#advFeatureTest  = np.load(inFolder+"/data/advFeatureTest.npy")
advFeatureTrain = np.load(inFolder+"/data/advFeatureTrain.npy")     
advFeatureVal   = np.load(inFolder+"/data/advFeatureVal.npy")


# %%
# scale with standard scalers and apply log to any pt and mass distributions

Xtrain = scale(Xtrain,featuresForTraining,  scalerName= outFolder + "/model/myScaler.pkl" ,fit=True)
Xval  = scale(Xval, featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)
#advFeatureTrain = scale(pd.DataFrame(advFeatureTrain, columns=['jet1_btagDeepFlavB']),['jet1_btagDeepFlavB'],  scalerName= outFolder + "/model/myScaler_adv.pkl" ,fit=True)
#advFeatureVal  = scale(pd.DataFrame(advFeatureVal,columns=['jet1_btagDeepFlavB']), ['jet1_btagDeepFlavB'], scalerName= outFolder + "/model/myScaler_adv.pkl" ,fit=False)

#size = 5000
#Xtrain, Ytrain, rWtrain, genMassTrain, advFeatureTrain = Xtrain[:size], Ytrain[:size], rWtrain[:size], genMassTrain[:size], advFeatureTrain[:size]
#Xval, Yval, rWval, genMassVal, advFeatureVal = Xval[:size], Yval[:size], rWval[:size], genMassVal[:size], advFeatureVal[:size]


# %%
XtrainTensor = torch.tensor(np.float32(Xtrain[featuresForTraining].values)).float()
YtrainTensor = torch.tensor(Ytrain).unsqueeze(1).float()
rWtrainTensor = torch.tensor(rWtrain).unsqueeze(1).float()
advFeatureTrain_tensor = torch.tensor(advFeatureTrain).float()

Xval_tensor = torch.tensor(np.float32(Xval[featuresForTraining].values)).float()
Yval_tensor = torch.tensor(Yval).unsqueeze(1).float()
rWval_tensor = torch.tensor(rWval).unsqueeze(1).float()
advFeatureVal_tensor = torch.tensor(advFeatureVal).float()

#Xtest_tensor = torch.tensor(np.float32(Xtest[featuresForTraining].values)).float()



traindataset = TensorDataset(XtrainTensor, YtrainTensor, rWtrainTensor, advFeatureTrain_tensor)
val_dataset = TensorDataset(Xval_tensor, Yval_tensor, rWval_tensor, advFeatureVal_tensor)

traindataloader = DataLoader(traindataset, batch_size=hp["batch_size"], shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=hp["batch_size"], shuffle=False)
# %%
# Model, loss, optimizer
model = Classifier(input_dim=Xtrain[featuresForTraining].shape[1])
model.to(device)
criterion = nn.BCELoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=hp["learning_rate"])

early_stopping_patience = hp["patienceES"]
best_val_loss = float('inf')
patience_counter = 0


# Training loop

epochs = hp["epochs"]

trainloss_history = []
trainclassifier_loss_history = []
traindcor_loss_history = []

val_loss_history = []
val_classifier_loss_history = []
val_dcor_loss_history = []
best_model_weights = None # weights saved for RestoreBestWeights
best_epoch = None
# %%
for epoch in range(epochs):
    model.train()
    total_trainloss = 0.0
    total_trainclassifier_loss = 0.0
    total_traindcor_loss = 0.0

    # Training phase
    for batch in traindataloader:
        X_batch, Y_batch, W_batch, advFeature_batch = [item.to(device) for item in batch]

        optimizer.zero_grad()
        predictions = model(X_batch).squeeze().unsqueeze(1)
        raw_loss = criterion(predictions, Y_batch)
        # Apply weights manually
        classifier_loss = (raw_loss * W_batch.unsqueeze(1)).mean()

         # Apply the mask to select only Y_batch == 0 for dcor calculation
        mask = (Y_batch < 0.5).squeeze().squeeze()

        # Apply the mask to the relevant tensors for distance correlation calculation
        X_batch_filtered = X_batch[mask]
        W_batch_filtered = W_batch[mask]
        advFeature_batch_filtered = advFeature_batch[mask]

        # If there are any remaining entries after filtering, calculate dcor
        if X_batch_filtered.size(0) > 0:
            dCorr = distance_corr(predictions[mask], advFeature_batch_filtered, W_batch_filtered)
        else:
            dCorr = torch.tensor(0.0, device=device)

        # Combined loss
        loss = classifier_loss + hp["lambda_reg"] * dCorr
        loss.backward()
        optimizer.step()

        total_trainloss += loss.item()
        total_trainclassifier_loss += classifier_loss.item()
        total_traindcor_loss += dCorr.item()

    # Validation phase
    model.eval()
    total_val_loss = 0.0
    total_val_classifier_loss = 0.0
    total_val_dcor_loss = 0.0

    with torch.no_grad():
        for batch in val_dataloader:
            X_batch, Y_batch, W_batch, advFeature_batch = [item.to(device) for item in batch]
        

            predictions = model(X_batch).squeeze().unsqueeze(1)

            raw_loss = criterion(predictions, Y_batch)
        # Apply weights manually
            classifier_loss = (raw_loss * W_batch.unsqueeze(1)).mean()


            mask = (Y_batch < 0.5).squeeze().squeeze()

            # Apply the mask to the relevant tensors for distance correlation calculation
            X_batch_filtered = X_batch[mask]
            W_batch_filtered = W_batch[mask]
            advFeature_batch_filtered = advFeature_batch[mask]

            # If there are any remaining entries after filtering, calculate dcor
            if X_batch_filtered.size(0) > 0:
                dCorr = distance_corr(predictions[mask], advFeature_batch_filtered, W_batch_filtered)
            else:
                dCorr = torch.tensor(0.0, device=device)

            # Combined loss
            loss = classifier_loss + hp["lambda_reg"] * dCorr
            total_val_loss += loss.item()
            total_val_classifier_loss += classifier_loss.item()
            total_val_dcor_loss += dCorr.item()


    # Calculate average losses
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
          flush=(epoch % 10 == 0))

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
plot_lossTorch(trainloss_history, val_loss_history, 
              trainclassifier_loss_history, val_classifier_loss_history,
              traindcor_loss_history, val_dcor_loss_history,
              best_epoch,
              outFolder)
# %%
torch.save(model, outFolder+"/model/model.pth")
model = torch.load(outFolder+"/model/model.pth")


# Sets the model in inference mode
# * normalisation layers use running statistics
# * de-activates Dropout layers if any
model.eval()

with torch.no_grad():  # No need to track gradients for inference
    YPredTrain_tensor = model(XtrainTensor)
    YPredVal_tensor = model(Xval_tensor)


# %%
Xtrain = unscale(Xtrain, featuresForTraining=featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl")
Xval = unscale(Xval, featuresForTraining=featuresForTraining,   scalerName =  outFolder + "/model/myScaler.pkl")
#advFeatureTrain = unscale(pd.DataFrame(advFeatureTrain),['jet1_btagDeepFlavB'],  scalerName= outFolder + "/model/myScaler_adv.pkl" )
#advFeatureVal  = unscale(pd.DataFrame(advFeatureVal), ['jet1_btagDeepFlavB'], scalerName= outFolder + "/model/myScaler_adv.pkl")




# %%

YPredTrain = YPredTrain_tensor.numpy()
YPredVal = YPredVal_tensor.numpy()

dcor_value = dcor.distance_correlation(YPredVal[Yval==0][:40000], advFeatureVal[Yval==0][:40000])


logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, etc.)
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=outFolder+"/logging.log"
)

logging.info("DisCor value: %f", dcor_value)

# %%
#runPlotsTorch(Xtrain, Xval, Ytrain, Yval, np.array(Xtrain.sf), np.array(Xval.sf), YPredTrain, YPredVal, featuresForTraining, model, inFolder, outFolder, genMassTrain, genMassVal)
Xval['jet1_btagDeepFlavB'] = advFeatureVal
checkOrthogonality(Xval[Yval==0], 'jet1_btagDeepFlavB', np.array(YPredVal[Yval==0]>0.7), np.array(YPredVal[Yval==0]<0.7), label_mask1='Prediction NN > 0.7', label_mask2='Prediction NN < 0.7', label_toPlot = 'jet1_btagDeepFlavB', bins=np.linspace(0.2783,1, 31), ax=None, axLegend=False, outName=outFolder+"/performance/orthogonalityInclusive.png" )

# %%
Xval['PNN'] = YPredVal
mass_bins = np.linspace(40, 300, 25)
ks_p_value_PNN, p_value_PNN, chi2_values_PNN = checkOrthogonalityInMassBins(
    df=Xval[Yval==0],
    featureToPlot='PNN',
    mask1=np.array(Xval[Yval==0]['jet1_btagDeepFlavB'] >= 0.7100),
    mask2=np.array(Xval[Yval==0]['jet1_btagDeepFlavB'] < 0.7100),
    label_mask1  = 'NN High',
    label_mask2  = 'NN Low',
    label_toPlot = 'PNN',
    bins=np.linspace(0.2, 1, 21),
    mass_bins=mass_bins,
    mass_feature='dijet_mass',
    figsize=(20, 30),
    outName=outFolder+"/performance/orthogonalityBinned.png" 
)
# %%
from scipy.stats import ks_2samp, chisquare, chi2
sum_chi2_PNN = np.sum(chi2_values_PNN)
# nbins in the variable
# -1 (binned chi2 -> ndof = observed - 1)
# -1 (constrain from normalization)
# x 24 bins
# -1 due to the overall normalization
ddof = (20-1-1)*24 - 1
p_value_PNN = chi2.sf(sum_chi2_PNN, ddof)
print()
logging.info(f"{sum_chi2_PNN} / {ddof} -> pval = {p_value_PNN}")
print(f"{sum_chi2_PNN} / {ddof} -> pval = {p_value_PNN}")

# %%
