# %%
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import sys
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
from plotFeatures import plotNormalizedFeatures
sys.path.append("/t3home/gcelotto/ggHbb/PNN/helpers")
import torch
import torch.nn as nn
import torch.optim as optim
from helpers.getFeatures import getFeatures
from helpers.getParams import getParams
from helpers.loadData_adversarial import loadData_adversarial
from helpers.getInfolderOutfolder import getInfolderOutfolder
from helpers.scaleUnscale import scale, unscale
from helpers.dcorLoss import *
from torch.utils.data import DataLoader, TensorDataset
from helpers.flattenWeights import flattenWeights
import numpy as np
import dcor
import logging
from sklearn.model_selection import train_test_split
from helpers.doPlots import runPlotsTorch, plot_lossTorch
from checkOrthogonality import checkOrthogonality, checkOrthogonalityInMassBins

# %%

gpuFlag=True if torch.cuda.is_available() else False
gpuFlag=True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define folder of input and output. Create the folders if not existing
hp = getParams()
inFolder, outFolder = getInfolderOutfolder(name = "dec11_dcor_%s"%(str(hp["lambda_reg"]).replace('.', 'p')))

# Define features to read and to train the pNN (+parameter massHypo) and save the features for training in outfolder
featuresForTraining, columnsToRead = getFeatures(outFolder, massHypo=True)
# %%
# define the parameters for the nn


# load data for the samples and preprocess the data(pT cut)
# fill the massHypo column
# cut the data to have same length in all the samples
# reweight each sample to have total weight 1, shuffle and split in train and test
data = loadData_adversarial(hp["nReal"], hp["nMC"], 1e5, outFolder, columnsToRead, featuresForTraining, hp)
Xtrain_, Xtest, Ytrain_, Ytest, advFeatureTrain_, advFeatureTest, Wtrain_, Wtest, genMassTrain_, genMassTest = data


# %%
# Higgs and Data have flat distribution in m_jj
rWtrain_, rWtest = flattenWeights(Xtrain_, Xtest, Ytrain_, Ytest, Wtrain_, Wtest, inFolder, outName=outFolder+ "/performance/massReweighted.png")
#rWtrain, rWtest = Wtrain.copy(), Wtest.copy()
# %%
if gpuFlag==False:
    plotNormalizedFeatures(data=[Xtrain_[Ytrain_==0], Xtrain_[Ytrain_==1], Xtest[Ytest==0], Xtest[Ytest==1]],
                       outFile=outFolder+"/performance/features.png", legendLabels=['Data Train', 'Higgs Train', 'Data Test', 'Higgs Test'],
                       colors=['blue', 'red', 'blue', 'red'], histtypes=[u'step', u'step', 'bar', 'bar'],
                       alphas=[1, 1, 0.4, 0.4], figsize=(20,40), autobins=False,
                       weights=[rWtrain_[Ytrain_==0], rWtrain_[Ytrain_==1], rWtest[Ytest==0], rWtest[Ytest==1]], error=True)
# %%
# scale with standard scalers and apply log to any pt and mass distributions
import pandas as pd
Xtrain_ = scale(Xtrain_,featuresForTraining,  scalerName= outFolder + "/model/myScaler.pkl" ,fit=True)
Xtest  = scale(Xtest, featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)
advFeatureTrain_ = scale(pd.DataFrame(advFeatureTrain_),['jet1_btagDeepFlavB'],  scalerName= outFolder + "/model/myScaler_adv.pkl" ,fit=True)
advFeatureTest  = scale(pd.DataFrame(advFeatureTest), ['jet1_btagDeepFlavB'], scalerName= outFolder + "/model/myScaler_adv.pkl" ,fit=False)
# %%
if gpuFlag==False:
    plotNormalizedFeatures(data=[Xtrain_[Ytrain_==0], Xtrain_[Ytrain_==1], Xtest[Ytest==0], Xtest[Ytest==1]],
                       outFile=outFolder+"/performance/features_scaled.png", legendLabels=['Data Train', 'Higgs Train', 'Data Test', 'Higgs Test'],
                       colors=['blue', 'red', 'blue', 'red'], histtypes=[u'step', u'step', 'bar', 'bar'],
                       alphas=[1, 1, 0.4, 0.4], figsize=(20,40), autobins=True,
                       weights=[Wtrain_[Ytrain_==0], Wtrain_[Ytrain_==1], Wtest[Ytest==0], Wtest[Ytest==1]], error=True)

# %%
Wtrain_[Ytrain_==0] = Wtrain_[Ytrain_==0]/np.mean(Wtrain_[Ytrain_==0])
Wtrain_[Ytrain_==1] = Wtrain_[Ytrain_==1]/np.mean(Wtrain_[Ytrain_==1])
Xtrain, Xval, Ytrain, Yval, Wtrain, Wval, advFeatureTrain, advFeatureVal, genMassTrain, genMassVal = train_test_split(Xtrain_, Ytrain_, Wtrain_, advFeatureTrain_, genMassTrain_, test_size=hp["val_split"])


Xtrain_tensor = torch.tensor(np.float32(Xtrain[featuresForTraining].values)).float()
Ytrain_tensor = torch.tensor(Ytrain).unsqueeze(1).float()
Wtrain_tensor = torch.tensor(Wtrain).unsqueeze(1).float()
advFeatureTrain_tensor = torch.tensor(advFeatureTrain.values).float()

Xval_tensor = torch.tensor(np.float32(Xval[featuresForTraining].values)).float()
Yval_tensor = torch.tensor(Yval).unsqueeze(1).float()
Wval_tensor = torch.tensor(Wval).unsqueeze(1).float()
advFeatureVal_tensor = torch.tensor(advFeatureVal.values).float()

Xtest_tensor = torch.tensor(np.float32(Xtest[featuresForTraining].values)).float()



train_dataset = TensorDataset(Xtrain_tensor, Ytrain_tensor, Wtrain_tensor, advFeatureTrain_tensor)
val_dataset = TensorDataset(Xval_tensor, Yval_tensor, Wval_tensor, advFeatureVal_tensor)

train_dataloader = DataLoader(train_dataset, batch_size=hp["batch_size"], shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=hp["batch_size"], shuffle=False)
# %%
# Model, loss, optimizer
model = Classifier(input_dim=Xtrain[featuresForTraining].shape[1])
model.to(device)
criterion = nn.BCELoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=1e-5)

early_stopping_patience = 20
best_val_loss = float('inf')
patience_counter = 0


# Training loop

epochs = 1000

train_loss_history = []
train_classifier_loss_history = []
train_dcor_loss_history = []

val_loss_history = []
val_classifier_loss_history = []
val_dcor_loss_history = []
best_model_weights = None # weights saved for RestoreBestWeights
best_epoch = None
for epoch in range(epochs):
    model.train()
    total_train_loss = 0.0
    total_train_classifier_loss = 0.0
    total_train_dcor_loss = 0.0

    # Training phase
    for batch in train_dataloader:
        X_batch, Y_batch, W_batch, dijet_batch = [item.to(device) for item in batch]

        optimizer.zero_grad()
        predictions = model(X_batch).squeeze().unsqueeze(1)
        raw_loss = criterion(predictions, Y_batch)
        # Apply weights manually
        classifier_loss = (raw_loss * W_batch.unsqueeze(1)).mean()

        # Distance correlation
        dCorr = distance_corr(predictions, dijet_batch, W_batch)

        # Combined loss
        loss = classifier_loss + hp["lambda_reg"] * dCorr
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        total_train_classifier_loss += classifier_loss.item()
        total_train_dcor_loss += dCorr.item()

    # Validation phase
    model.eval()
    total_val_loss = 0.0
    total_val_classifier_loss = 0.0
    total_val_dcor_loss = 0.0

    with torch.no_grad():
        for batch in val_dataloader:
            X_batch, Y_batch, W_batch, dijet_batch = [item.to(device) for item in batch]
        

            predictions = model(X_batch).squeeze().unsqueeze(1)

            raw_loss = criterion(predictions, Y_batch)
        # Apply weights manually
            classifier_loss = (raw_loss * W_batch.unsqueeze(1)).mean()


            # Distance correlation
            dCorr = distance_corr(predictions, dijet_batch, W_batch)

            # Combined loss
            loss = classifier_loss + hp["lambda_reg"] * dCorr
            total_val_loss += loss.item()
            total_val_classifier_loss += classifier_loss.item()
            total_val_dcor_loss += dCorr.item()


    # Calculate average losses
    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_train_classifier_loss = total_train_classifier_loss / len(train_dataloader)
    avg_train_dcor_loss = total_train_dcor_loss / len(train_dataloader)

    avg_val_loss = total_val_loss / len(val_dataloader)
    avg_val_classifier_loss = total_val_classifier_loss / len(val_dataloader)
    avg_val_dcor_loss = total_val_dcor_loss / len(val_dataloader)

    train_loss_history.append(avg_train_loss)
    train_classifier_loss_history.append(avg_train_classifier_loss)
    train_dcor_loss_history.append(avg_train_dcor_loss)
    val_loss_history.append(avg_val_loss)
    val_classifier_loss_history.append(avg_val_classifier_loss)
    val_dcor_loss_history.append(avg_val_dcor_loss)

    # Print losses
    print(f"Epoch [{epoch+1}/{epochs}], "
          f"Train Loss: {avg_train_loss:.4f}, Classifier Loss: {avg_train_classifier_loss:.4f}, dCor Loss: {avg_train_dcor_loss:.8f}, "
          f"Val Loss: {avg_val_loss:.4f}, Val Classifier Loss: {avg_val_classifier_loss:.4f}, Val dCor Loss: {avg_val_dcor_loss:.8f}")

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
plot_lossTorch(train_loss_history, val_loss_history, 
              train_classifier_loss_history, val_classifier_loss_history,
              train_dcor_loss_history, val_dcor_loss_history,
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
    YPredTrain_tensor = model(Xtrain_tensor)
    YPredVal_tensor = model(Xval_tensor)
    YPredTest_tensor = model(Xtest_tensor)  


# %%

YPredTrain = YPredTrain_tensor.numpy()
YPredVal = YPredVal_tensor.numpy()
YPredTest = YPredTest_tensor.numpy()

dcor_value = dcor.distance_correlation(YPredTest[:40000], advFeatureTest.values[:40000])


logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, etc.)
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=outFolder+"/logging.log"
)

logging.info("DisCor value: %f", dcor_value)
# %%
Xtrain = unscale(Xtrain, featuresForTraining=featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl")
Xtest = unscale(Xtest, featuresForTraining=featuresForTraining,   scalerName =  outFolder + "/model/myScaler.pkl")
Xval = unscale(Xval, featuresForTraining=featuresForTraining,   scalerName =  outFolder + "/model/myScaler.pkl")
advFeatureTrain = unscale(pd.DataFrame(advFeatureTrain),['jet1_btagDeepFlavB'],  scalerName= outFolder + "/model/myScaler_adv.pkl" )
advFeatureTest  = unscale(pd.DataFrame(advFeatureTest), ['jet1_btagDeepFlavB'], scalerName= outFolder + "/model/myScaler_adv.pkl")
advFeatureVal  = unscale(pd.DataFrame(advFeatureVal), ['jet1_btagDeepFlavB'], scalerName= outFolder + "/model/myScaler_adv.pkl")
# %%
runPlotsTorch(Xtrain, Xval, Ytrain, Yval, Wtrain, Wval, YPredTrain, YPredVal, featuresForTraining, model, inFolder, outFolder, genMassTrain, genMassVal)

# %%
paths =["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/others",
        "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/others",
        #"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/EWKZJets",
        #"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-100to200",
        #"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-200to400",
        #"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-400to600",
        #"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-600to800",
        #"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-800toInf"
        ]
dfs= []
from functions import loadMultiParquet
featuresForTraining.remove('massHypo')
featuresForTraining+=['dijet_mass']
dfs, numEventsList, fileNumberList = loadMultiParquet(paths=paths, nReal=10, nMC=-1, columns=np.append(featuresForTraining, ['sf', 'PU_SF', 'jet1_btagDeepFlavB']), returnNumEventsTotal=True, selectFileNumberList=None, returnFileNumberList=True)

massHypothesis = [50, 70, 100, 200, 300]
massHypothesis = np.array([125]+massHypothesis)
for idx, df in enumerate(dfs):
    dfs[idx]['massHypo'] = dfs[idx]['dijet_mass'].apply(lambda x: massHypothesis[np.abs(massHypothesis - x).argmin()])

# %%
featuresForTraining.remove('dijet_mass')
featuresForTraining+=['massHypo']
#featuresForTraining.remove('dijet_mass')
dfs[0]  = scale(dfs[0], featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)
dfs[0]  = scale(dfs[0], ['jet1_btagDeepFlavB'], scalerName= outFolder + "/model/myScaler_adv.pkl" ,fit=False)
# %%
XLongTest_tensor = torch.tensor(np.float32(dfs[0][featuresForTraining].values)).float()
with torch.no_grad():  # No need to track gradients for inference
    YPredTrain_tensor = model(Xtrain_tensor)
    YPredTest_tensor = model(XLongTest_tensor)  # Output will depend on the model's architecture
YPredLongTest= YPredTest_tensor.numpy()
YLongTest=np.zeros(len(dfs[0]))
dfs[0] = unscale(dfs[0],['jet1_btagDeepFlavB'],  scalerName= outFolder + "/model/myScaler_adv.pkl" )
dfs[0]  = unscale(dfs[0], featuresForTraining, scalerName= outFolder + "/model/myScaler_adv.pkl")

# %%
checkOrthogonality(dfs[0], 'jet1_btagDeepFlavB', np.array(YPredLongTest[YLongTest==0]>0.7), np.array(YPredLongTest[YLongTest==0]<0.7), label_mask1='Prediction NN > 0.7', label_mask2='Prediction NN < 0.7', label_toPlot = 'jet1_btagDeepFlavB', bins=np.linspace(0.2783,1, 31), ax=None, axLegend=False, outName=outFolder+"/performance/orthogonalityInclusive.png" )

# %%
from functions import cut
#dfs[0] = cut([dfs[0]], 'PNN', 0.2, None)
dfs[0]['PNN'] = YPredLongTest[YLongTest==0]
mass_bins = np.linspace(40, 300, 25)
ks_p_value_PNN, p_value_PNN, chi2_values_PNN = checkOrthogonalityInMassBins(
    df=dfs[0],
    featureToPlot='PNN',
    mask1=np.array(dfs[0]['jet1_btagDeepFlavB'] >= 0.7100),
    mask2=np.array(dfs[0]['jet1_btagDeepFlavB'] < 0.7100),
    label_mask1  = 'NN High',
    label_mask2  = 'NN Low',
    label_toPlot = 'PNN',
    bins=np.linspace(0.2, 1, 21),
    mass_bins=mass_bins,
    mass_feature='dijet_mass',
    figsize=(20, 30)
)

# %%
