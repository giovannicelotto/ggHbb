# %%
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import sys
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
#from checkOrthogonality import checkOrthogonality, checkOrthogonalityInMassBins, plotLocalPvalues
from helpers.getFeatures import getFeatures
from helpers.getParams import getParams
from helpers.getInfolderOutfolder import getInfolderOutfolder
from helpers.doPlots import runPlotsTorch, doPlotLoss_Torch
from helpers.loadSaved import loadXYWrWSaved
from helpers.scaleUnscale import scale, unscale
import torch
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
from plotFeatures import plotNormalizedFeatures
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from sklearn.feature_selection import mutual_info_regression
import glob
# Get current month and day
current_date = datetime.now().strftime("%b%d")  # This gives the format like 'Dec12'
# %%
hp = getParams()
parser = argparse.ArgumentParser(description="Script.")
# Define arguments
try:
    parser.add_argument("-v", "--version", type=float, help="version of the model", default=0)
    parser.add_argument("-dt", "--date", type=str, help="MonthDay format e.g. Dec17", default=None)
    parser.add_argument("-b", "--boosted", type=int, help="boosted class", default=1)
    parser.add_argument("-s", "--sampling", type=int, help="sampling", default=0)
    args = parser.parse_args()
    if args.version is not None:
        hp["version"] = args.version 
    if args.date is not None:
        current_date = args.date
    if args.boosted is not None:
        boosted = args.boosted
except:
    hp["version"] = 0.
    current_date="May27"
    boosted=1
    print("Interactive mode")
# %%
sampling=args.sampling
results = {}
inFolder_, outFolder = getInfolderOutfolder(name = "%s_%d_%s"%(current_date, boosted, str(hp["version"]).replace('.', 'p')), suffixResults='_mjjDisco', createFolder=False)
inFolder = "/t3home/gcelotto/ggHbb/PNN/input/data_sampling_pt%d_1D"%boosted if sampling else "/t3home/gcelotto/ggHbb/PNN/input/data_pt%d_1D"%boosted
modelName = "model.pth"
featuresForTraining = list(np.load(outFolder+"/featuresForTraining.npy"))
#featuresForTraining +=['dijet_mass']
#featuresForTraining.remove('jet2_btagDeepFlavB')
# %%
Xtrain, Xval, Ytrain, Yval, Wtrain, Wval, rWtrain, rWval, genMassTrain, genMassVal = loadXYWrWSaved(inFolder=inFolder, isTest=False)


# %%

model = torch.load(outFolder+"/model/%s"%modelName, map_location=torch.device('cpu'), weights_only=False)
model.eval()
with open(outFolder+"/model/model_summary.txt", "w") as f:
    f.write("Model Architecture:\n")
    f.write(str(model))  # Prints the architecture

    f.write("\n\nLayer-wise Details:\n")
    for name, module in model.named_modules():
        f.write(f"Layer: {name}\n")
        f.write(f"  Type: {type(module)}\n")
        if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
            f.write(f"  Weight Shape: {tuple(module.weight.shape)}\n")
        if hasattr(module, 'bias') and isinstance(module.bias, torch.Tensor):
            f.write(f"  Bias Shape: {tuple(module.bias.shape)}\n")
        f.write("\n")




Xtrain = scale(Xtrain,featuresForTraining,  scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)
Xval  = scale(Xval, featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)
#Xtest  = scale(Xtest, featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)
#rawFiles_bkg  = scale(rawFiles_bkg, featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)
#rawFiles_sig  = scale(rawFiles_sig, featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)
#print(Xtest.isna().sum())

Xtrain_tensor = torch.tensor(np.float32(Xtrain[featuresForTraining].values)).float()
Xval_tensor = torch.tensor(np.float32(Xval[featuresForTraining].values)).float()
#Xtest_tensor = torch.tensor(np.float32(Xtest[featuresForTraining].values)).float()
#rawFiles_tensor = torch.tensor(np.float32(rawFiles_bkg[featuresForTraining].values)).float()
#rawFiles_sig_tensor = torch.tensor(np.float32(rawFiles_sig[featuresForTraining].values)).float()
# %%
with torch.no_grad():  # No need to track gradients for inference
    YPredTrain = model(Xtrain_tensor).numpy()
    YPredVal = model(Xval_tensor).numpy()
    #YPredTest = model(Xtest_tensor).numpy()
    #YPredRawFiles = model(rawFiles_tensor).numpy()
    #YPredRawFiles_sig = model(rawFiles_sig_tensor).numpy()
Xtrain = unscale(Xtrain, featuresForTraining=featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl")
Xval = unscale(Xval, featuresForTraining=featuresForTraining,   scalerName =  outFolder + "/model/myScaler.pkl")
#Xtest = unscale(Xtest, featuresForTraining=featuresForTraining,   scalerName =  outFolder + "/model/myScaler.pkl")

#print(np.sum(np.isnan(YPredTest)))

# %%
####
####
####            PLOTS START HERE
####

from sklearn.metrics import roc_curve, auc
maskHiggsData_train = (genMassTrain==0) | (genMassTrain==125)
maskHiggsData_val = (genMassVal==0) | (genMassVal==125)
Xtrain['weights']=Xtrain.PU_SF * Xtrain.sf
Xval['weights']=Xval.PU_SF * Xval.sf

def plot_roc_curve(y_true, y_scores, weights, label, ax):
    fpr, tpr, _ = roc_curve(y_true, y_scores, sample_weight=weights)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{label} (Weighted AUC = {roc_auc:.3f})")

fig, ax = plt.subplots()

plot_roc_curve(Ytrain[maskHiggsData_train], YPredTrain[maskHiggsData_train].ravel(), weights=Wtrain[maskHiggsData_train], label="Train", ax=ax)
plot_roc_curve(Yval[maskHiggsData_val], YPredVal[maskHiggsData_val].ravel(),weights=Wval[maskHiggsData_val], label="Validation", ax=ax)
#plot_roc_curve((genMassTest==125).astype(int), YPredTest.ravel(), "Test", ax)
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
fig.savefig(outFolder + "/performance/roc125_weighted.png", bbox_inches='tight')
print("Saved", outFolder + "/performance/roc.png")



# %%



# %%
# Sig Bkg Efficiency and SIG
ts = np.linspace(0, 1, 21)
efficiencies = {
    'sigTrain':[],
    'bkgTrain':[],
    'significanceTrain':[],
    
    'sigVal':[],
    'bkgVal':[],
    'significanceVal':[],



}
for t in ts:
    sigEff = np.sum(YPredTrain[(genMassTrain==125) & (Xtrain.dijet_mass > 100) & (Xtrain.dijet_mass < 150)] > t)/len(YPredTrain[(genMassTrain==125) & (Xtrain.dijet_mass > 100) & (Xtrain.dijet_mass < 150)])
    bkgEff = np.sum(YPredTrain[(genMassTrain==0) & (Xtrain.dijet_mass > 100) & (Xtrain.dijet_mass < 150)] > t)/len(YPredTrain[(genMassTrain==0) & (Xtrain.dijet_mass > 100) & (Xtrain.dijet_mass < 150)])
    efficiencies["sigTrain"].append(sigEff)
    efficiencies["bkgTrain"].append(bkgEff)
    significanceTrain = sigEff/np.sqrt(bkgEff) if bkgEff!=0 else 0
    efficiencies["significanceTrain"].append(significanceTrain)

    sigEff = np.sum(YPredVal[(genMassVal==125) & (Xval.dijet_mass > 100) & (Xval.dijet_mass < 150)] > t)/len(YPredVal[(genMassVal==125) & (Xval.dijet_mass > 100) & (Xval.dijet_mass < 150)])
    bkgEff = np.sum(YPredVal[(genMassVal==0) & (Xval.dijet_mass>100) & ((Xval.dijet_mass<150))] > t)/len(YPredVal[(genMassVal==0) & (Xval.dijet_mass>100) & ((Xval.dijet_mass<150))])
    efficiencies["sigVal"].append(sigEff)
    efficiencies["bkgVal"].append(bkgEff)
    significanceVal = sigEff/np.sqrt(bkgEff) if bkgEff!=0 else 0
    efficiencies["significanceVal"].append(significanceVal)


fig, ax = plt.subplots(1, 1)
ax.plot(ts, efficiencies["sigTrain"], color='red', label="Sig Train", linestyle='dashed')
ax.plot(ts, efficiencies["bkgTrain"], color='blue', label="Bkg Train", linestyle='dashed')
ax.plot(ts, efficiencies["significanceTrain"], color='green', label="Significance Train", linestyle='dashed')

ax.plot(ts, efficiencies["bkgVal"], color='blue', label="Bkg Val")
ax.plot(ts, efficiencies["sigVal"], color='red', label="Sig Val")
ax.plot(ts, efficiencies["significanceVal"], color='green', label="Significance Val")


ax.legend()
fig.savefig(outFolder+"/performance/effScan.png", bbox_inches='tight')
print("Saved ", outFolder+"/performance/effScan.png")







# %%

Xval.columns = [str(Xval.columns[_]) for _ in range((Xval.shape[1]))]
from helpers.doPlots import NNoutputs
NNoutputs(signal_predictions=YPredVal[genMassVal==125], realData_predictions=YPredVal[genMassVal==0], signalTrain_predictions=YPredTrain[genMassTrain==125], realDataTrain_predictions=YPredTrain[Ytrain==0], outName=outFolder+"/performance/NNoutput.png", log=False, doubleDisco=False, label='NN output')
# %%

# LOSS
train_loss_history = np.load(outFolder + "/model/train_loss_history.npy")
val_loss_history = np.load(outFolder + "/model/val_loss_history.npy")

doPlotLoss_Torch(train_loss_history, val_loss_history, outName=outFolder+"/performance/loss.png", earlyStop=np.argmin(val_loss_history))