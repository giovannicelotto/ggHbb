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
import dcor
# Get current month and day
current_date = datetime.now().strftime("%b%d")  # This gives the format like 'Dec12'
# %%
hp = getParams()
parser = argparse.ArgumentParser(description="Script.")
parser.add_argument("-v", "--version", type=float, help="version of the model", default=0.)
parser.add_argument("-dt", "--date", type=str, help="MonthDay format e.g. Dec17", default=None)
parser.add_argument("-b", "--boosted", type=int, help="boosted class", default=4)
parser.add_argument("-s", "--sampling", type=int, help="sampling", default=0)
if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
    # Interactive: ignore CLI args, just use defaults
    args = parser.parse_args([])
else:
    # Script run from terminal: parse sys.argv
    args = parser.parse_args()

print(args)
# %%
results = {}
inFolder = "/t3home/gcelotto/ggHbb/PNN/input/data_sampling_pt%d_1D"%(args.boosted) if args.sampling else "/t3home/gcelotto/ggHbb/PNN/input/data_pt%d_1D"%(args.boosted)
outFolder = "/t3home/gcelotto/ggHbb/PNN/NN_FitScore/results/%s_%d_%s"%(current_date, args.boosted, str(args.version).replace('.', 'p'))
modelName = "model.pth"
featuresForTraining = list(np.load(outFolder+"/featuresForTraining.npy"))
featuresForTraining.remove('jet1_btagTight')
featuresForTraining.remove('jet2_btagTight')
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
# %%
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
ax.grid(True)
ax.set_ylim(0.01, 1)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
fig.savefig(outFolder + "/performance/roc125_weighted.png", bbox_inches='tight')
print("Saved", outFolder + "/performance/roc.png")



# %%
# Sig Bkg Efficiency and SIG
ts = np.linspace(0, 1, 21)
efficiencies = {
    'sigTrain':[],
    'bkgTrain':[],
    'significanceTrain':[],
    's_over_b_train':[],
    
    'sigVal':[],
    'bkgVal':[],
    'significanceVal':[],
    's_over_b_val':[],



}
for t in ts:
    num = np.sum(Wtrain[(genMassTrain==125) & (Xtrain.dijet_mass > 100) & (Xtrain.dijet_mass < 150) & (YPredTrain.reshape(-1)>t)])
    den = np.sum(Wtrain[(genMassTrain==125) & (Xtrain.dijet_mass > 100) & (Xtrain.dijet_mass < 150)])
    sigEff = num/den
    bkgEff = np.sum(Wtrain[(genMassTrain==0) & (Xtrain.dijet_mass > 100) & (Xtrain.dijet_mass < 150) & (YPredTrain.reshape(-1)>t)])/np.sum(Wtrain[(genMassTrain==0) & (Xtrain.dijet_mass > 100) & (Xtrain.dijet_mass < 150)])
    efficiencies["sigTrain"].append(sigEff)
    efficiencies["bkgTrain"].append(bkgEff)
    significanceTrain = sigEff/np.sqrt(bkgEff) if bkgEff!=0 else 0
    s_over_train = sigEff/bkgEff if bkgEff!=0 else 0
    efficiencies["significanceTrain"].append(significanceTrain)
    efficiencies["s_over_b_train"].append(s_over_train)

    sigEff = np.sum(Wval[(genMassVal==125) & (Xval.dijet_mass > 100) & (Xval.dijet_mass < 150) & (YPredVal.reshape(-1)>t)])/np.sum(Wval[(genMassVal==125) & (Xval.dijet_mass > 100) & (Xval.dijet_mass < 150)])
    bkgEff = np.sum(Wval[(genMassVal==0) & (Xval.dijet_mass > 100) & (Xval.dijet_mass < 150) & (YPredVal.reshape(-1)>t)])/np.sum(Wval[(genMassVal==0) & (Xval.dijet_mass > 100) & (Xval.dijet_mass < 150)])
    efficiencies["sigVal"].append(sigEff)
    efficiencies["bkgVal"].append(bkgEff)
    
    significanceVal = sigEff/np.sqrt(bkgEff) if bkgEff!=0 else 0
    s_over_val = sigEff/bkgEff if bkgEff!=0 else 0
    efficiencies["significanceVal"].append(significanceVal)
    efficiencies["s_over_b_val"].append(s_over_val)

print("Plotting scan S/B")
fig, ax = plt.subplots(1, 1)
ax.plot(ts, efficiencies["sigTrain"], color='red', label="Sig Train", linestyle='dashed')
ax.plot(ts, efficiencies["bkgTrain"], color='blue', label="Bkg Train", linestyle='dashed')
ax.plot(ts, efficiencies["significanceTrain"], color='green', label="Significance Train", linestyle='dashed')

ax.plot(ts, efficiencies["bkgVal"], color='blue', label="Bkg Val")
ax.plot(ts, efficiencies["sigVal"], color='red', label="Sig Val")
ax.plot(ts, efficiencies["significanceVal"], color='green', label="Significance Val")

#ax.plot(ts, efficiencies["s_over_b_train"], color='purple', label="S Over B Train", linestyle='dashed')
#ax.plot(ts, efficiencies["s_over_b_val"], color='purple', label="S Over B Val")
# Secondary Y-axis for S/B
ax2 = ax.twinx()
ax2.plot(ts, efficiencies["s_over_b_train"], color='purple', label="S Over B Train", linestyle='dashed')
ax2.plot(ts, efficiencies["s_over_b_val"], color='purple', label="S Over B Val")
ax2.set_ylabel("S / B")

# Combine legends from both axes
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

fig.savefig(outFolder + "/performance/effScan.png", bbox_inches='tight')
print("Saved ", outFolder + "/performance/effScan.png")


#ax.legend()
#fig.savefig(outFolder+"/performance/effScan.png", bbox_inches='tight')
#print("Saved ", outFolder+"/performance/effScan.png")
# %%
import numpy as np
from itertools import combinations

# Only for events in mass window
sig_mask_val = (genMassVal == 125) & (Xval.dijet_mass > 100) & (Xval.dijet_mass < 150)
bkg_mask_val = (genMassVal == 0) & (Xval.dijet_mass > 100) & (Xval.dijet_mass < 150)

sig_scores = YPredVal[sig_mask_val]
bkg_scores = YPredVal[bkg_mask_val]

ts = np.linspace(0, 1, 50)  # Finer granularity than 21 for better scan

best_sig = 0
best_cuts = None
best_n_cats = None

# Try 2-category combinations
for t1 in ts[1:-1]:
    # Define 2 bins: [0, t1], [t1, 1]
    bins = [t1, 1]

    total_sig = 0
    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1]
        sig_eff = np.sum((sig_scores >= low) & (sig_scores < high)) / len(sig_scores)
        bkg_eff = np.sum((bkg_scores >= low) & (bkg_scores < high)) / len(bkg_scores)
        if bkg_eff > 0:
            total_sig += np.sqrt(sig_eff/np.sqrt(bkg_eff))**2
    total_sig = np.sqrt(total_sig)
    print(t1, total_sig)
    if total_sig > best_sig:
        best_sig = total_sig
        best_cuts = bins
        best_n_cats = 2

# Try 3-category combinations
for t1, t2 in combinations(ts[1:-1], 2):
    if t1 >= t2:
        continue
    # Define 3 bins: [0, t1], [t1, t2], [t2, 1]
    bins = [t1, t2, 1]

    total_sig = 0
    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1]
        sig_eff = np.sum((sig_scores >= low) & (sig_scores < high)) / len(sig_scores)
        bkg_eff = np.sum((bkg_scores >= low) & (bkg_scores < high)) / len(bkg_scores)
        if bkg_eff > 0:
            total_sig += np.sqrt(sig_eff/np.sqrt(bkg_eff))**2
    total_sig = np.sqrt(total_sig)
    if total_sig > best_sig:
        best_sig = total_sig
        best_cuts = bins
        best_n_cats = 3
# %%
# Try 4-category combinations
for t1, t2, t3 in combinations(ts[1:-1], 3):
    if (t1 >= t2) | (t2 >=t3) | (t1>=t3):
        continue
    # Define 4 bins: [0, t1], [t1, t2], [t2, 1]
    bins = [0,t1, t2, t3, 1]

    total_sig = 0
    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1]
        sig_eff = np.sum((sig_scores >= low) & (sig_scores < high)) / len(sig_scores)
        bkg_eff = np.sum((bkg_scores >= low) & (bkg_scores < high)) / len(bkg_scores)
        if bkg_eff > 0:
            total_sig += np.sqrt(sig_eff/np.sqrt(bkg_eff))**2
    total_sig = np.sqrt(total_sig)
    if total_sig > best_sig:
        best_sig = total_sig
        best_cuts = bins
        best_n_cats = 4

# %%
print(f"Best total significance: {best_sig:.3f}")
print(f"Best number of categories: {best_n_cats}")
print(f"Best NN score cuts: {best_cuts}")








# %%

Xval.columns = [str(Xval.columns[_]) for _ in range((Xval.shape[1]))]
from helpers.doPlots import NNoutputs

NNoutputs(signal_predictions=YPredVal[genMassVal==125], realData_predictions=YPredVal[genMassVal==0], signalTrain_predictions=YPredTrain[genMassTrain==125], realDataTrain_predictions=YPredTrain[Ytrain==0], outName=outFolder+"/performance/NNoutput.png", log=False, doubleDisco=False, label='NN output')
# %%

# LOSS
train_loss_history = np.load(outFolder + "/model/train_loss_history.npy")
val_loss_history = np.load(outFolder + "/model/val_loss_history.npy")

doPlotLoss_Torch(train_loss_history, val_loss_history, outName=outFolder+"/performance/loss.png", earlyStop=np.argmin(val_loss_history))


plt.close('all')
print(YPredVal.reshape(-1).shape)
print(Yval.shape)
print(Xval.shape)

nn_score_bins = [0 ,0.3, 0.6, 0.68 ,1]
fig, ax = plt.subplots(1, 1)
for low, high in zip(nn_score_bins[:-1], nn_score_bins[1:]):

    maskVal = (YPredVal.reshape(-1)>low) & (YPredVal.reshape(-1)<high) & (Yval==0)
    ax.hist(Xval.dijet_mass[(maskVal)], bins=np.linspace(50, 300, 101), label=f'{low}< NN < {high} . DisCo = %.3f'%dcor.distance_correlation(YPredVal.reshape(-1)[maskVal], Xval.dijet_mass[maskVal]), density=True, histtype='step')
ax.set_xlabel("Dijet mass [GeV]")
ax.legend()
fig.savefig(outFolder+"/performance/scan_val.png", bbox_inches='tight')
nn_score_bins = [0 ,0.3, 0.6, 0.68 ,1]
fig, ax = plt.subplots(1, 1)
for low, high in zip(nn_score_bins[:-1], nn_score_bins[1:]):
    maskTrain = (YPredTrain.reshape(-1)>low) & (YPredTrain.reshape(-1)<high) & (Ytrain==0)
    ax.hist(Xtrain.dijet_mass[maskTrain], bins=np.linspace(50, 300, 101), label=f'{low} < NN < {high}. DisCo = %.3f'%dcor.distance_correlation(YPredTrain.reshape(-1)[maskTrain], Xtrain.dijet_mass[maskTrain]), density=True, histtype='step')
    ax.legend()
ax.set_xlabel("Dijet mass [GeV]")
fig.savefig(outFolder+"/performance/scan_train.png", bbox_inches='tight')
# %%
from helpers.doPlots import getShapTorch
#getShapTorch(Xtest, model, outName, nFeatures, class_names='NN output', tensor=None):
#Xtrain_tensor = torch.tensor(np.float32(Xtrain[featuresForTraining].values[(YPredTrain<1e-4).reshape(-1)])).float()
getShapTorch(Xtrain[featuresForTraining], model, outName=outFolder+"/performance/shap.png", nFeatures=10, tensor=Xtrain_tensor[:100])
# %%
sys.path.append("/t3home/gcelotto/ggHbb/scripts/plotScripts/")
from plotFeatures import plotNormalizedFeatures
# %%
plotNormalizedFeatures(data=[Xtrain[(YPredTrain<0.5).reshape(-1) & (Ytrain==0)],
                             Xtrain[(YPredTrain>0.5).reshape(-1) & (Ytrain==0)]],
                        outFile=outFolder+"/performance/features.png",
                        legendLabels=["Low", "High"],
                        colors=["blue", 'red'],
                        figsize=(30,50),
                        histtypes=["step", "step"],
                        error=False)


# %%
