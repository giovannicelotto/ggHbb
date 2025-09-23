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
from helpers.doPlots import runPlotsTorch, doPlotLoss_Torch, plot_lossTorch
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
import os
# Get current month and day
current_date = datetime.now().strftime("%b%d")  # This gives the format like 'Dec12'
# %%
hp = getParams()
parser = argparse.ArgumentParser(description="Script.")
# Define arguments

parser.add_argument("-v", "--version", type=float, help="version of the model", default=20.01)
parser.add_argument("-dt", "--date", type=str, help="MonthDay format e.g. Dec17", default="Aug28")
parser.add_argument("-b", "--boosted", type=int, help="boosted class", default=3)
parser.add_argument("-s", "--sampling", type=int, help="sampling", default=0)



if hasattr(sys, 'ps1') or not sys.argv[1:]:
    # Interactive mode (REPL, Jupyter) OR no args provided → use defaults
    args = parser.parse_args([])
else:
    # Normal CLI usage
    args = parser.parse_args()
# %%
results = {}
inFolder_, outFolder = getInfolderOutfolder(name = "%s_%d_%s"%(args.date, args.boosted, str(args.version).replace('.', 'p')), suffixResults='_mjjDisco', createFolder=False)
inFolder = "/t3home/gcelotto/ggHbb/PNN/input/data_sampling_pt%d_1D"%args.boosted if args.sampling else "/t3home/gcelotto/ggHbb/PNN/input/data_pt%d_1D"%args.boosted
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

NNoutputs(signal_predictions=YPredVal[(genMassVal==125) & (Xval.jet1_btagTight>=0.5)& (Xval.jet2_btagTight>=0.71)], realData_predictions=YPredVal[(genMassVal==0) & (Xval.jet1_btagTight>=0.5)& (Xval.jet2_btagTight>=0.5)], signalTrain_predictions=YPredTrain[(genMassTrain==125) & (Xtrain.jet1_btagTight>=0.5)& (Xtrain.jet2_btagTight>0.5)], realDataTrain_predictions=YPredTrain[(Ytrain==0) & (Xtrain.jet1_btagTight>=0.5) & (Xtrain.jet2_btagTight>0.5)], outName=outFolder+"/performance/NNoutput_tight.png", log=False, doubleDisco=False, label='NN output')
# %%

# LOSS
train_loss_history = np.load(outFolder + "/model/train_loss_history.npy")
val_loss_history = np.load(outFolder + "/model/val_loss_history.npy")
if os.path.exists(outFolder + "/model/train_classifier_loss_history.npy"):
    train_classifier_loss_history = np.load(outFolder + "/model/train_classifier_loss_history.npy")
    val_classifier_loss_history = np.load(outFolder + "/model/val_classifier_loss_history.npy")
    train_dcor_loss_history = np.load(outFolder + "/model/train_disco_loss_history.npy")
    val_dcor_loss_history = np.load(outFolder + "/model/val_disco_loss_history.npy")
    plot_lossTorch(train_loss_history, val_loss_history, 
                train_classifier_loss_history, val_classifier_loss_history,
                train_dcor_loss_history, val_dcor_loss_history,
                train_closure_loss_history=None, val_closure_loss_history=None,
                outFolder=outFolder, gpu=False)
doPlotLoss_Torch(train_loss_history, val_loss_history, outName=outFolder+"/performance/loss.png", earlyStop=np.argmin(val_loss_history))

plt.close('all')
print(YPredVal.reshape(-1).shape)
print(Yval.shape)
print(Xval.shape)

nn_score_bins = [0 ,0.3, 0.6, 0.7 ,1]
fig, ax = plt.subplots(1, 1)
for low, high in zip(nn_score_bins[:-1], nn_score_bins[1:]):

    maskVal = (YPredVal.reshape(-1)>low) & (YPredVal.reshape(-1)<high) & (Yval==0)
    ax.hist(Xval.dijet_mass[(maskVal)], bins=np.linspace(50, 300, 101), label=f'{low}< NN < {high} . DisCo = %.3f'%dcor.distance_correlation(YPredVal.reshape(-1)[maskVal], Xval.dijet_mass[maskVal]), density=True, histtype='step')
ax.set_xlabel("Dijet mass [GeV]")
ax.legend()
fig.savefig(outFolder+"/performance/scan_val.png", bbox_inches='tight')
nn_score_bins = [0 ,0.3, 0.6, 0.7 ,1]
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
nEvents = 3000
subdf_0 =Xtrain[genMassTrain==0][featuresForTraining].iloc[:int(nEvents/2)]
subdf_1 = Xtrain[genMassTrain==125][featuresForTraining].iloc[:int(nEvents/2)]
subdf_0_scaled = scale(subdf_0,featuresForTraining,  scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)
subdf_1_scaled = scale(subdf_1,featuresForTraining,  scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)
subTensor_0 =  torch.tensor(np.float32(subdf_0_scaled.values)).float()
subTensor_1 =  torch.tensor(np.float32(subdf_1_scaled.values)).float()
subTensor = torch.cat([subTensor_0, subTensor_1])
# %%
import random
from math import comb

def monte_carlo_shap(model, x_sample, x_baseline, n_samples=100):
    """
    Compute approximate SHAP values for a single sample using Monte Carlo.

    Args:
        model: function that takes an input vector and returns scalar output.
        x_sample: 1D array of shape (N_features,) - the sample to explain
        x_baseline: 1D array of shape (N_features,) - reference input
        n_samples: number of random subsets per feature

    Returns:
        phi: 1D array of approximate SHAP values
    """
    N = len(x_sample)
    phi = np.zeros(N)

    for i in range(N):
        contribs = []
        for _ in range(n_samples):
            # Random subset of other features
            other_features = [j for j in range(N) if j != i]
            S_size = random.randint(0, N-1)
            S = random.sample(other_features, S_size)

            # Input with features in S from sample, others from baseline
            x_S = x_baseline.copy()
            x_S[S] = x_sample[S]

            # Input with S + i from sample
            x_Si = x_S.copy()
            x_Si[i] = x_sample[i]

            # Model outputs
            y_S = model(x_S)
            y_Si = model(x_Si)

            # Weight by combinatorial factor (optional, can use uniform for simplicity)
            # weight = comb(len(S), N-1)
            # contribs.append(weight * (y_Si - y_S))
            contribs.append(y_Si - y_S)

        # Average contribution for feature i
        phi[i] = np.mean(contribs)

    return phi

import torch

def model_forward(x_input):
    """
    Wrapper to compute model output for a single sample.

    Args:
        x_input : 1D numpy array, shape (N_features,)

    Returns:
        float : model output (scalar)
    """
    # Convert to torch tensor and add batch dimension
    x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0)  # shape (1, N_features)

    # Make sure model is in eval mode
    model.eval()

    with torch.no_grad():
        y = model(x_tensor)  # output tensor
    # If output is 1D (binary classification), extract scalar
    return y.item()

import numpy as np
import matplotlib.pyplot as plt

def plot_shap_bar(phi_all, feature_names, top_n=10, out_file=None):
    """
    Produce a bar plot of feature importances from SHAP values.

    Args:
        phi_all : numpy array
            SHAP values. Shape = (n_events, n_features) or (n_features,)
        feature_names : list of str
            Names of features, same order as columns in phi_all
        top_n : int
            Number of top features to display
        out_file : str or None
            If given, save plot to this file
    """
    # Ensure 2D array
    phi_all = np.array(phi_all)
    if phi_all.ndim == 1:
        phi_all = phi_all.reshape(1, -1)
    
    # Global importance: mean absolute SHAP
    importance = np.mean(np.abs(phi_all), axis=0)

    # Rank features
    indices = np.argsort(importance)[::-1][:top_n]
    top_features = np.array(feature_names)[indices]
    top_values = importance[indices]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(top_features)), top_values, color='C0')
    ax.set_xticks(np.arange(len(top_features)))
    ax.set_xticklabels(top_features, rotation=90, ha='right')
    ax.set_ylabel('Mean(|SHAP|)')
    ax.set_title('Top {} Feature Importances'.format(top_n))
    plt.tight_layout()

    if out_file is not None:
        plt.savefig(out_file)
    plt.show()

    # Return a dictionary of feature -> importance
    return dict(zip(top_features, top_values))

# -----------------------
# Example usage:

# phi_all: shape (n_events, n_features) or (n_features,) for single event
# feature_names: list of feature names


# Example with 62 features
N_features = len(featuresForTraining)
x_sample = subTensor[0].numpy()           # take one event to explain
x_baseline = np.mean(subTensor.numpy(), axis=0)  # baseline = average of background

phi = monte_carlo_shap(model_forward, x_sample, x_baseline, n_samples=3000)
import numpy as np

# phi: 1D array, length = number of features
# feature_names: list of feature names, same order as phi

# Get absolute values to measure magnitude of impact
phi_abs = np.abs(phi)

# Sort features by descending importance
indices = np.argsort(phi_abs)[::-1]
top_features = np.array(featuresForTraining)[indices]
top_values = phi_abs[indices]

# Display top 10 features
for f, v in zip(top_features[:10], top_values[:10]):
    print(f"{f}: {v:.4f}")

# phi[i] = approximate SHAP value for feature i
plot_shap_bar(phi_abs, featuresForTraining, top_n=15, out_file=outFolder+"/performance/shapMC.png")
#getShapTorch(Xtrain[featuresForTraining], model, outName=outFolder+"/performance/shap.png", nFeatures=10, tensor=subTensor_1)
# %%
sys.path.append("/t3home/gcelotto/ggHbb/scripts/plotScripts/")
from plotFeatures import plotNormalizedFeatures
# %%
plotNormalizedFeatures(data=[Xtrain[(YPredTrain<0.5).reshape(-1) & (Ytrain==0)],
                             Xtrain[(YPredTrain>0.5).reshape(-1) & (Ytrain==0)]],
                        outFile=outFolder+"/performance/features_low_vs_High.png",
                        legendLabels=["Low", "High"],
                        colors=["blue", 'red'],
                        figsize=(30,50),
                        histtypes=["step", "step"],
                        error=False)

# %%
plotNormalizedFeatures(data=[Xtrain[(genMassTrain==125)],
                             Xtrain[(genMassTrain==0)]],
                        outFile=outFolder+"/performance/features_SvsB.png",
                        legendLabels=["S", "B"],
                        colors=["blue", 'red'],
                        figsize=(30,50),
                        histtypes=["step", "step"],
                        error=False)
# %%
