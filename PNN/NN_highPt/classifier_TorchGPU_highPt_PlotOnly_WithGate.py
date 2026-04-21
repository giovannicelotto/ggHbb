# %%
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import sys
import os
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
import yaml
from datetime import datetime
from sklearn.feature_selection import mutual_info_regression
import glob
import dcor
import os
from datetime import datetime
# Get current month and day
current_date = datetime.now().strftime("%b%d")  # This gives the format like 'Dec12'
# %%
hp = getParams()
parser = argparse.ArgumentParser(description="Script.")
# Define arguments

parser.add_argument("-v", "--version", type=float, help="version of the model", default=20.)
parser.add_argument("-dt", "--date", type=str, help="MonthDay format e.g. Dec17", default=current_date)
parser.add_argument("-b", "--boosted", type=int, help="boosted class", default=12)
parser.add_argument("-s", "--sampling", type=int, help="sampling", default=0)
parser.add_argument("-e", "--epoch", type=int, help="epoch", default=183)



if hasattr(sys, 'ps1') or not sys.argv[1:]:
    # Interactive mode (REPL, Jupyter) OR no args provided → use defaults
    args = parser.parse_args([])
    print("[INFO] Interactive mode")
else:
    # Normal CLI usage
    args = parser.parse_args()
# %%
results = {}
inFolder_, outFolder = getInfolderOutfolder(name = "%s_%d_%s"%(args.date, args.boosted, str(args.version).replace('.', 'p')), suffixResults='_mjjDisco/gateSV', createFolder=False)
inFolder = f"/work/gcelotto/ggHbb_work/input_NN/data_pt{args.boosted}_1D"
modelName = "model.pth" if args.epoch==-1 else "model_e%d.pth"%args.epoch
featuresForTraining = list(np.load(outFolder+"/model/featuresForTraining.npy"))
#featuresForTraining +=['dijet_mass']
#featuresForTraining.remove('jet2_btagDeepFlavB')
# %%
print("Loading dataset...")
Xtrain, Xval, Ytrain, Yval, Wtrain, Wval, rWtrain, rWval, genMassTrain, genMassVal = loadXYWrWSaved(inFolder=inFolder, isTest=False, btagWP="M")
print("Loading dataset... Done")
# %%
print("Loading model... ")
model = torch.load(outFolder+"/model/%s"%modelName, map_location=torch.device('cpu'), weights_only=False)
print("Loading model... Done ")
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



print("Scaling datasets... ")
Xtrain = scale(Xtrain,featuresForTraining,  scalerName= outFolder + "/model/myScaler.pkl" ,fit=False, features_to_exclude=['jet1_has_sv', 'jet2_has_sv'])
Xval  = scale(Xval, featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False, features_to_exclude=['jet1_has_sv', 'jet2_has_sv'])
#Xtest  = scale(Xtest, featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)
#rawFiles_bkg  = scale(rawFiles_bkg, featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)
#rawFiles_sig  = scale(rawFiles_sig, featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)
#print(Xtest.isna().sum())
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

main_features = [f for f in featuresForTraining if f not in sv1_features + sv2_features]
main_features+=['jet1_has_sv', 'jet2_has_sv']

Xtrain_main = torch.tensor(Xtrain[main_features].values, dtype=torch.float32)
Xtrain_sv1  = torch.tensor(Xtrain[sv1_features].values, dtype=torch.float32)
Xtrain_sv2  = torch.tensor(Xtrain[sv2_features].values, dtype=torch.float32)

Xval_main = torch.tensor(Xval[main_features].values, dtype=torch.float32)
Xval_sv1  = torch.tensor(Xval[sv1_features].values, dtype=torch.float32)
Xval_sv2  = torch.tensor(Xval[sv2_features].values, dtype=torch.float32)

#Xtrain_tensor = torch.tensor(np.float32(Xtrain[featuresForTraining].values)).float()
#Xval_tensor = torch.tensor(np.float32(Xval[featuresForTraining].values)).float()
# %%
print("Computing predictions...")
with torch.no_grad():  # No need to track gradients for inference
    YPredTrain = model(Xtrain_main, Xtrain_sv1, Xtrain_sv2).numpy()
    YPredVal = model(Xval_main, Xval_sv1, Xval_sv2).numpy()
# %%
print("Unscaling predictions...")
Xtrain = unscale(Xtrain, featuresForTraining=featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl", features_to_exclude=['jet1_has_sv', 'jet2_has_sv'])
Xval = unscale(Xval, featuresForTraining=featuresForTraining,   scalerName =  outFolder + "/model/myScaler.pkl", features_to_exclude=['jet1_has_sv', 'jet2_has_sv'])
#Xtest = unscale(Xtest, featuresForTraining=featuresForTraining,   scalerName =  outFolder + "/model/myScaler.pkl")

#print(np.sum(np.isnan(YPredTest)))

# %%
####
####
####            PLOTS START HERE
####
print("Plotting")
from sklearn.metrics import roc_curve, auc
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
maskHiggsData_train = (genMassTrain==0) | (genMassTrain==125)
maskHiggsData_val = (genMassVal==0) | (genMassVal==125)
Xtrain['weights']=Xtrain.flat_weight
Xval['weights']=Xval.flat_weight





fig, ax = plt.subplots(1, 1)
bins_nn, bins_dijet_pt = np.linspace(0, 1, 101), np.linspace(80, 500, 101)
print(YPredTrain[genMassTrain==125])
print(Xtrain.dijet_pt[genMassTrain==125])
ax.hist2d(YPredTrain[genMassTrain==125].reshape(-1), Xtrain.dijet_pt[genMassTrain==125].values, bins=(bins_nn, bins_dijet_pt), cmap="viridis", norm=LogNorm())
ax.set_xlabel("NN")
ax.set_ylabel("dijet pt")
ax.set_title("Train Sample (Signal)")
fig.savefig(outFolder + "/performance/NN_vs_dijetPt.png", bbox_inches='tight')





# %%
def plot_roc_curve(y_true, y_scores, weights, label, ax, color=None, linestyle='solid'):
    fpr, tpr, _ = roc_curve(y_true, y_scores, sample_weight=weights)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{label} (Weighted AUC = {roc_auc:.3f})", color=color, linestyle=linestyle)
    return roc_auc
# %%
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
plt.close('all')
def partial_auc(fpr, tpr, max_fpr=0.005):
    """
    Compute partial AUC up to max_fpr.
    """
    # Ensure arrays are sorted
    fpr = np.array(fpr)
    tpr = np.array(tpr)

    # Cut at max_fpr
    mask = fpr <= max_fpr

    fpr_cut = fpr[mask]
    tpr_cut = tpr[mask]

    # If last point is below max_fpr, interpolate one point at max_fpr
    if fpr_cut[-1] < max_fpr:
        tpr_interp = np.interp(max_fpr, fpr, tpr)
        fpr_cut = np.append(fpr_cut, max_fpr)
        tpr_cut = np.append(tpr_cut, tpr_interp)

    return np.trapz(tpr_cut, fpr_cut)
def plot_roc_curve_zoom(y_true, y_scores, weights, label, ax, color=None, linestyle='solid'):
    fpr, tpr, _ = roc_curve(y_true, y_scores, sample_weight=weights)

    roc_auc = auc(fpr, tpr)
    p_auc = partial_auc(fpr, tpr, max_fpr=0.005)

    ax.plot(
        fpr, tpr,
        label=f"{label} (AUC = {roc_auc:.3f}, pAUC@0.005 = {p_auc:.1e})",
        color=color,
        linestyle=linestyle
    )

    return roc_auc, p_auc
fig, ax = plt.subplots()

plot_roc_curve_zoom(Ytrain[maskHiggsData_train], YPredTrain[maskHiggsData_train].ravel(), weights=Wtrain[maskHiggsData_train], label="Train", ax=ax)
plot_roc_curve_zoom(Yval[maskHiggsData_val], YPredVal[maskHiggsData_val].ravel(),weights=Wval[maskHiggsData_val], label="Validation", ax=ax)
#plot_roc_curve((genMassTest==125).astype(int), YPredTest.ravel(), "Test", ax)
ax.plot([0, 1], [0, 1], 'k--')
ax.grid(True)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
print("Saved", outFolder + "/performance/roc125_weighted.png")
ax.set_xlim(0, 0.005)
ax.set_ylim(0,0.1)
ax.hlines(y=0.05, xmin=0, xmax=1, colors='red', linestyles='dashed', label='Bkg Eff = 5%')
fig.savefig(outFolder + "/performance/roc125_weighted_zoomed.png", bbox_inches='tight')
print("Saved", outFolder + "/performance/roc125_weighted_zoomed.png")


# -----------------
fig, ax = plt.subplots(1,1)
maskHiggsData_train = ((genMassTrain==0) | (genMassTrain==125) ) & (Xtrain['dijet_mass']>100) & (Xtrain['dijet_mass']<150)
maskHiggsData_val = ((genMassVal==0) | (genMassVal==125) ) & (Xval['dijet_mass']>100) & (Xval['dijet_mass']<150)
plot_roc_curve(Ytrain[maskHiggsData_train], YPredTrain[maskHiggsData_train].ravel(), weights=Wtrain[maskHiggsData_train], label="Train", ax=ax)
plot_roc_curve(Yval[maskHiggsData_val], YPredVal[maskHiggsData_val].ravel(),weights=Wval[maskHiggsData_val], label="Validation", ax=ax)
ax.set_xlim(0, 0.02)
ax.set_ylim(0,0.1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
ax.grid(True)
fig.savefig(outFolder + "/performance/roc125_weighted_zoomed_mjjWindow.png", bbox_inches='tight')
print("Saved", outFolder + "/performance/roc125_weighted_zoomed_mjjWindow.png")

# -----------------
# %%
fig, ax = plt.subplots(1,1)
masses = {
    50 :    {"color": "C0"},
    70 :    {"color": "C1"},
    100 :   {"color": "C2"},
    125 :   {"color": "C3"},
    200 :   {"color": "C4"},
    300 :   {"color": "C5"},
}
for mass in [50,70,100,125,200,300]:
    masses[mass]["AUC_train"] = plot_roc_curve(Ytrain[(genMassTrain==0) | (genMassTrain==mass)], YPredTrain[(genMassTrain==0) | (genMassTrain==mass)].ravel(), weights=Wtrain[(genMassTrain==0) | (genMassTrain==mass)], label="Train m=%d"%mass, ax=ax, color=f"{masses[mass]['color']}", linestyle='solid')
    masses[mass]["AUC_val"] = plot_roc_curve(Yval[(genMassVal==0) | (genMassVal==mass)], YPredVal[(genMassVal==0) | (genMassVal==mass)].ravel(),weights=Wval[(genMassVal==0) | (genMassVal==mass)], label="Validation m=%d"%mass, ax=ax, color=f"{masses[mass]['color']}", linestyle='dashed')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
fig.savefig(outFolder + "/performance/rocSpin0_weighted.png", bbox_inches='tight')
print("Saved", outFolder + "/performance/rocSpin0_weighted.png")
# %%
fig, ax = plt.subplots(1, 1)
x_ = np.array(list(masses.keys()))
y_train = np.array([masses[mass]["AUC_train"] for mass in masses.keys()])
y_val = np.array([masses[mass]["AUC_val"] for mass in masses.keys()])
ax.plot(x_, y_train, marker='o', label='Train')
ax.plot(x_, y_val, marker='o', label='Val')
ax.set_xlabel('Mass (GeV)')
ax.set_ylabel('AUC per Mass Point')
ax.legend()
ax.set_ylim(0, 1)
fig.savefig(outFolder + "/performance/AUC_vs_Mass.png", bbox_inches='tight')
print("Saved", outFolder + "/performance/AUC_vs_Mass.png")
# %%
# Sig Bkg Efficiency and SIG
ts = np.linspace(0, 1, 41)
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
    num = np.sum(Xtrain.flat_weight[(genMassTrain==125) & (Xtrain.dijet_mass > 100) & (Xtrain.dijet_mass < 150) & (YPredTrain.reshape(-1)>t)])
    den = np.sum(Xtrain.flat_weight[(genMassTrain==125) & (Xtrain.dijet_mass > 100) & (Xtrain.dijet_mass < 150)])
    sigEff = num/den
    bkgEff = np.sum((genMassTrain==0) & (Xtrain.dijet_mass > 100) & (Xtrain.dijet_mass < 150) & (YPredTrain.reshape(-1)>t))/np.sum((genMassTrain==0) & (Xtrain.dijet_mass > 100) & (Xtrain.dijet_mass < 150))
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



# %%

from helpers.doPlots import getShapTorch
import shap

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
    #plt.show()

    # Return a dictionary of feature -> importance
    return dict(zip(top_features, top_values))



nEvents = 100
subTensor_main = Xtrain_main[:nEvents]
subTensor_sv1 = Xtrain_sv1[:nEvents]
subTensor_sv2 = Xtrain_sv2[:nEvents]
# %%
import torch.nn as nn
X_all = torch.cat([subTensor_main, subTensor_sv1, subTensor_sv2], dim=1)
class WrappedModel(nn.Module):
    def __init__(self, model, dim_main, dim_sv):
        super().__init__()
        self.model = model
        self.dim_main = dim_main
        self.dim_sv = dim_sv

    def forward(self, x):
        x_main = x[:, :self.dim_main]
        x_sv1  = x[:, self.dim_main:self.dim_main + self.dim_sv]
        x_sv2  = x[:, self.dim_main + self.dim_sv:]

        return self.model(x_main, x_sv1, x_sv2)
dim_main = Xtrain_main.shape[1]
dim_sv   = Xtrain_sv1.shape[1]

wrapped_model = WrappedModel(model, dim_main, dim_sv)
wrapped_model.eval()
# %%
explainer = shap.GradientExplainer(wrapped_model, X_all)

shap_values = explainer.shap_values(X_all)


# %%
# explain a subset (recommended for speed)


phi_all = abs(shap_values).mean(axis=0).reshape(-1)
jet1_sv_features__ = [
    "jet1_sv_pt_prime",
    "jet1_sv_mass_prime",
    "jet1_sv_Ntrk",
]

jet2_sv_features__ = [
    "jet2_sv_pt_prime",
    "jet2_sv_mass_prime",
    "jet2_sv_Ntrk",
]
featuresForTraining_main = [f for f in featuresForTraining if f not in jet1_sv_features__ + jet2_sv_features__]
main_features = featuresForTraining_main  # without SV features


feature_names_all = main_features + sv1_features + sv2_features
# Sort features by descending importance
indices = np.argsort(phi_all)[::-1]
top_features = np.array(feature_names_all)[indices]
top_values = phi_all[indices]

# Display top 10 features
#for f, v in zip(top_features[:10], top_values[:10]):
#    print(f"{f}: {v:.4f}")

plot_shap_bar(phi_all, feature_names_all, top_n=40, out_file=outFolder+"/performance/shapMC.png")


import numpy as np
from itertools import combinations





# %%

Xval.columns = [str(Xval.columns[_]) for _ in range((Xval.shape[1]))]
from helpers.doPlots import NNoutputs

NNoutputs(signal_predictions=YPredVal[genMassVal==125], realData_predictions=YPredVal[genMassVal==0], signalTrain_predictions=YPredTrain[genMassTrain==125], realDataTrain_predictions=YPredTrain[Ytrain==0], outName=outFolder+"/performance/NNoutput.png", log=False, doubleDisco=False, label='NN output')

NNoutputs(signal_predictions=YPredVal[(genMassVal==125) & (Xval.jet1_btagWP>=3)& (Xval.jet2_btagWP>=3)], realData_predictions=YPredVal[(genMassVal==0) & (Xval.jet1_btagWP>=3)& (Xval.jet2_btagWP>=3)], signalTrain_predictions=YPredTrain[(genMassTrain==125) & (Xtrain.jet1_btagWP>=3)& (Xtrain.jet2_btagWP>=3)], realDataTrain_predictions=YPredTrain[(Ytrain==0) & (Xtrain.jet1_btagWP>=3) & (Xtrain.jet2_btagWP>=3)], outName=outFolder+"/performance/NNoutput_tight.png", log=False, doubleDisco=False, label='NN output')
# %%

# LOSS
if os.path.exists(outFolder + "/model/train_loss_history.npy"):
    train_loss_history = np.load(outFolder + "/model/train_loss_history.npy")
    val_loss_history = np.load(outFolder + "/model/val_loss_history.npy")
    if os.path.exists(outFolder + "/model/train_classifier_loss_history.npy"):
        train_classifier_loss_history = np.load(outFolder + "/model/train_classifier_loss_history.npy")
        val_classifier_loss_history = np.load(outFolder + "/model/val_classifier_loss_history.npy")
        if os.path.exists(outFolder + "/model/train_shape_loss_history.npy"):
            train_dcor_loss_history = np.load(outFolder + "/model/train_shape_loss_history.npy")
            val_dcor_loss_history = np.load(outFolder + "/model/val_shape_loss_history.npy")
        else:
            train_dcor_loss_history = np.load(outFolder + "/model/train_disco_loss_history.npy")
            val_dcor_loss_history = np.load(outFolder + "/model/val_disco_loss_history.npy")
        plot_lossTorch(train_loss_history, val_loss_history, 
                    train_classifier_loss_history, val_classifier_loss_history,
                    train_dcor_loss_history, val_dcor_loss_history,
                    train_closure_loss_history=None, val_closure_loss_history=None,
                    outFolder=outFolder, gpu=False)
    doPlotLoss_Torch(train_loss_history, val_loss_history, outName=outFolder+"/performance/loss.png", earlyStop=np.argmin(val_loss_history))
else:
    print("No loss history found.")
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




nn_score_bins = [0.7 ,0.75, 0.8, 0.85 ,0.9, 1]
fig, ax = plt.subplots(1, 1)
for low, high in zip(nn_score_bins[:-1], nn_score_bins[1:]):
    maskTrain = (YPredTrain.reshape(-1)>low) & (Ytrain==0)
    ax.hist(Xtrain.dijet_mass[maskTrain], bins=np.linspace(50, 200, 81), label=f'{low} < NN . DisCo = %.3f'%dcor.distance_correlation(YPredTrain.reshape(-1)[maskTrain], Xtrain.dijet_mass[maskTrain]), density=True, histtype='step')
    ax.legend()
ax.set_xlabel("Dijet mass [GeV]")
fig.savefig(outFolder+"/performance/scan_train_highNN.png", bbox_inches='tight')








# target background efficiencies
bkg_effs = [0.1, 0.05, 0.01, 0.005, 0.0025, 0.001]
y_pred = YPredTrain.reshape(-1)
bkg_mask = (Ytrain == 0)
sig_mask = (genMassTrain == 125)
nn_thresholds = {
    eff: np.quantile(y_pred[bkg_mask], 1 - eff)
    for eff in bkg_effs
}

fig, ax = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
for idx, (bkg_eff, thr) in enumerate(nn_thresholds.items()):

    # background passing the cut
    mask_bkg = (y_pred > thr) & (bkg_mask) 

    # signal efficiency
    sig_eff = np.sum(Xtrain['flat_weight'][(sig_mask) & (Xtrain.dijet_mass>100) & (Xtrain.dijet_mass<150)& (y_pred>thr)])/np.sum(Xtrain['flat_weight'][(sig_mask)  & (Xtrain.dijet_mass>100) & (Xtrain.dijet_mass<150)])

    disco = dcor.distance_correlation(
        y_pred[mask_bkg],
        Xtrain.dijet_mass[mask_bkg]
    )
    if idx< len(bkg_effs)//2:
        ax[0].hist(
            Xtrain.dijet_mass[mask_bkg],
            bins=np.linspace(50, 200, 81),
            density=True,
            histtype='step',
            label=(
                f"Bkg eff = {100*bkg_eff:.2f}%  "
                f"Sig eff = {100*sig_eff:.2f}%  "
                f"NN thr = {thr:.3f} | "
                f"DisCo = {disco:.3f}"
            )
        )
        ax[0].set_xlabel("Dijet mass [GeV]")
        ax[0].legend(fontsize=14)
    else:

        ax[1].hist(
            Xtrain.dijet_mass[mask_bkg],
            bins=np.linspace(50, 200, 81),
            density=True,
            histtype='step',
            label=(
                f"Bkg eff = {100*bkg_eff:.2f}%  "
                f"Sig eff = {100*sig_eff:.2f}%  "
                #f"NN thr = {thr:.3f} | "
                f"DisCo = {disco:.3f}"
            )
        )
        ax[1].set_xlabel("Dijet mass [GeV]")
        ax[1].legend(fontsize=10)


fig.savefig(outFolder + "/performance/scan_train_highNN_bkgRejection.png", bbox_inches='tight')

#fig, ax = plt.subplots(1, 1, figsize=(10, 10))

#for idx, (bkg_eff, thr) in enumerate(nn_thresholds.items()):
#    ...
#    ax.hist(...)

#fig.savefig(outFolder + "/performance/scan_train_all.png", bbox_inches='tight')
# %%


#bkg_effs = [0.01, 0.005, 0.0025, 0.001]
#from functions import loadMultiParquet_Data_new, getCommonFilters
#data_5d = loadMultiParquet_Data_new(dataTaking=[21], nReals=-1, columns=featuresForTraining, filters=getCommonFilters(btagWP='T', cutDijet=True, ttbarCR=False))[0]
## %%
#data_5d = scale(data_5d[0], featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)
#data_5d_tensor = torch.tensor(np.float32(data_5d[featuresForTraining].values)).float()
## %%
#with torch.no_grad():  # No need to track gradients for inference
#    YPred_data_5d = model(data_5d_tensor).numpy()
#fig, ax = plt.subplots(1, 1)
#
#for bkg_eff, thr in nn_thresholds.items():
#    if bkg_eff < 0.0003:
#        continue
#
#    # background passing the cut
#    mask_bkg = (y_pred > thr) & bkg_mask
#
#    # signal efficiency
#    sig_eff = np.sum(Xtrain['flat_weight'][(sig_mask) & (Xtrain.dijet_mass>100) & (Xtrain.dijet_mass<150)& (y_pred>thr)])/np.sum(Xtrain['flat_weight'][(sig_mask)  & (Xtrain.dijet_mass>100) & (Xtrain.dijet_mass<150)])
#
#    disco = dcor.distance_correlation(
#        y_pred[mask_bkg],
#        Xtrain.dijet_mass[mask_bkg]
#    )
#
#    ax.hist(
#        Xtrain.dijet_mass[mask_bkg],
#        bins=np.linspace(60, 240, 51),
#        density=True,
#        histtype='step',
#        label=(
#            f"Bkg eff = {100*bkg_eff:.3g}% | "
#            f"Sig eff = {100*sig_eff:.2f}% | "
#            f"NN thr = {thr:.3f} | "
#            f"DisCo = {disco:.3f}"
#        )
#    )
#ax.set_xlabel("Dijet mass [GeV]")
#ax.legend()
#fig.savefig(outFolder + "/performance/scan_train_highNN_bkgRejection_5d_T.png", bbox_inches='tight')

# %%

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
                        weights=[Xtrain[(genMassTrain==125)].flat_weight,
                                 Xtrain[(genMassTrain==0)].flat_weight],
                        histtypes=["step", "step"],
                        error=False)
# %%
