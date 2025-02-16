# %%
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import sys
sys.path.append("/t3home/gcelotto/ggHbb/PNN/helpers")
from checkOrthogonality import checkOrthogonality, checkOrthogonalityInMassBins, plotLocalPvalues
from helpers.getFeatures import getFeatures
from helpers.getParams import getParams
from helpers.getInfolderOutfolder import getInfolderOutfolder
from helpers.doPlots import runPlotsTorch, plot_lossTorch
from helpers.loadSaved import loadXYWrWSaved
import torch
from helpers.scaleUnscale import scale, unscale
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
from plotFeatures import plotNormalizedFeatures
import numpy as np
import argparse
from datetime import datetime
from sklearn.feature_selection import mutual_info_regression
from helpers.doPlots import ggHscoreScan, getShapTorch
sys.path.append("/t3home/gcelotto/ggHbb/PNN/helpers")
from checkOrthogonality import checkOrthogonality, checkOrthogonalityInMassBins, plotLocalPvalues
from dcorLoss import Classifier

# Get current month and day
current_date = datetime.now().strftime("%b%d")  # This gives the format like 'Dec12'
# %%
hp = getParams()
parser = argparse.ArgumentParser(description="Script.")
# Define arguments
try:
    parser.add_argument("-l", "--lambdaCor", type=float, help="lambda for penalty term", default=None)
    parser.add_argument("-dt", "--date", type=str, help="MonthDay format e.g. Dec17", default=None)
    args = parser.parse_args()
    if args.lambdaCor is not None:
        hp["lambda_reg"] = args.lambdaCor 
    if args.date is not None:
        current_date = args.date
except:
    current_date = "Jan24"
    hp["lambda_reg"] = 900.0
    print("Interactive mode")
# %%
results = {}
inFolder, outFolder = getInfolderOutfolder(name = "%s_%s"%(current_date, str(hp["lambda_reg"]).replace('.', 'p')), suffixResults='DoubleDisco', createFolder=False)
modelName1, modelName2 = "nn1.pth", "nn2.pth"

# %%
Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, Wtrain, Wval, Wtest, rWtrain, rWval, genMassTrain, genMassVal, genMassTest = loadXYWrWSaved(inFolder=inFolder+"/data")
columnsToRead = getFeatures(outFolder=None,  massHypo=True)[1]
featuresForTraining = np.load(outFolder+"/featuresForTraining.npy")
print(featuresForTraining)
if 'bin_center' in featuresForTraining:
    mass_bins = np.load(outFolder + "/mass_bins.npy")
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

    bin_indices = np.digitize(Xtest['dijet_mass'].values, mass_bins) - 1
    Xtest['bin_center'] = np.where(
        (bin_indices >= 0) & (bin_indices < len(bin_centers)),  # Ensure valid indices
        np.array(bin_centers)[bin_indices],
        np.nan  # Assign NaN for out-of-range dijet_mass
    )
advFeatureTrain = np.load(inFolder+"/data/advFeatureTrain.npy")     
advFeatureVal   = np.load(inFolder+"/data/advFeatureVal.npy")


state_dict1 = torch.load(outFolder + "/model/nn1.pth", map_location=torch.device('cpu'))
state_dict2 = torch.load(outFolder + "/model/nn2.pth", map_location=torch.device('cpu'))
#
## Remove the 'module.' prefix if it exists
state_dict1 = {k.replace('module.', ''): v for k, v in state_dict1.items()}
state_dict2 = {k.replace('module.', ''): v for k, v in state_dict2.items()}
def get_layer_sizes(state_dict, n_input_features):
    layer_sizes = []
    current_features = n_input_features

    for key, tensor in state_dict.items():
        if "weight" in key and tensor.dim() == 2:
            weight_shape = state_dict[key].shape
            if weight_shape[1] == current_features:  # Ensure it's a valid layer
                layer_sizes.append(weight_shape[0])
                current_features = weight_shape[0]  # Update for the next layer

    return layer_sizes
layer_sizes1 = get_layer_sizes(state_dict1, n_input_features=len(featuresForTraining))
layer_sizes2 = get_layer_sizes(state_dict2, n_input_features=len(featuresForTraining))
nn1 = Classifier(input_dim=Xtrain[featuresForTraining].shape[1], nNodes=layer_sizes1[:-1])
nn2 = Classifier(input_dim=Xtrain[featuresForTraining].shape[1], nNodes=layer_sizes2[:-1])

## Now load the state_dict into the model
nn1.load_state_dict(state_dict1)
nn2.load_state_dict(state_dict2)

#nn1 = torch.load(outFolder+"/model/%s"%modelName1, map_location=torch.device('cpu'))
#nn2 = torch.load(outFolder+"/model/%s"%modelName2, map_location=torch.device('cpu'))

nn1.eval()
nn2.eval()
with open(outFolder+"/model/model_summary.txt", "w") as f:
    f.write("Model 1:\n")
    f.write(str(nn1))  # Prints the architecture

    f.write("\n\nLayer-wise Details:\n")
    for name, module in nn1.named_modules():
        f.write(f"Layer: {name}\n")
        f.write(f"  Type: {type(module)}\n")
        if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
            f.write(f"  Weight Shape: {tuple(module.weight.shape)}\n")
        if hasattr(module, 'bias') and isinstance(module.bias, torch.Tensor):
            f.write(f"  Bias Shape: {tuple(module.bias.shape)}\n")
        f.write("\n")
    f.write("Model 2:\n")
    f.write(str(nn2))  # Prints the architecture

    f.write("\n\nLayer-wise Details:\n")
    for name, module in nn1.named_modules():
        f.write(f"Layer: {name}\n")
        f.write(f"  Type: {type(module)}\n")
        if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
            f.write(f"  Weight Shape: {tuple(module.weight.shape)}\n")
        if hasattr(module, 'bias') and isinstance(module.bias, torch.Tensor):
            f.write(f"  Bias Shape: {tuple(module.bias.shape)}\n")
        f.write("\n")
Xtrain = scale(Xtrain,featuresForTraining,  scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)
Xval  = scale(Xval, featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)
Xtest  = scale(Xtest, featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)

Xtrain_tensor = torch.tensor(np.float32(Xtrain[featuresForTraining].values)).float()
Xval_tensor = torch.tensor(np.float32(Xval[featuresForTraining].values)).float()
Xtest_tensor = torch.tensor(np.float32(Xtest[featuresForTraining].values)).float()

Ytrain_tensor = torch.tensor(Ytrain).unsqueeze(1).float()
Ytest_tensor = torch.tensor(Ytest).unsqueeze(1).float()
Yval_tensor = torch.tensor(Yval).unsqueeze(1).float()
# %%
with torch.no_grad():  # No need to track gradients for inference
    YPredTrain1 = nn1(Xtrain_tensor).numpy()
    YPredTrain2 = nn2(Xtrain_tensor).numpy()
    YPredVal1 = nn1(Xval_tensor).numpy()
    YPredVal2 = nn2(Xval_tensor).numpy()
    YPredTest1 = nn1(Xtest_tensor).numpy()
    YPredTest2 = nn2(Xtest_tensor).numpy()
Xtrain = unscale(Xtrain, featuresForTraining=featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl")
Xval = unscale(Xval, featuresForTraining=featuresForTraining,   scalerName =  outFolder + "/model/myScaler.pkl")
Xtest = unscale(Xtest, featuresForTraining=featuresForTraining,   scalerName =  outFolder + "/model/myScaler.pkl")
# %%

# %%
nRanks = 8
for rank in range(nRanks):
    train_loss_history_rank = np.load(outFolder + "/model/train_loss_history_rank%d.npy"%(rank))
    val_loss_history_rank = np.load(outFolder + "/model/val_loss_history_rank%d.npy"%(rank))
    train_classifier_loss_history_rank = np.load(outFolder + "/model/train_classifier_loss_history_rank%d.npy"%(rank))
    val_classifier_loss_history_rank = np.load(outFolder + "/model/val_classifier_loss_history_rank%d.npy"%(rank))
    train_dcor_loss_history_rank = np.load(outFolder + "/model/train_dcor_loss_history_rank%d.npy"%(rank))
    val_dcor_loss_history_rank = np.load(outFolder + "/model/val_dcor_loss_history_rank%d.npy"%(rank))
    if rank == 0:
        train_loss_history = train_loss_history_rank
        val_loss_history = val_loss_history_rank
        train_classifier_loss_history = train_classifier_loss_history_rank
        val_classifier_loss_history = val_classifier_loss_history_rank
        train_dcor_loss_history = train_dcor_loss_history_rank
        val_dcor_loss_history = val_dcor_loss_history_rank
    else:
        train_loss_history = train_loss_history + train_loss_history_rank
        val_loss_history = val_loss_history + val_loss_history_rank
        train_classifier_loss_history = train_classifier_loss_history + train_classifier_loss_history_rank
        val_classifier_loss_history = val_classifier_loss_history + val_classifier_loss_history_rank
        train_dcor_loss_history = train_dcor_loss_history + train_dcor_loss_history_rank
        val_dcor_loss_history = val_dcor_loss_history + val_dcor_loss_history_rank
train_loss_history=train_loss_history/nRanks
val_loss_history=val_loss_history/nRanks
train_classifier_loss_history=train_classifier_loss_history/nRanks
val_classifier_loss_history=val_classifier_loss_history/nRanks
train_dcor_loss_history=train_dcor_loss_history/nRanks
val_dcor_loss_history=val_dcor_loss_history/nRanks
plot_lossTorch(train_loss_history, val_loss_history, 
              train_classifier_loss_history, val_classifier_loss_history,
              train_dcor_loss_history, val_dcor_loss_history,
              outFolder)
# %%



# Plots start here

Xval['PNN1'] = YPredVal1
Xval['PNN2'] = YPredVal2
Xtest['PNN1'] = YPredTest1
Xtest['PNN2'] = YPredTest2
fig, ax = plt.subplots(1, 2, figsize=(15, 8))
bins=np.linspace(0, 1, 101)
cS = np.histogram(Xval[Yval==0].PNN1,bins=bins)[0]
cD = np.histogram(Xval[Yval==1].PNN1,bins=bins)[0]
cS=cS/np.sum(cS)
cD = cD/np.sum(cD)

ax[0].hist(bins[:-1], bins=bins, weights=cS, histtype='step', color='blue') 
ax[0].hist(bins[:-1], bins=bins, weights=cD, histtype='step', color='red') 
ax[0].set_xlabel("Classifer 1")

cS = np.histogram(Xval[Yval==0].PNN2,bins=bins)[0]
cD = np.histogram(Xval[Yval==1].PNN2,bins=bins)[0]
cS=cS/np.sum(cS)
cD = cD/np.sum(cD)

ax[1].hist(bins[:-1], bins=bins, weights=cS, histtype='step', color='blue') 
ax[1].hist(bins[:-1], bins=bins, weights=cD, histtype='step', color='red') 
ax[1].set_xlabel("Classifier 2")
fig.savefig(outFolder+"/performance/outputSpin0.png", bbox_inches='tight')
from helpers.doPlots import NNoutputs
NNoutputs(signal_predictions=YPredVal1[genMassVal==125], realData_predictions=YPredVal1[genMassVal==0],
          signalTrain_predictions=YPredTrain1[genMassTrain==125], realDataTrain_predictions=YPredTrain1[genMassTrain==0],
          outName=outFolder+"/performance/output1_125.png", log=False)
NNoutputs(signal_predictions=YPredVal2[genMassVal==125], realData_predictions=YPredVal2[genMassVal==0],
          signalTrain_predictions=YPredTrain2[genMassTrain==125], realDataTrain_predictions=YPredTrain2[genMassTrain==0],
          outName=outFolder+"/performance/output2_125.png", log=False)
NNoutputs(signal_predictions=YPredVal1[genMassVal==125], realData_predictions=YPredVal1[genMassVal==0],
          signalTrain_predictions=YPredTrain1[genMassTrain==125], realDataTrain_predictions=YPredTrain1[genMassTrain==0],
          outName=outFolder+"/performance/output1_125_log.png", log=True)
NNoutputs(signal_predictions=YPredVal2[genMassVal==125], realData_predictions=YPredVal2[genMassVal==0],
          signalTrain_predictions=YPredTrain2[genMassTrain==125], realDataTrain_predictions=YPredTrain2[genMassTrain==0],
          outName=outFolder+"/performance/output2_125_log.png", log=True)
# Output for 125 GeV
plt.close('all')
fig, ax = plt.subplots(1, 2, figsize=(15, 8))
bins=np.linspace(0, 1, 51)
cS = np.histogram(Xval[genMassVal==0].PNN1,bins=bins)[0]
cD = np.histogram(Xval[genMassVal==125].PNN1,bins=bins)[0]
cS=cS/np.sum(cS)
cD = cD/np.sum(cD)

ax[0].hist(bins[:-1], bins=bins, weights=cS, histtype='step', color='blue') 
ax[0].hist(bins[:-1], bins=bins, weights=cD, histtype='step', color='red') 
ax[0].set_xlabel("Classifer 1")

cS = np.histogram(Xval[genMassVal==0].PNN2,bins=bins)[0]
cD = np.histogram(Xval[genMassVal==125].PNN2,bins=bins)[0]
cS=cS/np.sum(cS)
cD = cD/np.sum(cD)

ax[1].hist(bins[:-1], bins=bins, weights=cS, histtype='step', color='blue') 
ax[1].hist(bins[:-1], bins=bins, weights=cD, histtype='step', color='red') 
ax[1].set_xlabel("Classifier 2")
fig.savefig(outFolder+"/performance/output125.png", bbox_inches='tight')
plt.close('all')
# %%
from helpers.doPlots import auc_vs_m
auc_vs_m(Ytrain, Yval, YPredTrain1, YPredVal1, genMassTrain, genMassVal, outFile=outFolder+"/performance/auc1_vs_m.png")
auc_vs_m(Ytrain, Yval, YPredTrain2, YPredVal2, genMassTrain, genMassVal, outFile=outFolder+"/performance/auc2_vs_m.png")
plt.close('all')
# %%

from sklearn.metrics import roc_curve, auc
# ROC 1
fig, ax = plt.subplots(1, 1)

fpr_train, tpr_train, _ = roc_curve(Ytrain[(genMassTrain==125) | (genMassTrain==0)], YPredTrain1[(genMassTrain==125) | (genMassTrain==0)])
roc_auc_train = auc(fpr_train, tpr_train)
fpr_test, tpr_test, _ = roc_curve(Yval[(genMassVal==125) | (genMassVal==0)], YPredVal1[(genMassVal==125) | (genMassVal==0)])
roc1_auc_test = auc(fpr_test, tpr_test)
ax.plot(fpr_train, tpr_train, color='blue', lw=2, label=f'Train ROC curve (AUC = {roc_auc_train:.2f})')
ax.plot(fpr_test, tpr_test, color='red', lw=2, label=f'Validation ROC curve (AUC = {roc1_auc_test:.2f})')
ax.plot([0, 1], [0, 1], color='green', linestyle='--', lw=1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
ax.grid(alpha=0.3)
fig.savefig(outFolder+"/performance/ROC1_125.png")
plt.close('all')

# ROC 2
fig, ax = plt.subplots(1, 1)

fpr_train, tpr_train, _ = roc_curve(Ytrain[(genMassTrain==125) | (genMassTrain==0)], YPredTrain2[(genMassTrain==125) | (genMassTrain==0)])
roc_auc_train = auc(fpr_train, tpr_train)
fpr_test, tpr_test, _ = roc_curve(Yval[(genMassVal==125) | (genMassVal==0)], YPredVal2[(genMassVal==125) | (genMassVal==0)])
roc2_auc_test = auc(fpr_test, tpr_test)
results["roc_125"] = (roc1_auc_test + roc2_auc_test)/2
ax.plot(fpr_train, tpr_train, color='blue', lw=2, label=f'Train ROC curve (AUC = {roc_auc_train:.2f})')
ax.plot(fpr_test, tpr_test, color='red', lw=2, label=f'Validation ROC curve (AUC = {roc2_auc_test:.2f})')
ax.plot([0, 1], [0, 1], color='green', linestyle='--', lw=1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
ax.grid(alpha=0.3)
fig.savefig(outFolder+"/performance/ROC2_125.png")
plt.close('all')


# Efficiency in 2D plane
x_bins = np.linspace(0, 1, 11)
y_bins = np.linspace(0, 1, 11)


sig_mask = genMassVal == 125
bkg_mask = genMassVal == 0

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# Iterate through bins
sig_gain_start = 0
xmax, ymax=0, 0
efficiencySig_max, efficiencyBkg_max =0, 0
for i in range(len(x_bins) - 1):
    for j in range(len(y_bins) - 1):
        x_min  = x_bins[i]
        y_min  = y_bins[j]
        
        # Apply cuts
        sig_cut = (YPredVal2[sig_mask] > x_min)  & (YPredVal1[sig_mask] > y_min) 
        bkg_cut = (YPredVal2[bkg_mask] > x_min)  & (YPredVal1[bkg_mask] > y_min) 
        
        sig_count = np.sum(sig_cut)
        bkg_count = np.sum(bkg_cut)
        
        efficiencySig = (sig_count / np.sum(sig_mask)) * 100 if np.sum(sig_mask) > 0 else 0
        efficiencyBkg = (bkg_count / np.sum(bkg_mask)) * 100 if np.sum(bkg_mask) > 0 else 0
        
        # Place text in the middle of the cell
        x_text = (x_min + x_bins[i+1]) / 2
        y_text = (y_min + y_bins[j+1]) / 2
        if efficiencySig/np.sqrt(efficiencyBkg)>sig_gain_start:
            sig_gain_start=efficiencySig/np.sqrt(efficiencyBkg)
            xmax=x_text
            ymax=y_text
            efficiencySig_max=efficiencySig
            efficiencyBkg_max=efficiencyBkg
        ax.text(x_text, y_text, 'Sig: %.1f%%\nBkg: %.1f%%'%(efficiencySig, efficiencyBkg),
                ha='center', va='center', fontsize=8)

ax.fill_between(x=[xmax-.05, xmax+0.05], y1=[ymax-0.05, ymax-0.05],y2=[ymax+0.05, ymax+0.05], color='green', alpha=0.4 )
ax.text(xmax, ymax, 'Sig: %.1f%%\nBkg: %.1f%%'%(efficiencySig_max, efficiencyBkg_max),
                ha='center', va='center', fontsize=8)
ax.set_xlabel('Score 2')
ax.set_ylabel('Score 1')
ax.set_xticks(x_bins)
ax.set_yticks(y_bins)
ax.grid(True, linestyle='--', linewidth=0.5)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

fig.savefig(outFolder+"/performance/gridEfficiency.png", bbox_inches='tight')
print("Saved ", outFolder+"/performance/gridEfficiency.png")


# %%
# Create a 2D histogram
import matplotlib.colors as mcolors
fig, ax = plt.subplots(1, 1)
bins = np.linspace(0, 1, 50)
counts, xedges, yedges = np.histogram2d(np.array(YPredVal1[genMassVal==125]).reshape(-1), 
                                        np.array(YPredVal2[genMassVal==125]).reshape(-1),
                                        bins=(bins, bins))
counts = counts/np.sum(counts)
mesh = ax.pcolormesh(
    xedges, yedges, counts.T, cmap='viridis', norm=mcolors.LogNorm()  # Use log scale
)

cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label('Probability')
ax.set_xlabel('PNN 1 score')
ax.set_ylabel('PNN 2 score')

ax.grid(alpha=0.3)
fig.savefig(outFolder+"/performance/Higgs_nn1_vs_nn2.png")
plt.close('all')
# %%

fig, ax = plt.subplots(1, 1)
bins = np.linspace(0, 1, 50)
counts, xedges, yedges = np.histogram2d(np.array(YPredVal1[genMassVal==0]).reshape(-1), 
                                        np.array(YPredVal2[genMassVal==0]).reshape(-1),
                                        bins=(bins, bins))
counts = counts/np.sum(counts)
mesh = ax.pcolormesh(
    xedges, yedges, counts.T, cmap='viridis', norm=mcolors.LogNorm()  # Use log scale
)
cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label('Probability')
ax.set_xlabel('PNN 1 score')
ax.set_ylabel('PNN 2 score')

ax.grid(alpha=0.3)
fig.savefig(outFolder+"/performance/Data_nn1_vs_nn2.png")
plt.close('all')

# %%

ggHscoreScan(Xtest=Xval, Ytest=Yval, YPredTest=YPredVal1, genMassTest=genMassVal, Wtest=Wval, outName=outFolder + "/performance/ggHScoreScan_NN1.png")
ggHscoreScan(Xtest=Xval, Ytest=Yval, YPredTest=YPredVal2, genMassTest=genMassVal, Wtest=Wval, outName=outFolder + "/performance/ggHScoreScan_NN2.png")
plt.close('all')
# %%
nn2_t =0.4
Xval['PNN1'] = YPredVal1
Xval['PNN2'] = YPredVal2
mass_bins = np.load(outFolder+"/mass_bins.npy")
ks_p_value_PNN, p_value_PNN, chi2_values_PNN = checkOrthogonalityInMassBins(
    df=Xval[Yval==0],
    featureToPlot='PNN1',
    mask1=np.array(Xval[Yval==0]['PNN2'] >= nn2_t),
    mask2=np.array(Xval[Yval==0]['PNN2'] < nn2_t),
    label_mask1  = 'PNN2 High',
    label_mask2  = 'PNN2 Low',
    label_toPlot = 'PNN1',
    bins=np.linspace(0., 1, 3),
    mass_bins=mass_bins,
    mass_feature='dijet_mass',
    figsize=(20, 25),
    outName=outFolder+"/performance/PNN1_orthogonalityBinned.png" 
)
#plotLocalPvalues(pvalues=ks_p_value_PNN, mass_bins=mass_bins, pvalueLabel="KS", outFolder=outFolder+"/performance/PNN1_KSpvalues.png")
#plotLocalPvalues(pvalues=p_value_PNN, mass_bins=mass_bins, pvalueLabel="$\chi^2$", outFolder=outFolder+"/performance/PNN1_Chi2pvalues.png")
plt.close('all')
# %%
nn1_t = 0.4
ks_p_value_PNN, p_value_PNN, chi2_values_PNN = checkOrthogonalityInMassBins(
    df=Xval[Yval==0],
    featureToPlot='PNN2',
    mask1=np.array(Xval[Yval==0]['PNN1'] >= nn1_t),
    mask2=np.array(Xval[Yval==0]['PNN1'] < nn1_t),
    label_mask1  = 'PNN1 High',
    label_mask2  = 'PNN1 Low',
    label_toPlot = 'PNN2',
    bins=np.linspace(0., 1, 3),
    mass_bins=mass_bins,
    mass_feature='dijet_mass',
    figsize=(20, 25),
    outName=outFolder+"/performance/PNN2_orthogonalityBinned.png" 
)

plotLocalPvalues(pvalues=ks_p_value_PNN, mass_bins=mass_bins, pvalueLabel="KS", outFolder=outFolder+"/performance/PNN2_KSpvalues.png")
plotLocalPvalues(pvalues=p_value_PNN, mass_bins=mass_bins, pvalueLabel="$\chi^2$", outFolder=outFolder+"/performance/PNN2_Chi2pvalues.png")
plt.close('all')












# %%
print("*"*30)
print("Mutual Information binned")
print("*"*30, "\n\n")
mutualInfos = []
mass_feature = 'dijet_mass'
for i, (low, high) in enumerate(zip(mass_bins[:-1], mass_bins[1:])):
        mask_mass = (Xval[mass_feature] >= low) & (Xval[mass_feature] < high) & (Yval==0)
        mi = mutual_info_regression(np.array(Xval[mask_mass].PNN2).reshape(-1, 1), np.array(Xval[mask_mass].PNN1))[0]
        mutualInfos.append(mi)
        print("%.1f < mjj < %.1f : %.5f"%(low, high, mi))
plotLocalPvalues(mutualInfos, mass_bins, pvalueLabel='Mutual Info', type='', outFolder=outFolder+'/performance/mutualInfo.png')
from scipy.stats import  chi2
#Chi2 test
mass_feature = 'dijet_mass'
print("Chi2")
chi2_tot = 0
for i, (low, high) in enumerate(zip(mass_bins[:-1], mass_bins[1:])):
        df = Xval[(Xval[mass_feature] >= low) & (Xval[mass_feature] < high) & (Yval==0)]
        regionA      = np.sum((df['PNN2']<nn2_t ) & (df['PNN1']>nn1_t) )
        regionB      = np.sum((df['PNN2']>nn2_t ) & (df['PNN1']>nn1_t) )
        regionC      = np.sum((df['PNN2']<nn2_t ) & (df['PNN1']<nn1_t) )
        regionD      = np.sum((df['PNN2']>nn2_t ) & (df['PNN1']<nn1_t) )
        
        expected = regionA*regionD/regionC
        err = regionA*regionD/regionC * np.sqrt(1/regionA + 1/regionC + 1/regionD)
        chi2_bin = (regionB - expected)**2/(err**2 + regionB)
        chi2_tot = chi2_tot+chi2_bin
        print("%.1f < mjj < %.1f : %.5f"%(low, high, chi2_bin))
ndof = len(mass_bins)-1
print("ndof = ", ndof)
chi2_pvalue = 1 - chi2.cdf(chi2_tot, ndof)
print("chi2_tot", chi2_tot)
results['chi2_SR'] = chi2_tot
results['chi2pvalue_SR'] = chi2_pvalue
plt.close('all')


# %%
# Compute disCo
print("*"*30)
print("Disco binned")
print("*"*30, "\n\n")
import dcor
discos_bin_train = []
discos_bin_test = []
discos_bin_val = []
mass_bins = np.load(outFolder+"/mass_bins.npy")
for blow, bhigh in zip(mass_bins[:-1], mass_bins[1:]):
    mask = (Xval.dijet_mass >=blow) & (Xval.dijet_mass < bhigh)  & (Yval==0)
    distance_corr = dcor.distance_correlation(np.array(YPredVal1[mask], dtype=np.float64), np.array(YPredVal2[mask], dtype=np.float64))
    print("%.1f < mjj < %.1f  : %.5f"%(blow, bhigh, distance_corr))
    discos_bin_val.append(distance_corr)
    
    mask = (Xtrain.dijet_mass >=blow) & (Xtrain.dijet_mass < bhigh)  & (Ytrain==0)
    distance_corr = dcor.distance_correlation(np.array(YPredTrain1[mask], dtype=np.float64), np.array(YPredTrain2[mask], dtype=np.float64))
    discos_bin_train.append(distance_corr)

    mask = (Xtest.dijet_mass >=blow) & (Xtest.dijet_mass < bhigh)  & (Ytest==0)
    distance_corr = dcor.distance_correlation(np.array(YPredTest1[mask], dtype=np.float64), np.array(YPredTest2[mask], dtype=np.float64))
    discos_bin_test.append(distance_corr)
results['averageBin_sqaured_disco'] = np.mean(discos_bin_val)
results['error_averageBin_sqaured_disco'] = np.std(discos_bin_val)/np.sqrt(len(discos_bin_val))
plotLocalPvalues(discos_bin_val, mass_bins, pvalueLabel="Distance Corr.", type = '', outFolder=outFolder+"/performance/disco_mjjbins_val.png")
plotLocalPvalues(discos_bin_train, mass_bins, pvalueLabel="Distance Corr.", type = '', outFolder=outFolder+"/performance/disco_mjjbins_train.png", color='red')
plotLocalPvalues(discos_bin_test, mass_bins, pvalueLabel="Distance Corr.", type = '', outFolder=outFolder+"/performance/disco_mjjbins_test.png", color='green')
# %%
print("*"*30)
print("Pull ABCD Delta/sigmaDelta")
print("*"*30, "\n\n")
for i, (low, high) in enumerate(zip(mass_bins[:-1], mass_bins[1:])):

    regionA      = np.sum((Xval['PNN2']<nn2_t ) & (Xval['PNN1']>nn1_t) & (Yval==0) & (Xval['dijet_mass']>low) & (Xval['dijet_mass']<high))
    regionB      = np.sum((Xval['PNN2']>nn2_t ) & (Xval['PNN1']>nn1_t) & (Yval==0) & (Xval['dijet_mass']>low) & (Xval['dijet_mass']<high))
    regionC      = np.sum((Xval['PNN2']<nn2_t ) & (Xval['PNN1']<nn1_t) & (Yval==0) & (Xval['dijet_mass']>low) & (Xval['dijet_mass']<high))
    regionD      = np.sum((Xval['PNN2']>nn2_t ) & (Xval['PNN1']<nn1_t) & (Yval==0) & (Xval['dijet_mass']>low) & (Xval['dijet_mass']<high))

    pull = (regionB-regionA*regionD/regionC)/np.sqrt(regionB + (regionA*regionD/regionC)**2 * (1/regionA + 1/regionC + 1/regionD))
    print("Pull %.1f < mjj < %.1f : "%(low, high), pull)
    print("Observed/Expected ABCD", regionB, "/", regionA*regionD/regionC)




from hist import Hist
x1 = 'PNN2'
x2 = 'PNN1'
#mass_bins=np.array([40.,60.,70.,80.,90.,110.,130.,150.,200.,300.])
hA = Hist.new.Var(mass_bins, name="mjj").Weight()
hB = Hist.new.Var(mass_bins, name="mjj").Weight()
hC = Hist.new.Var(mass_bins, name="mjj").Weight()
hD = Hist.new.Var(mass_bins, name="mjj").Weight()
regions = {
    'A' : hA,
    'B' : hB,
    'C' : hC,
    'D' : hD,
}


# Fill regions with data
df = Xval[Yval==0]

xx = 'dijet_mass'
mA      = (df[x1]<nn2_t ) & (df[x2]>nn1_t ) 
mB      = (df[x1]>nn2_t ) & (df[x2]>nn1_t ) 
mC      = (df[x1]<nn2_t ) & (df[x2]<nn1_t ) 
mD      = (df[x1]>nn2_t ) & (df[x2]<nn1_t ) 
regions['A'].fill(df[mA][xx])
regions['B'].fill(df[mB][xx])
regions['C'].fill(df[mC][xx])
regions['D'].fill(df[mD][xx])
# %%


sys.path.append("/t3home/gcelotto/ggHbb/abcd/new/helpersABCD")
from plot_v2 import plot4ABCD, QCD_SR

hB_ADC = plot4ABCD(regions, mass_bins, x1, x2, nn2_t, nn1_t, suffix='temp', blindPar=(False, 125, 20), outName=outFolder+"/performance/abcd_check4R", sameWidth_flag=False)
qcd_mc = regions['B']
print(qcd_mc.values())
print(hB_ADC.values())
QCD_SR(mass_bins, hB_ADC, qcd_mc, suffix="temp", blindPar=(False, 125, 10), outName=outFolder+"/performance/abcd_checkSR", sameWidth_flag=False)





#%%
hA = Hist.new.Var(mass_bins, name="mjj").Weight()
hB = Hist.new.Var(mass_bins, name="mjj").Weight()
hC = Hist.new.Var(mass_bins, name="mjj").Weight()
hD = Hist.new.Var(mass_bins, name="mjj").Weight()
regions = {
    'A' : hA,
    'B' : hB,
    'C' : hC,
    'D' : hD,
}


# Fill regions with data
df = Xtest[Ytest==0]

xx = 'dijet_mass'
mA      = (df[x1]<nn2_t ) & (df[x2]>nn1_t ) 
mB      = (df[x1]>nn2_t ) & (df[x2]>nn1_t ) 
mC      = (df[x1]<nn2_t ) & (df[x2]<nn1_t ) 
mD      = (df[x1]>nn2_t ) & (df[x2]<nn1_t ) 
regions['A'].fill(df[mA][xx])
regions['B'].fill(df[mB][xx])
regions['C'].fill(df[mC][xx])
regions['D'].fill(df[mD][xx])
hB_ADC = plot4ABCD(regions, mass_bins, x1, x2, nn2_t, nn1_t, suffix='temp', blindPar=(False, 125, 20), outName=outFolder+"/performance/abcd_check4R_test", sameWidth_flag=False)
qcd_mc = regions['B']
print(qcd_mc.values())
print(hB_ADC.values())
QCD_SR(mass_bins, hB_ADC, qcd_mc, suffix="temp", blindPar=(False, 125, 10), outName=outFolder+"/performance/abcd_checkSR_test", sameWidth_flag=False, corrected=True)



# %%
import json
with open(outFolder+"/model/dict.json", "w") as file:
    json.dump(results, file, indent=4)


# %%
print("Shap started")
XvalScaled  = scale(Xval, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False, featuresForTraining=featuresForTraining)
nEvents = 500
X_tensor_test = torch.tensor(np.float32(XvalScaled[featuresForTraining].values[:nEvents])).float()
nn1_featureImportance = getShapTorch(Xtest=Xtest[featuresForTraining].head(nEvents), model=nn1, outName=outFolder+'/performance/shap1.png', nFeatures=-1, class_names=['NN1 output'], tensor=X_tensor_test)
nn2_featureImportance = getShapTorch(Xtest=Xtest[featuresForTraining].head(nEvents), model=nn2, outName=outFolder+'/performance/shap2.png', nFeatures=-1, class_names=['NN2 output'], tensor=X_tensor_test)


# %%
nns_featureImportance = {}
for f in nn1_featureImportance.keys():
    nns_featureImportance[f] = nn1_featureImportance[f][0] + nn2_featureImportance[f][0]

# %%
sorted_features = sorted(nns_featureImportance.items(), key=lambda x: x[1], reverse=True)
feature_names, feature_importance = zip(*sorted_features)

# Creating the bar plot
plt.figure(figsize=(10, 6))
plt.bar(feature_names, feature_importance, color='skyblue')

# Adding labels and title
plt.xlabel('Feature Name', fontsize=12)
plt.ylabel('Feature Importance', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()

# Display the plot
plt.savefig(outFolder+"/performance/shapNNs.png", bbox_inches='tight')
# %%
