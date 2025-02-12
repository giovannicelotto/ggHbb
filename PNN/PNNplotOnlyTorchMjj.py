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

# Get current month and day
current_date = datetime.now().strftime("%b%d")  # This gives the format like 'Dec12'
# %%
hp = getParams()
parser = argparse.ArgumentParser(description="Script.")
# Define arguments
try:
    parser.add_argument("-l", "--lambda_dcor", type=float, help="lambda for penalty term", default=None)
    parser.add_argument("-dt", "--date", type=str, help="MonthDay format e.g. Dec17", default=None)
    args = parser.parse_args()
    if args.lambda_dcor is not None:
        hp["lambda_dcor"] = args.lambda_dcor 
    if args.date is not None:
        current_date = args.date
except:
    hp["lambda_dcor"] = 2000
    print("Interactive mode")
# %%
sampling=True
results = {}
inFolder, outFolder = getInfolderOutfolder(name = "%s_%s"%(current_date, str(hp["lambda_dcor"]).replace('.', 'p')), suffixResults='_mjjDisco', createFolder=False)
inFolder = "/t3home/gcelotto/ggHbb/PNN/input/data_sampling_highPt" if sampling else "/t3home/gcelotto/ggHbb/PNN/input/data_highPt"
modelName = "model.pth"
featuresForTraining, columnsToRead = getFeatures(outFolder,  massHypo=True)

# %%
Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, Wtrain, Wval,Wtest, rWtrain, rWval, genMassTrain, genMassVal, genMassTest = loadXYWrWSaved(inFolder=inFolder)




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

Xtrain_tensor = torch.tensor(np.float32(Xtrain[featuresForTraining].values)).float()
Ytrain_tensor = torch.tensor(Ytrain).unsqueeze(1).float()

Xval_tensor = torch.tensor(np.float32(Xval[featuresForTraining].values)).float()
Yval_tensor = torch.tensor(Yval).unsqueeze(1).float()
# %%
with torch.no_grad():  # No need to track gradients for inference
    YPredTrain = model(Xtrain_tensor).numpy()
    YPredVal = model(Xval_tensor).numpy()
Xtrain = unscale(Xtrain, featuresForTraining=featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl")
Xval = unscale(Xval, featuresForTraining=featuresForTraining,   scalerName =  outFolder + "/model/myScaler.pkl")

# %%



print("*"*30)
print("Run plots")
print("*"*30, "\n\n")
# %%

fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
fig.align_ylabels([ax[0],ax[1]])

bins = np.linspace(40, 300, 51)
t = [0, 0.5, 1]


Yval=Yval.reshape(-1)
YPredVal=YPredVal.reshape(-1)

combinedMask = (Yval==0) & (YPredVal<0.5)
lowCounts = np.histogram(Xval.dijet_mass[combinedMask], bins=bins)[0]
err_lowCounts = np.sqrt(lowCounts)
err_lowCounts_n = np.sqrt(lowCounts)/np.sum(lowCounts)
lowCounts_n = lowCounts/np.sum(lowCounts)
ax[0].hist(bins[:-1], bins=bins, weights=lowCounts_n, label='NN < 0.5', density=False, color='blue')[0]

x = (bins[1:] + bins[:-1])/2

combinedMask = (Yval==0) & (YPredVal>0.5) 
highCounts = np.histogram(Xval.dijet_mass[combinedMask], bins=bins)[0]
highCounts_n = highCounts/np.sum(highCounts)
err_highCounts = np.sqrt(highCounts)
err_highCounts_n = np.sqrt(highCounts)/np.sum(highCounts)
ax[0].hist(bins[:-1], bins=bins, weights=highCounts_n, label='NN > 0.5', histtype=u'step', color='red')
ax[1].errorbar(x, highCounts_n/lowCounts_n, yerr = highCounts_n/lowCounts_n * np.sqrt(1/highCounts  +  1/lowCounts), color='red', linestyle='none', marker='o')
ax[1].set_ylim(0.9, 1.1)
ax[1].set_ylabel("Ratio")
ax[0].legend()
ax[0].set_xlim(bins[0], bins[-1])
ax[1].hlines(xmin=bins[0], xmax=bins[-1], y=1, color='black')



ndof = len(highCounts_n)
sigma = np.sqrt(err_highCounts_n**2 + err_lowCounts_n**2)
chi2_stat = np.sum(((highCounts_n - lowCounts_n)/sigma)**2)
from scipy.stats import  chi2
chi2_pvalue = 1- chi2.cdf(chi2_stat, ndof)
ax[0].text(x=0.1, y=0.12, s="$\chi^2$/ndof = %.1f/%d, p-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax[0].transAxes)

fig.savefig(outFolder + "/performance/ggHScan_HighLow.png", bbox_inches='tight')


# %%




from helpers.doPlots import ggHscoreScan
ggHscoreScan(Xtest=Xval, Ytest=Yval, YPredTest=YPredVal, Wtest=Wval, genMassTest=genMassVal, outName=outFolder + "/performance/ggHScoreScanMulti.png", t=[0, 0.2, 0.4, 0.6, 0.8,1 ])
ggHscoreScan(Xtest=Xval, Ytest=Yval, YPredTest=YPredVal, Wtest=Wval, genMassTest=genMassVal, outName=outFolder + "/performance/ggHScoreScan_2.png", t=[0, 0.5,1 ])
results = runPlotsTorch(Xtrain, Xval, Ytrain, Yval, np.ones(len(Xtrain)), np.ones(len(Xval)), YPredTrain, YPredVal, featuresForTraining, model, inFolder, outFolder, genMassTrain, genMassVal, results)
# %%
train_loss_history = np.load(outFolder + "/model/train_loss_history.npy")
val_loss_history = np.load(outFolder + "/model/val_loss_history.npy")
train_classifier_loss_history = np.load(outFolder + "/model/train_classifier_loss_history.npy")
val_classifier_loss_history = np.load(outFolder + "/model/val_classifier_loss_history.npy")
train_dcor_loss_history = np.load(outFolder + "/model/train_dcor_loss_history.npy")
val_dcor_loss_history = np.load(outFolder + "/model/val_dcor_loss_history.npy")
print("*"*30)
print("Loss function")
print("*"*30, "\n\n")

plot_lossTorch(train_loss_history, val_loss_history, 
              train_classifier_loss_history, val_classifier_loss_history,
              train_dcor_loss_history, val_dcor_loss_history,
              outFolder)
# %%

mass_bins = np.load(outFolder+"/mass_bins.npy")
# %%

print("*"*30)
print("Mutual Information binned")
print("*"*30, "\n\n")
mass_feature = 'dijet_mass'
Xval['PNN'] = YPredVal
mi = mutual_info_regression(np.array(Xval[Yval==0].dijet_mass).reshape(-1, 1), np.array(Xval[Yval==0].PNN))[0]
print("Mutual Information : %.5f"%(mi))


# Compute disCo
import dcor
arr1 = np.array(YPredVal[Yval==0], dtype=np.float64)
arr2 = np.array(Xval[Yval==0].dijet_mass, dtype=np.float64)
distance_corr = dcor.distance_correlation(arr1, arr2)**2
print("dcor**2 : ", distance_corr)






# %%
#import json
#with open(outFolder+"/model/dict.json", "w") as file:
#    json.dump(results, file, indent=4)
# %%
