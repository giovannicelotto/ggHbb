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
from helpers.doPlots import runPlotsTorch, doPlotLoss_Torch
from helpers.loadSaved import loadXYWrWSaved
import torch
from helpers.scaleUnscale import scale, unscale
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
    parser.add_argument("-l", "--lambda_dcor", type=float, help="lambda for penalty term", default=None)
    parser.add_argument("-dt", "--date", type=str, help="MonthDay format e.g. Dec17", default=None)
    parser.add_argument("-b", "--boosted", type=int, help="boosted class", default=1)
    args = parser.parse_args()
    if args.lambda_dcor is not None:
        hp["lambda_dcor"] = args.lambda_dcor 
    if args.date is not None:
        current_date = args.date
    if args.boosted is not None:
        boosted = args.boosted
except:
    hp["lambda_dcor"] = 0.
    current_date="Mar29"
    boosted=60
    print("Interactive mode")
# %%
sampling=True
results = {}
inFolder_, outFolder = getInfolderOutfolder(name = "%s_%d_%s"%(current_date, boosted, str(hp["lambda_dcor"]).replace('.', 'p')), suffixResults='_mjjDisco', createFolder=False)
inFolder = "/t3home/gcelotto/ggHbb/PNN/input/data_sampling_pt%d_1A"%boosted if sampling else "/t3home/gcelotto/ggHbb/PNN/input/data_highPt"
modelName = "model.pth"
featuresForTraining = list(np.load(outFolder+"/featuresForTraining.npy"))
#featuresForTraining +=['dijet_mass']
#featuresForTraining.remove('jet2_btagDeepFlavB')
# %%
Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, Wtrain, Wval,Wtest, rWtrain, rWval, genMassTrain, genMassVal, genMassTest = loadXYWrWSaved(inFolder=inFolder)


# Take the test dataset from the other folder which is unbiased wrt to sampling
if sampling:
    Xtest = pd.read_parquet(inFolder+"/Xtest.parquet")
    Ytest = np.load(inFolder+"/Ytest.npy")
    Wtest = np.load(inFolder+"/Wtest.npy")
    genMassTest = np.load(inFolder+"/genMassTest.npy")

rawFiles = pd.read_parquet(glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/training/*.parquet")[:10])
rawFiles = rawFiles[(rawFiles.jet1_btagTight>0.5) & (rawFiles.jet2_btagTight>0.5)]


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
Xtest  = scale(Xtest, featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)
rawFiles  = scale(rawFiles, featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl" ,fit=False)

Xtrain_tensor = torch.tensor(np.float32(Xtrain[featuresForTraining].values)).float()
Xval_tensor = torch.tensor(np.float32(Xval[featuresForTraining].values)).float()
Xtest_tensor = torch.tensor(np.float32(Xtest[featuresForTraining].values)).float()
rawFiles_tensor = torch.tensor(np.float32(rawFiles[featuresForTraining].values)).float()
# %%
with torch.no_grad():  # No need to track gradients for inference
    YPredTrain = model(Xtrain_tensor).numpy()
    YPredVal = model(Xval_tensor).numpy()
    YPredTest = model(Xtest_tensor).numpy()
    YPredRawFiles = model(rawFiles_tensor).numpy()
Xtrain = unscale(Xtrain, featuresForTraining=featuresForTraining, scalerName= outFolder + "/model/myScaler.pkl")
Xval = unscale(Xval, featuresForTraining=featuresForTraining,   scalerName =  outFolder + "/model/myScaler.pkl")
Xtest = unscale(Xtest, featuresForTraining=featuresForTraining,   scalerName =  outFolder + "/model/myScaler.pkl")
rawFiles = unscale(rawFiles, featuresForTraining=featuresForTraining,   scalerName =  outFolder + "/model/myScaler.pkl")

# %%



print("*"*30)
print("Run plots")
print("*"*30, "\n\n")
# %%

fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
fig.align_ylabels([ax[0],ax[1]])

bins = np.linspace(50, 300, 51)
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
ggHscoreScan(Xtest=Xval,    Ytest=Yval,     YPredTest=YPredVal,     Wtest=Wval,     genMassTest=genMassVal,     outName=outFolder + "/performance/ggHScoreScanMulti.png", t=[0, 0.2, 0.4, 0.6, 0.8,1 ])
ggHscoreScan(Xtest=Xval,    Ytest=Yval,     YPredTest=YPredVal,     Wtest=Wval,     genMassTest=genMassVal,     outName=outFolder + "/performance/ggHScoreScan_2.png", t=[0, 0.5,1 ])
ggHscoreScan(Xtest=Xtest,   Ytest=Ytest,    YPredTest=YPredTest,    Wtest=Wtest,    genMassTest=genMassTest,    outName=outFolder + "/performance/ggHScoreScanTest_2.png", t=[0, 0.5,1 ])
ggHscoreScan(Xtest=rawFiles,   Ytest=np.zeros(len(rawFiles)),    YPredTest=YPredRawFiles,    Wtest=np.ones(len(rawFiles)),    genMassTest=np.zeros(len(rawFiles)),    outName=outFolder + "/performance/ggHScoreScan_rawFiles.png", t=[0, 0.6,1 ])




# Compute disCo
#print("*"*30)
#print("Disco binned")
#print("*"*30, "\n\n")
#bins = np.linspace(40, 300, 21)
#discos_bin_train = []
#discos_bin_test = []
#discos_bin_val = []
#for blow, bhigh in zip(bins[:-1], bins[1:]):
#    mask = (Xval.dijet_mass >=blow) & (Xval.dijet_mass < bhigh)  & (Yval==0)
#    distance_corr = dcor.distance_correlation(np.array(YPredVal[mask], dtype=np.float64), np.array(Xval.dijet_mass[mask], dtype=np.float64))
#    print("%.1f < mjj < %.1f  : %.5f"%(blow, bhigh, distance_corr))
#    discos_bin_val.append(distance_corr)
#    
#    mask = (Xtrain.dijet_mass >=blow) & (Xtrain.dijet_mass < bhigh)  & (Ytrain==0)
#    distance_corr = dcor.distance_correlation(np.array(YPredTrain[mask], dtype=np.float64), np.array(Xtrain.dijet_mass[mask], dtype=np.float64))
#    discos_bin_train.append(distance_corr)
#
#    mask = (Xtest.dijet_mass >=blow) & (Xtest.dijet_mass < bhigh)  & (Ytest==0)
#    distance_corr = dcor.distance_correlation(np.array(YPredTest[mask], dtype=np.float64), np.array(Xtest.dijet_mass[mask], dtype=np.float64))
#    discos_bin_test.append(distance_corr)
#results['averageBin_sqaured_disco'] = np.mean(discos_bin_val)
#results['error_averageBin_sqaured_disco'] = np.std(discos_bin_val)/np.sqrt(len(discos_bin_val))
#plotLocalPvalues(discos_bin_val, bins, pvalueLabel="Distance Corr.", type = '', outFolder=outFolder+"/performance/disco_mjjbins_val.png")
#plotLocalPvalues(discos_bin_train, bins, pvalueLabel="Distance Corr.", type = '', outFolder=outFolder+"/performance/disco_mjjbins_train.png", color='red')
#plotLocalPvalues(discos_bin_test, bins, pvalueLabel="Distance Corr.", type = '', outFolder=outFolder+"/performance/disco_mjjbins_test.png", color='green')


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

    'sigTest':[],
    'bkgTest':[],
    'significanceTest':[],
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

    sigEff = np.sum(YPredTest[(genMassTest==125) & (Xtest.dijet_mass > 100) & (Xtest.dijet_mass < 150)] > t)/len(YPredTest[(genMassTest==125) & (Xtest.dijet_mass > 100) & (Xtest.dijet_mass < 150)])
    bkgEff = np.sum(YPredTest[(genMassTest==0) & (Xtest.dijet_mass > 100) & (Xtest.dijet_mass < 150)] > t)/len(YPredTest[(genMassTest==0) & (Xtest.dijet_mass > 100) & (Xtest.dijet_mass < 150)])
    efficiencies["sigTest"].append(sigEff)
    efficiencies["bkgTest"].append(bkgEff)
    significanceTest = sigEff/np.sqrt(bkgEff) if bkgEff!=0 else 0
    efficiencies["significanceTest"].append(significanceTest)

fig, ax = plt.subplots(1, 1)
ax.plot(ts, efficiencies["sigTrain"], color='red', label="Sig Train", linestyle='dashed')
ax.plot(ts, efficiencies["bkgTrain"], color='blue', label="Bkg Train", linestyle='dashed')
ax.plot(ts, efficiencies["significanceTrain"], color='green', label="Significance Train", linestyle='dashed')

ax.plot(ts, efficiencies["bkgVal"], color='blue', label="Bkg Val")
ax.plot(ts, efficiencies["sigVal"], color='red', label="Sig Val")
ax.plot(ts, efficiencies["significanceVal"], color='green', label="Significance Val")

ax.plot(ts, efficiencies["bkgTest"], color='blue', label="Bkg Test", linestyle='-.')
ax.plot(ts, efficiencies["sigTest"], color='red', label="Sig Test", linestyle='-.')
ax.plot(ts, efficiencies["significanceTest"], color='green', label="Significance Test", linestyle='-.')
ax.legend()
fig.savefig(outFolder+"/performance/effScan.png", bbox_inches='tight')
print("Saved ", outFolder+"/performance/effScan.png")


#sys.exit()
fig, ax =plt.subplots(1, 1)
ax.hist(Xval.dijet_mass[Yval==0], bins=np.linspace(30, 310, 100))
fig.savefig(outFolder+"/performance/dijetMassVal")
plt.close('all')
# %%
print("Efficiency at WP")
if boosted == 2:
    sigEff = np.sum(YPredTest[(genMassTest==125) ] > 0.45)/len(YPredTest[(genMassTest==125) ])
    bkgEff = np.sum(YPredTest[(genMassTest==0) ] > 0.45)/len(YPredTest[(genMassTest==0) ])
    print(sigEff, bkgEff)

    sigEff = np.sum(YPredTest[(genMassTest==125) ] > 0.7)/len(YPredTest[(genMassTest==125) ])
    bkgEff = np.sum(YPredTest[(genMassTest==0) ] > 0.7)/len(YPredTest[(genMassTest==0) ])
    print(sigEff, bkgEff)
elif boosted == 1:
    sigEff = np.sum(YPredTest[(genMassTest==125) ] > 0.4)/len(YPredTest[(genMassTest==125) ])
    bkgEff = np.sum(YPredTest[(genMassTest==0) ] > 0.4)/len(YPredTest[(genMassTest==0) ])
    print(sigEff, bkgEff)






# %%

Xtest.columns = [str(Xtest.columns[_]) for _ in range((Xtest.shape[1]))]
Xval.columns = [str(Xval.columns[_]) for _ in range((Xval.shape[1]))]
results = runPlotsTorch(Xtrain, Xval, Ytrain, Yval, np.ones(len(Xtrain)), np.ones(len(Xval)), YPredTrain, YPredVal, featuresForTraining, model, inFolder, outFolder, genMassTrain, genMassVal, results)
# %%
train_loss_history = np.load(outFolder + "/model/train_loss_history.npy")
val_loss_history = np.load(outFolder + "/model/val_loss_history.npy")
print("*"*30)
print("Loss function")
print("*"*30, "\n\n")

doPlotLoss_Torch(train_loss_history, val_loss_history, outName=outFolder+"/performance/loss.png", earlyStop=np.argmin(train_loss_history))
#plot_lossTorch(train_loss_history, , 
#              train_classifier_loss_history, val_classifier_loss_history,
#              train_dcor_loss_history, val_dcor_loss_history,
#              outFolder)

# %%
from sklearn.metrics import roc_curve, auc
# ROC 1
fig, ax = plt.subplots(1, 1)

fpr_train, tpr_train, _ = roc_curve(Ytrain[(genMassTrain==125) | (genMassTrain==0)], YPredTrain[(genMassTrain==125) | (genMassTrain==0)])
roc_auc_train = auc(fpr_train, tpr_train)
fpr_test, tpr_test, _ = roc_curve(Yval[(genMassVal==125) | (genMassVal==0)], YPredVal[(genMassVal==125) | (genMassVal==0)])
roc1_auc_test = auc(fpr_test, tpr_test)
ax.plot(fpr_train, tpr_train, color='blue', lw=2, label=f'Train ROC curve (AUC = {roc_auc_train:.2f})')
ax.plot(fpr_test, tpr_test, color='red', lw=2, label=f'Validation ROC curve (AUC = {roc1_auc_test:.2f})')
ax.plot([0, 1], [0, 1], color='green', linestyle='--', lw=1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
ax.grid(alpha=0.3)
fig.savefig(outFolder+"/performance/ROC_125.png")
plt.close('all')

# %%
