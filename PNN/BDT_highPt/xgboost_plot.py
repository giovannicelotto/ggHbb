# %%
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import sys
sys.path.append("/t3home/gcelotto/ggHbb/PNN")
from helpers.getFeatures import getFeatures
from helpers.getParams import getParams
from helpers.getInfolderOutfolder import getInfolderOutfolder
from helpers.doPlots import runPlotsTorch, doPlotLoss_Torch
from helpers.loadSaved import loadXYWrWSaved
import torch
from helpers.scaleUnscale import scale, unscale
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from sklearn.feature_selection import mutual_info_regression
import xgboost as xgb
from sklearn.metrics import log_loss
# Get current month and day
current_date = datetime.now().strftime("%b%d")  # This gives the format like 'Dec12'
# %%
hp = getParams()
parser = argparse.ArgumentParser(description="Script.")
# Define arguments
try:
    parser.add_argument("-v", "--version", type=float, help="version of the model e.g. 1.0", default=None)
    parser.add_argument("-dt", "--date", type=str, help="MonthDay format e.g. Dec17", default=None)
    parser.add_argument("-b", "--boosted", type=int, help="boosted class", default=1)
    args = parser.parse_args()
    if args.version is not None:
        hp["version"] = args.version 
    if args.date is not None:
        current_date = args.date
    if args.boosted is not None:
        boosted = args.boosted
except:
    hp["version"] = 2.
    current_date="May28"
    boosted=1
    print("Interactive mode")
# %%
sampling=False
results = {}
inFolder_, outFolder = getInfolderOutfolder(name = "%s_%d_%s"%(current_date, boosted, str(hp["version"]).replace('.', 'p')), suffixResults='_BDT', createFolder=False)
inFolder = "/t3home/gcelotto/ggHbb/PNN/input/data_sampling_pt%d_1D"%(boosted) if sampling else "/t3home/gcelotto/ggHbb/PNN/input/data_pt%d_1D"%(boosted)
modelName = "xgboost_model.json"
featuresForTraining = list(np.load(outFolder+"/featuresForTraining.npy"))
#featuresForTraining +=['dijet_mass']
#featuresForTraining.remove('jet2_btagDeepFlavB')
# %%
Xtrain, Xval, Ytrain, Yval, Wtrain, Wval, rWtrain, rWval, genMassTrain, genMassVal = loadXYWrWSaved(inFolder=inFolder, isTest=False)
# %%



# %%

model_path = outFolder+"/model/%s"%modelName
bst = xgb.Booster()  
bst.load_model(model_path)  

# %%
#dtest = xgb.DMatrix(Xtest[featuresForTraining].values, label=Ytest)
dtrain = xgb.DMatrix(Xtrain[featuresForTraining].values, label=Ytrain)
dval = xgb.DMatrix(Xval[featuresForTraining].values, label=Yval)

# %%

# %%
# %%

#YPredTest = bst.predict(dtest)
YPredVal = bst.predict(dval)
YPredTrain = bst.predict(dtrain)
# Evaluate the model using Log Loss (binary classification)
test_log_loss = log_loss(Yval, YPredVal)
print(f"Test Log Loss: {test_log_loss:.4f}")

# %%

####
####
####            PLOTS START HERE
####

from sklearn.metrics import roc_curve, auc
maskHiggsData_train = (genMassTrain==0) | (genMassTrain==125)
maskHiggsData_val = (genMassVal==0) | (genMassVal==125)

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
ts = np.linspace(0, 1, 51)
efficiencies = {
    'sigTrain':[],
    'bkgTrain':[],
    'significanceTrain':[],
    'sigVal':[],
    'bkgVal':[],
    'significanceVal':[],

    #'sigTest':[],
    #'bkgTest':[],
    #'significanceTest':[],
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

    #sigEff = np.sum(YPredTest[(genMassTest==125) & (Xtest.dijet_mass > 100) & (Xtest.dijet_mass < 150)] > t)/len(YPredTest[(genMassTest==125) & (Xtest.dijet_mass > 100) & (Xtest.dijet_mass < 150)])
    #bkgEff = np.sum(YPredTest[(genMassTest==0) & (Xtest.dijet_mass > 100) & (Xtest.dijet_mass < 150)] > t)/len(YPredTest[(genMassTest==0) & (Xtest.dijet_mass > 100) & (Xtest.dijet_mass < 150)])
    #efficiencies["sigTest"].append(sigEff)
    #efficiencies["bkgTest"].append(bkgEff)
    #significanceTest = sigEff/np.sqrt(bkgEff) if bkgEff!=0 else 0
    #efficiencies["significanceTest"].append(significanceTest)

fig, ax = plt.subplots(1, 1)
ax.plot(ts, efficiencies["sigTrain"], color='red', label="Sig Train", linestyle='dashed')
ax.plot(ts, efficiencies["bkgTrain"], color='blue', label="Bkg Train", linestyle='dashed')
ax.plot(ts, efficiencies["significanceTrain"], color='green', label="Significance Train", linestyle='dashed')

ax.plot(ts, efficiencies["bkgVal"], color='blue', label="Bkg Val")
ax.plot(ts, efficiencies["sigVal"], color='red', label="Sig Val")
ax.plot(ts, efficiencies["significanceVal"], color='green', label="Significance Val")

#ax.plot(ts, efficiencies["bkgTest"], color='blue', label="Bkg Test", linestyle='-.')
#ax.plot(ts, efficiencies["sigTest"], color='red', label="Sig Test", linestyle='-.')
#ax.plot(ts, efficiencies["significanceTest"], color='green', label="Significance Test", linestyle='-.')
ax.legend()
best_index = np.argmax(efficiencies["significanceVal"])
best_t = ts[best_index]
best_significance = efficiencies["significanceVal"][best_index]
ax.text(x=0.05,y=0.4,s='Best Significance : %.3f\nNN>%.3f'%(best_significance, best_t), transform=ax.transAxes, fontsize=18)
fig.savefig(outFolder+"/performance/effScan.png", bbox_inches='tight')
print("Saved ", outFolder+"/performance/effScan.png")

print(f"Best cut threshold on validation set: {best_t}")
print(f"Maximum significance on validation set: {best_significance}")

# %%





# %%

#Xtest.columns = [str(Xtest.columns[_]) for _ in range((Xtest.shape[1]))]
Xval.columns = [str(Xval.columns[_]) for _ in range((Xval.shape[1]))]
#results = runPlotsTorch(Xtrain, Xval, Ytrain, Yval, np.ones(len(Xtrain)), np.ones(len(Xval)), YPredTrain, YPredVal, featuresForTraining, None, inFolder, outFolder, genMassTrain, genMassVal, results)
from helpers.doPlots import NNoutputs
pval_sig, pval_bkg = NNoutputs(signal_predictions=YPredVal[genMassVal==125], realData_predictions=YPredVal[genMassVal==0], signalTrain_predictions=YPredTrain[genMassTrain==125], realDataTrain_predictions=YPredTrain[Ytrain==0], outName=outFolder+"/performance/NNoutput.png", log=False, doubleDisco=False, label='NN output')

# %%
