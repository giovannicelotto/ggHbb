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
from helpers.doPlots import runPlotsTorch
from helpers.loadSaved import loadXYrWSaved
import torch
from helpers.scaleUnscale import scale, unscale
sys.path.append('/t3home/gcelotto/ggHbb/scripts/plotScripts')
from plotFeatures import plotNormalizedFeatures
import numpy as np
import argparse
from datetime import datetime

# Get current month and day
current_date = datetime.now().strftime("%b%d")  # This gives the format like 'Dec12'
# %%
hp = getParams()
parser = argparse.ArgumentParser(description="Script.")
# Define arguments
try:
    parser.add_argument("-l", "--lambdaCor", type=float, help="lambda for penalty term", default=None)
    args = parser.parse_args()
    if args.lambdaCor is not None:
        hp["lambda_reg"] = args.lambdaCor 
except:
    print("Interactive mode")
# %%
inFolder, outFolder = getInfolderOutfolder(name = "%s_%s"%(current_date, str(hp["lambda_reg"]).replace('.', 'p')))
modelName = "model.pth"
featuresForTraining, columnsToRead = getFeatures(outFolder,  massHypo=True)
#featuresForTraining = featuresForTraining + ["massHypo"]
# %%
Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, rWtrain, rWval, Wtest, genMassTrain, genMassVal, genMassTest = loadXYrWSaved(inFolder=inFolder+"/data")
advFeatureTrain = np.load(inFolder+"/data/advFeatureTrain.npy")     
advFeatureVal   = np.load(inFolder+"/data/advFeatureVal.npy")
featuresForTraining, columnsToRead = getFeatures(outFolder, massHypo=True)

model = torch.load(outFolder+"/model/%s"%modelName, map_location=torch.device('cpu'))
model.eval()
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

runPlotsTorch(Xtrain, Xval, Ytrain, Yval, np.ones(len(Xtrain)), np.ones(len(Xval)), YPredTrain, YPredVal, featuresForTraining, model, inFolder, outFolder, genMassTrain, genMassVal)
# %%
train_loss_history = np.load(outFolder + "/model/trainloss_history.npy")
val_loss_history = np.load(outFolder + "/model/val_loss_history.npy")
train_classifier_loss_history = np.load(outFolder + "/model/trainclassifier_loss_history.npy")
val_classifier_loss_history = np.load(outFolder + "/model/val_classifier_loss_history.npy")
train_dcor_loss_history = np.load(outFolder + "/model/traindcor_loss_history.npy")
val_dcor_loss_history = np.load(outFolder + "/model/val_dcor_loss_history.npy")
from helpers.doPlots import plot_lossTorch
plot_lossTorch(train_loss_history, val_loss_history, 
              train_classifier_loss_history, val_classifier_loss_history,
              train_dcor_loss_history, val_dcor_loss_history,
              outFolder)
# %%
#runPlotsTorch(Xtrain, Xval, Ytrain, Yval, Wtrain, Wval, YPredTrain, YPredVal, featuresForTraining, model, inFolder, outFolder, genMassTrain, genMassVal)
Xval['jet1_btagDeepFlavB'] = advFeatureVal
Xval['PNN'] = YPredVal
NN_t = 0.5
checkOrthogonality(Xval[Yval==0], 'jet1_btagDeepFlavB', np.array(YPredVal[Yval==0]>=NN_t), np.array(YPredVal[Yval==0]<NN_t), label_mask1='Prediction NN > %.1f'%NN_t, label_mask2='Prediction NN < %.1f'%NN_t, label_toPlot = 'jet1_btagDeepFlavB', bins=np.linspace(0.2783,1, 31), ax=None, axLegend=False, outName=outFolder+"/performance/orthogonalityInclusive.png" )
checkOrthogonality(Xval[Yval==0], 'PNN', Xval.jet1_btagDeepFlavB[Yval==0]>=0.71, Xval.jet1_btagDeepFlavB[Yval==0]<0.71, label_mask1='jet1_btagDeepFlavB > 0.71', label_mask2='jet1_btagDeepFlavB < 0.71', label_toPlot = 'NN', bins=np.linspace(0,1, 31), ax=None, axLegend=False, outName=outFolder+"/performance/NNorthogonalityInclusive.png" )


#newT = (np.arctanh(0.71) - np.arctanh(advFeatureVal).min())/ (np.arctanh(advFeatureVal).max() - np.arctanh(advFeatureVal).min())
#Xval['jet1_btagDeepFlavB'] = (np.arctanh(advFeatureVal) - np.arctanh(advFeatureVal).min())/ (np.arctanh(advFeatureVal).max() - np.arctanh(advFeatureVal).min())
#Xval['PNN'] = YPredVal
#NN_t = 0.5
#checkOrthogonality(Xval[Yval==0], 'jet1_btagDeepFlavB', np.array(YPredVal[Yval==0]>NN_t), np.array(YPredVal[Yval==0]<NN_t), label_mask1='Prediction NN > %.1f'%NN_t, label_mask2='Prediction NN < %.1f'%NN_t, label_toPlot = 'jet1_btagDeepFlavB', bins=np.linspace(0.2783,1, 31), ax=None, axLegend=False, outName=outFolder+"/performance/orthogonalityInclusive.png" )
#checkOrthogonality(Xval[Yval==0], 'PNN', Xval.jet1_btagDeepFlavB[Yval==0]>newT, Xval.jet1_btagDeepFlavB[Yval==0]<newT, label_mask1='jet1_btagDeepFlavB > %.2f'%newT, label_mask2='jet1_btagDeepFlavB < %.2f'%newT, label_toPlot = 'NN', bins=np.linspace(0,1, 31), ax=None, axLegend=False, outName=outFolder+"/performance/NNorthogonalityInclusive.png" )

# %%
Xval['PNN'] = YPredVal
mass_bins = np.linspace(40, 300, 16)
ks_p_value_PNN, p_value_PNN, chi2_values_PNN = checkOrthogonalityInMassBins(
    df=Xval[Yval==0],
    featureToPlot='PNN',
    mask1=np.array(Xval[Yval==0]['jet1_btagDeepFlavB'] >= 0.7100),
    mask2=np.array(Xval[Yval==0]['jet1_btagDeepFlavB'] < 0.7100),
    label_mask1  = 'NN High',
    label_mask2  = 'NN Low',
    label_toPlot = 'PNN',
    bins=np.linspace(0., 1, 11),
    mass_bins=mass_bins,
    mass_feature='dijet_mass',
    figsize=(20, 30),
    outName=outFolder+"/performance/PNN_orthogonalityBinned.png" 
)

plotLocalPvalues(pvalues=ks_p_value_PNN, mass_bins=mass_bins, pvalueLabel="KS", outFolder=outFolder+"/performance/PNN_KSpvalues.png")
plotLocalPvalues(pvalues=p_value_PNN, mass_bins=mass_bins, pvalueLabel="$\chi^2$", outFolder=outFolder+"/performance/PNN_Chi2pvalues.png")

# %%
ks_p_value_PNN, p_value_PNN, chi2_values_PNN = checkOrthogonalityInMassBins(
    df=Xval[Yval==0],
    featureToPlot='jet1_btagDeepFlavB',
    mask1=np.array(Xval[Yval==0]['PNN'] >= NN_t),
    mask2=np.array(Xval[Yval==0]['PNN'] < NN_t),
    label_mask1  = 'Btag High',
    label_mask2  = 'Btag Low',
    label_toPlot = 'Jet1 btag',
    bins=np.linspace(0.2783, 1, 11),
    mass_bins=mass_bins,
    mass_feature='dijet_mass',
    figsize=(20, 30),
    outName=outFolder+"/performance/Jet1Btag_orthogonalityBinned.png" 
)

plotLocalPvalues(pvalues=ks_p_value_PNN, mass_bins=mass_bins, pvalueLabel="KS", outFolder=outFolder+"/performance/Jet1Btag_KSpvalues.png")
plotLocalPvalues(pvalues=p_value_PNN, mass_bins=mass_bins, pvalueLabel="$\chi^2$", outFolder=outFolder+"/performance/Jet1_BtagChi2pvalues.png")
# %%
