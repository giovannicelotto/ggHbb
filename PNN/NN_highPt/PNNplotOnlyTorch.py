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
from helpers.loadSaved import loadXYrWSaved
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
    parser.add_argument("-l", "--lambdaCor", type=float, help="lambda for penalty term", default=None)
    parser.add_argument("-dt", "--date", type=str, help="MonthDay format e.g. Dec17", default=None)
    args = parser.parse_args()
    if args.lambdaCor is not None:
        hp["lambda_reg"] = args.lambdaCor 
    if args.date is not None:
        current_date = args.date
except:
    hp["lambda_reg"] = 200.06
    print("Interactive mode")
# %%
results = {}
inFolder, outFolder = getInfolderOutfolder(name = "%s_%s"%(current_date, str(hp["lambda_reg"]).replace('.', 'p')))
modelName = "model.pth"
featuresForTraining, columnsToRead = getFeatures(outFolder,  massHypo=True)

# %%
Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, rWtrain, rWval, Wtest, genMassTrain, genMassVal, genMassTest = loadXYrWSaved(inFolder=inFolder+"/data")
advFeatureTrain = np.load(inFolder+"/data/advFeatureTrain.npy")     
advFeatureVal   = np.load(inFolder+"/data/advFeatureVal.npy")



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


print("Variance for Val_Background : ", np.std(YPredVal[Yval==0])**2)
print("Variance for Val_Signal : ", np.std(YPredVal[genMassVal==125])**2)

import seaborn as sns
def plot_weights(model):
    # Set up the plot
    num_layers = len(list(model.named_parameters()))
    fig, axes = plt.subplots(num_layers, 1, figsize=(10, 2 * num_layers))

    # Iterate through the model's layers and plot the weights
    for idx, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad:
            # Get the weight tensor and convert to numpy
            weight_data = param.data.cpu().numpy()

            # Check the shape of the weights
            if len(weight_data.shape) == 4:  # Conv2d weights (filters)
                # Convert 4D weights into 2D for visualization (e.g., filter size vs. number of filters)
                weight_data = weight_data.reshape(weight_data.shape[0], -1)
            elif len(weight_data.shape) == 2:  # Fully connected layer weights
                # Use them as they are
                pass
            elif len(weight_data.shape) == 1:  # Bias term (1D)
                weight_data = weight_data.reshape(1, -1)  # Make it 2D for the heatmap

            # Plot the heatmap of weights
            ax = axes[idx] if num_layers > 1 else axes
            sns.heatmap(weight_data, ax=ax, cmap='coolwarm', cbar=True, square=True)
            ax.set_title(f"Layer: {name}, Shape: {weight_data.shape}")

    # Adjust layout
    plt.tight_layout()
    #plt.savefig(outF)

# Example: Plot weights of a model
#plot_weights(model)
# %%
from scipy.stats import kurtosis, skew
n=len(Xval[Yval==0])
gamma1 = skew(YPredVal[Yval==0])
gamma2 = kurtosis(YPredVal[Yval==0], fisher=True)  # Fisher's definition (excess kurtosis)
    
    # Calculate the bimodality coefficient
bimodality = (gamma1**2 + 1) / (gamma2 + 3 * ((n - 1)**2) / ((n - 2) * (n - 3)))
print("Bimodality : ", bimodality)



print("*"*30)
print("Run plots")
print("*"*30, "\n\n")
results = runPlotsTorch(Xtrain, Xval, Ytrain, Yval, np.ones(len(Xtrain)), np.ones(len(Xval)), YPredTrain, YPredVal, featuresForTraining, model, inFolder, outFolder, genMassTrain, genMassVal, results)
# %%
train_loss_history = np.load(outFolder + "/model/trainloss_history.npy")
val_loss_history = np.load(outFolder + "/model/val_loss_history.npy")
train_classifier_loss_history = np.load(outFolder + "/model/trainclassifier_loss_history.npy")
val_classifier_loss_history = np.load(outFolder + "/model/val_classifier_loss_history.npy")
train_dcor_loss_history = np.load(outFolder + "/model/traindcor_loss_history.npy")
val_dcor_loss_history = np.load(outFolder + "/model/val_dcor_loss_history.npy")
print("*"*30)
print("Loss function")
print("*"*30, "\n\n")

plot_lossTorch(train_loss_history, val_loss_history, 
              train_classifier_loss_history, val_classifier_loss_history,
              train_dcor_loss_history, val_dcor_loss_history,
              outFolder)
# %%
print("*"*30)
print("Orthogonality Check")
print("*"*30, "\n\n")

Xval['jet1_btagDeepFlavB'] = advFeatureVal
Xval['PNN'] = YPredVal
NN_t = 0.4
btag_t = 0.71
checkOrthogonality(Xval[Yval==0], 'jet1_btagDeepFlavB', np.array(YPredVal[Yval==0]>=NN_t), np.array(YPredVal[Yval==0]<NN_t), label_mask1='Prediction NN > %.1f'%NN_t, label_mask2='Prediction NN < %.1f'%NN_t, label_toPlot = 'jet1_btagDeepFlavB', bins=np.linspace(0.2783,1, 31), ax=None, axLegend=False, outName=outFolder+"/performance/orthogonalityInclusive.png" )
checkOrthogonality(Xval[Yval==0], 'PNN', Xval.jet1_btagDeepFlavB[Yval==0]>=btag_t, Xval.jet1_btagDeepFlavB[Yval==0]<btag_t, label_mask1='jet1_btagDeepFlavB > %.2f'%btag_t, label_mask2='jet1_btagDeepFlavB < %.2f'%btag_t, label_toPlot = 'NN', bins=np.linspace(0,1, 31), ax=None, axLegend=False, outName=outFolder+"/performance/NNorthogonalityInclusive.png" )


mass_bins = np.load(outFolder+"/mass_bins.npy")
ks_p_value_PNN, p_value_PNN, chi2_values_PNN = checkOrthogonalityInMassBins(
    df=Xval[Yval==0],
    featureToPlot='PNN',
    mask1=np.array(Xval[Yval==0]['jet1_btagDeepFlavB'] >= btag_t),
    mask2=np.array(Xval[Yval==0]['jet1_btagDeepFlavB'] < btag_t),
    label_mask1  = 'NN High',
    label_mask2  = 'NN Low',
    label_toPlot = 'PNN',
    bins=np.linspace(0., 1, 11),
    mass_bins=mass_bins,
    mass_feature='dijet_mass',
    figsize=(20, 25),
    outName=outFolder+"/performance/PNN_orthogonalityBinned.png" 
)

plotLocalPvalues(pvalues=ks_p_value_PNN, mass_bins=mass_bins, pvalueLabel="KS", outFolder=outFolder+"/performance/PNN_KSpvalues.png")
plotLocalPvalues(pvalues=p_value_PNN, mass_bins=mass_bins, pvalueLabel="$\chi^2$", outFolder=outFolder+"/performance/PNN_Chi2pvalues.png")


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
    figsize=(20, 25),
    outName=outFolder+"/performance/Jet1Btag_orthogonalityBinned.png" 
)

plotLocalPvalues(pvalues=ks_p_value_PNN, mass_bins=mass_bins, pvalueLabel="KS", outFolder=outFolder+"/performance/Jet1Btag_KSpvalues.png")
plotLocalPvalues(pvalues=p_value_PNN, mass_bins=mass_bins, pvalueLabel="$\chi^2$", outFolder=outFolder+"/performance/Jet1_BtagChi2pvalues.png")
# %%

print("*"*30)
print("Mutual Information binned")
print("*"*30, "\n\n")
mass_feature = 'dijet_mass'
for i, (low, high) in enumerate(zip(mass_bins[:-1], mass_bins[1:])):
        mask_mass = (Xval[mass_feature] >= low) & (Xval[mass_feature] < high) & (Yval==0)
        mi = mutual_info_regression(np.array(Xval[mask_mass].jet1_btagDeepFlavB).reshape(-1, 1), np.array(Xval[mask_mass].PNN))[0]
        print("%.1f < mjj < %.1f \n Mutual Information : %.5f"%(low, high, mi))

from scipy.stats import shapiro, kstest, norm, chi2
#Chi2 test
print("*"*30)
print("Chi2 binned")
print("*"*30, "\n\n")
mass_feature = 'dijet_mass'
chi2_tot = 0
for i, (low, high) in enumerate(zip(mass_bins[:-1], mass_bins[1:])):
        df = Xval[(Xval[mass_feature] >= low) & (Xval[mass_feature] < high) & (Yval==0)]
        regionA      = np.sum((df['jet1_btagDeepFlavB']<btag_t ) & (df['PNN']>NN_t) )
        regionB      = np.sum((df['jet1_btagDeepFlavB']>btag_t ) & (df['PNN']>NN_t) )
        regionC      = np.sum((df['jet1_btagDeepFlavB']<btag_t ) & (df['PNN']<NN_t) )
        regionD      = np.sum((df['jet1_btagDeepFlavB']>btag_t ) & (df['PNN']<NN_t) )
        
        expected = regionA*regionD/regionC
        err = regionA*regionD/regionC * np.sqrt(1/regionA + 1/regionC + 1/regionD)
        chi2_bin = (regionB - expected)**2/(err**2 + regionB)
        chi2_tot = chi2_tot+chi2_bin
        print("%.1f < mjj < %.1f \n Chi2 : %.5f"%(low, high, chi2_bin))
ndof = len(mass_bins)-1
print("ndof = ", ndof)
chi2_pvalue = 1 - chi2.cdf(chi2_tot, ndof)
print("chi2_tot", chi2_tot)
results['chi2_SR'] = chi2_tot
results['chi2pvalue_SR'] = chi2_pvalue




# Compute disCo
print("*"*30)
print("Disco binned")
print("*"*30, "\n\n")
import dcor
discos_bin = []
mass_bins = np.load(outFolder+"/mass_bins.npy")
for blow, bhigh in zip(mass_bins[:-1], mass_bins[1:]):
    print(blow, " < mjj < ", bhigh)
    mask = (Xval.dijet_mass >=blow) & (Xval.dijet_mass < bhigh)  & (Yval==0)
    distance_corr = dcor.distance_correlation(YPredVal[mask], advFeatureVal[mask])**2
    print("dcor : ", distance_corr)
    discos_bin.append(distance_corr)
results['averageBin_sqaured_disco'] = np.mean(discos_bin)
results['error_averageBin_sqaured_disco'] = np.std(discos_bin)/np.sqrt(len(discos_bin))


# use plotlocalPvalues to plot the dcor in function of dcor
plotLocalPvalues(discos_bin, mass_bins, pvalueLabel="Distance Corr.", type = '', outFolder=outFolder+"/performance/disco_mjjbins.png")

# %%
print("*"*30)
print("Pull ABCD Delta/sigmaDelta")
print("*"*30, "\n\n")
for i, (low, high) in enumerate(zip(mass_bins[:-1], mass_bins[1:])):

    regionA      = np.sum((Xval['jet1_btagDeepFlavB']<btag_t ) & (Xval['PNN']>NN_t) & (Yval==0) & (Xval['dijet_mass']>low) & (Xval['dijet_mass']<high))
    regionB      = np.sum((Xval['jet1_btagDeepFlavB']>btag_t ) & (Xval['PNN']>NN_t) & (Yval==0) & (Xval['dijet_mass']>low) & (Xval['dijet_mass']<high))
    regionC      = np.sum((Xval['jet1_btagDeepFlavB']<btag_t ) & (Xval['PNN']<NN_t) & (Yval==0) & (Xval['dijet_mass']>low) & (Xval['dijet_mass']<high))
    regionD      = np.sum((Xval['jet1_btagDeepFlavB']>btag_t ) & (Xval['PNN']<NN_t) & (Yval==0) & (Xval['dijet_mass']>low) & (Xval['dijet_mass']<high))

    print("Expected from ABCD :", regionA*regionD/regionC)
    print("Observed", regionB)
    pull = (regionB-regionA*regionD/regionC)/np.sqrt(regionB + (regionA*regionD/regionC)**2 * (1/regionA + 1/regionC + 1/regionD))
    print("Pull %.1f < mjj < %.1f : "%(low, high), pull)




from hist import Hist
x1 = 'jet1_btagDeepFlavB'
x2 = 'PNN'

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
mA      = (df[x1]<btag_t ) & (df[x2]>NN_t ) 
mB      = (df[x1]>btag_t ) & (df[x2]>NN_t ) 
mC      = (df[x1]<btag_t ) & (df[x2]<NN_t ) 
mD      = (df[x1]>btag_t ) & (df[x2]<NN_t ) 
regions['A'].fill(df[mA][xx])
regions['B'].fill(df[mB][xx])
regions['C'].fill(df[mC][xx])
regions['D'].fill(df[mD][xx])



sys.path.append("/t3home/gcelotto/ggHbb/abcd/new/helpersABCD")
from plot import plot4ABCD, QCD_SR
hB_ADC = plot4ABCD(regions, mass_bins, x1, x2, btag_t, NN_t, suffix='temp', blindPar=(False, 125, 20), outName=outFolder+"/performance/abcd_check4R")
qcd_mc = regions['B']
print(qcd_mc.values())
print(hB_ADC.values())
QCD_SR(mass_bins, hB_ADC, qcd_mc, nReal=0, suffix="temp", blindPar=(False, 125, 10), outName=outFolder+"/performance/abcd_checkSR")


# %%
import json
with open(outFolder+"/model/dict.json", "w") as file:
    json.dump(results, file, indent=4)