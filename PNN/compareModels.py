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
from helpers.doPlots import *
import numpy as np
import re
from datetime import datetime
from sklearn.feature_selection import mutual_info_regression
import dcor
import json



models = [
"Jan14_1000p0",
"Jan14_100p0",
"Jan14_150p0",
"Jan14_200p0",
"Jan14_200p01",
"Jan14_250p0",
"Jan14_300p0",
"Jan14_300p01",
"Jan14_350p0",
"Jan14_400p0",
"Jan14_450p0",
"Jan14_450p01",
#"Jan14_45p0",
"Jan14_500p0",
"Jan14_50p0",
#"Jan14_55p0",
"Jan14_600p0",
"Jan14_60p0",
#"Jan14_65p0",
"Jan14_700p0",
"Jan14_70p0",
#"Jan14_75p0",
"Jan14_800p0",
"Jan14_80p0",
#"Jan14_85p0",
"Jan14_900p0",
"Jan14_90p0",
#"Jan14_95p0",
"Jan15_1050p0",
"Jan15_125p0",
"Jan15_175p0",
"Jan15_225p0",
"Jan15_275p0",
"Jan15_325p0",
"Jan15_375p0",
"Jan15_425p0",
"Jan15_475p0",
#"Jan15_550p0",
"Jan15_650p0",
#"Jan15_750p0",
"Jan15_850p0",
#"Jan15_950p0",

]
lambdas = np.array([float(re.search(r'(\d+)p(\d+)', model).group(1) + '.' + re.search(r'(\d+)p(\d+)', model).group(2)) for model in models])
# %%
auc_rocs = []
discos = []
err_disco = []
chi2_SR = []
for suffix in models:
    print(suffix)
    inFolder, outFolder = getInfolderOutfolder(name = suffix, doubleDisco=True)
    

    with open(outFolder+"/model/dict.json", "r") as file:
        data = json.load(file)
    auc_rocs.append(data['roc_125'])
    #discos.append(data['averageBin_sqaured_disco'])
    chi2_SR.append(data['chi2_SR'])
    #err_disco.append(data['error_averageBin_sqaured_disco'])
    


# %%
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
# Define kernel for Gaussian Process
l = 10 #rules the scale of influence horizontally
kernel = C(1.0, (1e-3, 1e3)) * RBF(l, (1e-2, 1e2))

# Create Gaussian Process Regressor
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6)

# Fit the model
gp.fit(lambdas.reshape(-1, 1), auc_rocs)

# Predict values for a finer range
lambda_pred = np.linspace(0, 500, 100).reshape(-1, 1)  # Intermediate points
auc_pred, sigma = gp.predict(lambda_pred, return_std=True)



# Same for disco
#kernel = (
#    C(1e-4, (1e-6, 1e-2))  # Variance scale matches the y-values
#    * RBF(50, (10, 300))   # Length scale on the x-values range
#    + WhiteKernel(noise_level=1e-10, noise_level_bounds=(1e-12, 1e-8))  # Variance from uncertainties
#)
#gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-9)
#gp.fit(lambdas.reshape(-1, 1), discos)
#discos_pred, discos_sigma = gp.predict(lambda_pred, return_std=True)

# %%
# Plot
fig, ax = plt.subplots(1, 1)
ax.plot(lambdas, auc_rocs, 'o', label="AUC", color='black')
ax.set_ylim(0, 1)
ax.set_xlabel("$\lambda$")
ax.set_ylabel("AUC-ROC")
# Add the secondary y-axis
ax2 = ax.twinx()
ax2.errorbar(lambdas, np.array(chi2_SR), None, marker='o', label="$\chi^2 SR$", color='red', linestyle='none')
ax.set_xlim(ax.get_xlim())
ax2.hlines(xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], y=19, label='ndof', linestyle='dotted', linewidth=2)
#ax2.set_ylim(0, 0.001)
#ax.plot(lambda_pred, auc_pred, '-', label="GP Prediction")
#ax.fill_between(
#    lambda_pred.ravel(),
#    auc_pred - 1.96 * sigma,
#    auc_pred + 1.96 * sigma,
#    color='gray',
#    alpha=0.2,
#    label="95% Confidence Interval",
#)
#ax2.plot(lambda_pred, discos_pred, '-', label="GP Prediction", color='red')
#ax2.fill_between(
#    lambda_pred.ravel(),
#    discos_pred - 1.96 * discos_sigma,
#    discos_pred + 1.96 * discos_sigma,
#    color='red',
#    alpha=0.2,
#    label="95% Confidence Interval",
#)
#ax2.set_ylim(np.min(discos)*0.9, np.max(discos)*1.1)


#ax2.set_ylabel("Average-MassBin Squared Distance Correlation", color='red')
ax2.set_ylabel("Chi2 SR", color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Add a legend for the secondary y-axis
lines_1, labels_1 = ax.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
fig.savefig("/t3home/gcelotto/ggHbb/PNN/results/comparedModels.png")
# %%
