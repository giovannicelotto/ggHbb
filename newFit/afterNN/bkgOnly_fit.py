# %%
import numpy as np
import pandas as pd
from functions import cut, getDfProcesses_v2
import mplhep as hep
hep.style.use("CMS")
import json, sys
import yaml
sys.path.append("/t3home/gcelotto/ggHbb/newFit/afterNN/")
from helpers.doFitParameterFixed import doFitParameterFixed
from helpers.doFit_bkgOnly import doFit_bkgOnly
from helpers.doFitParameterFree import doFitParameterFree
from helpers.plotFree import plotFree
from helpers.defineFunctions import defineFunctions
from helpers.allFunctions import *
from iminuit.cost import LeastSquares
from iminuit import Minuit
from scipy.stats import chi2
import inspect
from helpers.getBounds import getBounds
import argparse
# %%


# %%
#parser = argparse.ArgumentParser()
#parser.add_argument('-i', '--idx', type=int, default=2, help='Index value (default: 2)')
#
#args = parser.parse_args()
#idx = args.
idx = 2



config_path = ["/t3home/gcelotto/ggHbb/newFit/afterNN/cat1/bkgPlusZFit_config.yml",
               "/t3home/gcelotto/ggHbb/newFit/afterNN/cat2/cat2p0/bkgPlusZFit_config.yml",
               "/t3home/gcelotto/ggHbb/newFit/afterNN/cat2/cat2p1/bkgOnly_config.yml"][idx]
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
x1 = config["x1"]
x2 = config["x2"]
key = config["key"]
nbins = config["nbins"]
t0 = config["t0"]
t1 = config["t1"]
t2 = config["t2"]
t3 = config["t3"]
isDataList = config["isDataList"]
modelName = config["modelName"]
ptCut_min = config["ptCut_min"]
ptCut_max = config["ptCut_max"]
jet1_btagMin = config["jet1_btagMin"]
jet2_btagMin = config["jet2_btagMin"]
PNN_t = config["PNN_t"]
plotFolder = config["plotFolder"]
MCList_Z = config["MCList_Z"]
MCList_H = config["MCList_H"]
params = {}
paramsLimits = config["paramsLimits"]
output_file = config["output_file"]
fitZSystematics = config["fitZSystematics"]
PNN_t_max=config["PNN_t_max"]
set_x_bounds(x1, x2)
myBkgFunctions, myBkgSignalFunctions, myBkgParams = defineFunctions()




path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/mjjDisco/%s"%modelName
dfProcessesMC, dfProcessesData = getDfProcesses_v2()
dfProcessesData = dfProcessesData.iloc[isDataList]


# %%
# Data
dfsData = []
lumi_tot = 0.
for processName in dfProcessesData.process.values:
    print("Opening ", processName)
    df = pd.read_parquet(path+"/dataframes_%s_%s.parquet"%(processName, modelName))
    dfsData.append(df)
    lumi_tot = lumi_tot + np.load(path+"/lumi_%s_%s.npy"%(processName, modelName))




# %%

dfsMC_Z = []
for processName in dfProcessesMC.iloc[MCList_Z].process.values:
    print("Opening ", processName)
    df = pd.read_parquet(path+"/df_%s_%s.parquet"%(processName, modelName))
    dfsMC_Z.append(df)


dfsMC_H = []
for processName in dfProcessesMC.iloc[MCList_H].process.values:
    print("Opening ", processName)
    df = pd.read_parquet(path+"/df_%s_%s.parquet"%(processName, modelName))
    dfsMC_H.append(df)

# %%
for idx, df in enumerate(dfsMC_H):
    dfsMC_H[idx].weight=dfsMC_H[idx].weight*lumi_tot

for idx, df in enumerate(dfsMC_Z):
    dfsMC_Z[idx].weight=dfsMC_Z[idx].weight*lumi_tot
# %%
dfsData = cut(dfsData, 'PNN', PNN_t, PNN_t_max)
dfsMC_Z = cut(dfsMC_Z, 'PNN', PNN_t, PNN_t_max)
dfsMC_H = cut(dfsMC_H, 'PNN', PNN_t, PNN_t_max)
dfsData = cut(dfsData, 'dijet_pt', ptCut_min, ptCut_max)
dfsMC_Z = cut(dfsMC_Z, 'dijet_pt', ptCut_min, ptCut_max)
dfsMC_H = cut(dfsMC_H, 'dijet_pt', ptCut_min, ptCut_max)
dfsData = cut(dfsData, 'jet1_btagDeepFlavB', jet1_btagMin, None)
dfsMC_Z = cut(dfsMC_Z, 'jet1_btagDeepFlavB', jet1_btagMin, None)
dfsMC_H = cut(dfsMC_H, 'jet1_btagDeepFlavB', jet1_btagMin, None)
dfsData = cut(dfsData, 'jet2_btagDeepFlavB', jet2_btagMin, None)
dfsMC_Z = cut(dfsMC_Z, 'jet2_btagDeepFlavB', jet2_btagMin, None)
dfsMC_H = cut(dfsMC_H, 'jet2_btagDeepFlavB', jet2_btagMin, None)

for idx, df in enumerate(dfsMC_H):
    dfsMC_H[idx]['process'] = dfProcessesMC.iloc[MCList_H].iloc[idx].process
for idx, df in enumerate(dfsMC_Z):
    dfsMC_Z[idx]['process'] = dfProcessesMC.iloc[MCList_Z].iloc[idx].process

dfMC_Z = pd.concat(dfsMC_Z)
dfMC_H = pd.concat(dfsMC_H)
df = pd.concat(dfsData)
#Nominal


bins = np.linspace(x1, x2, nbins)
x=(bins[1:]+bins[:-1])/2
c=np.histogram(df.dijet_mass, bins=bins)[0]
maskFit = ((x>t0)&(x < t1)) | ((x<t3)&(x > t2))
maskUnblind = (x < t1) | (x > t2)
##
## variation happening
## 



# %%
import matplotlib.pyplot as plt
set_x_bounds(x1, x2)
x_fit = x[maskFit]
y_tofit = c[maskFit]
yerr = np.sqrt(y_tofit)

# Plot Data
fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(18, 15), constrained_layout=True)
fig.align_ylabels([ax[0],ax[1]])
ax[0].errorbar(x, c, yerr=np.sqrt(c), fmt='o', color='black', markersize=3, label="Data")
# Plot Z

cZ=ax[0].hist(dfMC_Z.dijet_mass, bins=bins, weights = dfMC_Z.weight, label='Zbb')[0]
cZ_err=np.histogram(dfMC_Z.dijet_mass, bins=bins, weights = (dfMC_Z.weight)**2)[0]
cZ_err = np.sqrt(cZ_err)
cumulativeMC=cZ.copy()

cHiggs = np.zeros(len(bins)-1)
for process in np.unique(dfMC_H.process):
    maskProcess = dfMC_H.process==process
    c_=ax[0].hist(dfMC_H[maskProcess].dijet_mass, bins=bins, weights = dfMC_H.weight[maskProcess] , label=process, bottom=cumulativeMC)[0]
    cHiggs += c_
    cumulativeMC +=c_


    
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
# Data for GPR
x_fit = x[maskFit].reshape(-1, 1)
y_tofit = c[maskFit]
yerr = np.sqrt(y_tofit)


from sklearn.preprocessing import StandardScaler

# Standardize x
x_scaler = StandardScaler()
x_fit_scaled = x_scaler.fit_transform(x_fit)

# Standardize y
y_scaler = StandardScaler()
y_tofit_scaled = y_scaler.fit_transform(y_tofit.reshape(-1, 1)).ravel()


# Define kernel: you can tune this
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

# Try a kernel with a white noise component
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))


from sklearn.gaussian_process import GaussianProcessRegressor

gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=10, normalize_y=False)
gp.fit(x_fit_scaled, y_tofit_scaled)

x_pred_scaled = x_scaler.transform(x.reshape(-1, 1))
y_pred_scaled, sigma_scaled = gp.predict(x_pred_scaled, return_std=True)

# Convert predictions back to original y-scale
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
sigma = sigma_scaled * y_scaler.scale_[0]
# Upper plot

ax[0].plot(x, y_pred, label="GPR Fit", color='red')
ax[0].fill_between(x, y_pred - sigma, y_pred + sigma, color='red', alpha=0.3)

residuals = c - y_pred
ax[1].errorbar(x, residuals, yerr=np.sqrt(c), fmt='o', color='black')

ax[1].axhline(0, color='red', linestyle='--')

ax[1].hist(dfMC_Z.dijet_mass, bins=bins, weights = dfMC_Z.weight, label='Zbb')[0]
cumulativeMC=cumulativeMC*0
for process in np.unique(dfMC_H.process):
    maskProcess = dfMC_H.process==process
    c_=ax[1].hist(dfMC_H[maskProcess].dijet_mass, bins=bins, weights = dfMC_H.weight[maskProcess] , label=process, bottom=cumulativeMC)[0]
    cHiggs += c_
    cumulativeMC +=c_


from scipy.stats import chi2 as chi2_dist

# Use only bins that were fitted
y_obs = c[maskFit]
y_pred_fit = y_pred[maskFit]
y_err = np.sqrt(c[maskFit])  # Poisson error from data

# Optional: include GPR uncertainty
# total_err = np.sqrt(y_err**2 + sigma[maskFit]**2)
total_err = y_err  # or use total_err if you want to include GPR uncertainty

# Compute chi2
chi2_val = np.sum((y_obs - y_pred_fit)**2 / total_err**2)
ndof = len(y_obs) - gp.kernel_.n_dims  # or just len(y_obs) - 1 if unsure
p_val = 1 - chi2_dist.cdf(chi2_val, ndof)
ax[1].fill_between(x, 0 - sigma, 0 + sigma, color='red', alpha=0.3)
print(f"Chi2 = {chi2_val:.2f}, ndof = {ndof}, p-value = {p_val:.3f}")




















# Minuit fits
# %%
#params["normSig"] = cZ.sum()*(bins[1]-bins[0]) -  dfMC_Z[dfMC_Z.dijet_mass<bins[0]].weight.sum()
def exp_gaus_turnOn(x, norm, B, b, c, mu_to, sigma_to, offset, f):
    def unnormalized_function(x):
        pol2 = c * (x-offset)**2 + b * (x-offset) + 1  # Quadratic polynomial
        exp1 = np.exp(-B * (x-offset))
        turnOn = np.exp(-0.5*((x)-mu_to)**2/sigma_to**2)
        return f*(exp1) * pol2 + (1-f)*turnOn
    
    # Compute normalization constant to ensure integral from x1 to x2 is 1
    integral_value, _ = integrate.quad(unnormalized_function, x1, x2)
    normalization_factor = norm / (integral_value+1e-12)
    
    return normalization_factor * unnormalized_function(x)

def f(x, m0, m_exp, p0, p1, norm):
    base = norm*(p1*(x-m0) + p0)*np.exp(-m_exp*(x-m0))
    
    return base

least_squares = LeastSquares(x_fit, y_tofit, yerr, f)
param_names = ["m0", "m_exp", "p1", "p0", "norm"]
params = dict.fromkeys(param_names, 0.0)
params["m0"]=30
params["m_exp"]=0.02
params["norm"]=c.sum()*(bins[1]-bins[0])-cZ.sum()*(bins[1]-bins[0])


print(params)
m_tot = Minuit(least_squares,
                **params
                )


    # Apply limits using the dictionary
for par in m_tot.parameters:
    print(par)
    if par in paramsLimits:
        m_tot.limits[par] = paramsLimits[par]  # Assign limits from the dictionary
    else:
        m_tot.limits[par] = None  # No limits if not specified



m_tot.migrad(ncall=100000, iterate=200)
m_tot.hesse()

p_tot = m_tot.values
print(p_tot)

# Generate fit curves
x_plot = np.linspace(bins[0], bins[-1], 500)
y_tot = f(x_plot, *p_tot)



ax[0].plot(x_plot, y_tot, label="Fit (Background + Z Peak)", color='red')
ax[0].fill_between(x, 0, max(c)*1.2, where=maskFit, color='green', alpha=0.2)
ax[0].set_xlim(bins[0], bins[-1])
ax[0].set_ylim(1, max(c)*1.2)

ax[1].errorbar(x[maskFit], (c-f(x, *p_tot))[maskFit], yerr=np.sqrt(c)[maskFit], fmt='o', color='black', markersize=3)
ax[1].set_ylim(ax[1].get_ylim())
ax[1].set_ylabel("Data - Background")
ax[1].fill_between(x, ax[1].get_ylim()[0], ax[1].get_ylim()[1], where=maskFit, color='green', alpha=0.2)

ax[1].hist(bins[:-1], bins=bins, weights = cZ)[0]
ax[1].hist(bins[:-1], bins=bins, weights = cHiggs, bottom=cZ)[0]

hep.cms.label(lumi="%.2f" %lumi_tot, ax=ax[0])
ax[1].set_xlabel("Dijet Mass [GeV]")
ax[0].set_ylabel("Counts")
ax[0].legend(bbox_to_anchor=(1, 1))

chi2_stat = np.sum(((c[maskFit] - f(x, *p_tot)[maskFit])**2) / np.sqrt(c)[maskFit]**2)
ndof = m_tot.ndof
chi2_pvalue = 1- chi2.cdf(chi2_stat, ndof)
ax[0].text(x=0.05, y=0.75, s="$\chi^2$/ndof = %.1f/%d\np-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax[0].transAxes, ha='left')

# %%
