# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functions import loadMultiParquet_v2, loadMultiParquet_Data_new, cut, getDfProcesses_v2, getCommonFilters
import mplhep as hep
hep.style.use("CMS")
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2
import json, sys
sys.path.append("/t3home/gcelotto/newFit/afterNN/")
from helpers.allFunctions import *
from helpers.createRootHist import createRootHists
columns=['dijet_mass', 'dijet_pt']
import yaml

with open("/t3home/gcelotto/newFit/afterNN/cat2/cat2p1/config_2.yaml", "r") as f:
    params = yaml.safe_load(f)  # Read as dictionary

x1 = params["x1"]
x2 = params["x2"]
key = params["key"]
nbins = params["nbins"]
t1 = params["t1"]
t2 = params["t2"]
t0 = params["t0"]
t3 = params["t3"]

set_x_bounds(x1, x2)

myBkgFunctions = {
    1:continuum_background_1,
    2:continuum_background_2,
    3:continuum_background_3}
myBkgSignalFunctions = {
    1:continuum_plus_Z_1,
    2:continuum_plus_Z_2,
    3:continuum_plus_Z_3}
myBkgParams = {
    1:["normBkg", "B", "b"],
    2:["normBkg", "B", "b", "c"],
    3:["normBkg", "B", "b", "c", "d"]
}





isDataList = [1,
              2]
MCList = [0,1, 3, 4, 19, 20, 21, 22, 36]
modelName = "Mar06_2_0p0"
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
MCList_Z = [1, 3, 4, 19, 20, 21, 22]
dfsMC_Z = []
for processName in dfProcessesMC.iloc[MCList_Z].process.values:
    print("Opening ", processName)
    df = pd.read_parquet(path+"/df_%s_%s.parquet"%(processName, modelName))
    dfsMC_Z.append(df)

MCList_H = [37, 36]
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
ptCut_min = 160
ptCut_max = None
jet1_btagMin = 0.71
jet2_btagMin = 0.71
PNN_t = 0.7
dfsData = cut(dfsData, 'PNN', PNN_t, None)
dfsMC_Z = cut(dfsMC_Z, 'PNN', PNN_t, None)
dfsMC_H = cut(dfsMC_H, 'PNN', PNN_t, None)
dfsData = cut(dfsData, 'dijet_pt', ptCut_min, None)
dfsMC_Z = cut(dfsMC_Z, 'dijet_pt', ptCut_min, None)
dfsMC_H = cut(dfsMC_H, 'dijet_pt', ptCut_min, None)
dfsData = cut(dfsData, 'jet1_btagDeepFlavB', jet1_btagMin, None)
dfsMC_Z = cut(dfsMC_Z, 'jet1_btagDeepFlavB', jet1_btagMin, None)
dfsMC_H = cut(dfsMC_H, 'jet1_btagDeepFlavB', jet1_btagMin, None)
dfsData = cut(dfsData, 'jet2_btagDeepFlavB', jet2_btagMin, None)
dfsMC_Z = cut(dfsMC_Z, 'jet2_btagDeepFlavB', jet2_btagMin, None)
dfsMC_H = cut(dfsMC_H, 'jet2_btagDeepFlavB', jet2_btagMin, None)

df = pd.concat(dfsData)
dfMC_Z = pd.concat(dfsMC_Z)
dfMC_H = pd.concat(dfsMC_H)

bins = np.linspace(x1, x2, nbins)
x=(bins[1:]+bins[:-1])/2
c=np.histogram(df.dijet_mass, bins=bins)[0]


# %%
# Blind a region for the fit


maskFit = ((x>t0)&(x < t1)) | ((x<t3)&(x > t2))
maskUnblind = (x < t1) | (x > t2)
x_fit = x[maskFit]
y_tofit = c[maskFit]
yerr = np.sqrt(y_tofit)

# Plot Data
fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(18, 15), constrained_layout=True)
fig.align_ylabels([ax[0],ax[1]])
ax[0].errorbar(x[maskUnblind], c[maskUnblind], yerr=np.sqrt(c)[maskUnblind], fmt='o', color='black', markersize=3, label="Data")
# Plot Z
cZ = np.zeros(len(bins)-1)
for idx, dfMC in enumerate(dfsMC_Z):
    c_=ax[0].hist(dfMC.dijet_mass, bins=bins, weights = dfMC.weight, label=dfProcessesMC.iloc[MCList_Z].process.iloc[idx], bottom=cZ)[0]
    cZ += c_
cumulativeMC = cZ.copy()
cHiggs = np.zeros(len(bins)-1)
for idx, dfMC in enumerate(dfsMC_H):
    c_=ax[0].hist(dfMC.dijet_mass, bins=bins, weights = dfMC.weight , label=dfProcessesMC.iloc[MCList_H].process.iloc[idx], bottom=cumulativeMC)[0]
    cHiggs += c_
    cumulativeMC +=c_
    

with open("/t3home/gcelotto/newFit/afterNN/cat2/cat2p1/fit_parameters_Z_cat2p1.json", "r") as f:
    fit_parameters_zPeak = json.load(f)


# Minuit fits

least_squares = LeastSquares(x_fit, y_tofit, yerr, myBkgSignalFunctions[key])
params = {
            "normBkg":c.sum()*(bins[1]-bins[0])-cZ.sum()*(bins[1]-bins[0]),
            "B":0.0127,
            "b":-0.0098,
            
               
            "normSig":cZ.sum()*(bins[1]-bins[0]) -  dfMC_Z[dfMC_Z.dijet_mass<bins[0]].weight.sum(),
            "fraction_dscb":fit_parameters_zPeak["fraction_dscb"]['value'],
            "mean":fit_parameters_zPeak['mean']['value'],
            "sigma":fit_parameters_zPeak['sigma']['value'],
            "alphaL":fit_parameters_zPeak['alphaL']['value'],
            "nL":fit_parameters_zPeak['nL']['value'],
            "alphaR":fit_parameters_zPeak['alphaR']['value'],
            "nR":fit_parameters_zPeak['nR']['value'],
            #"fraction_gaussian":fit_parameters_zPeak['fraction_gaussian']['value'],
            "sigmaG":fit_parameters_zPeak['sigmaG']['value'],
            #"p1":fit_parameters_zPeak['p1']['value']
}
if key == 2:
    params["c"] = 0
elif key == 3:
    params["c"] = 0
    params["d"] = 0
m_tot = Minuit(least_squares,
               **params
               )

m_tot.fixed['sigma'] = True
m_tot.fixed['mean'] = True
m_tot.fixed['sigmaG'] = True
m_tot.fixed['normSig'] = True
m_tot.fixed['alphaR'] = True
m_tot.fixed['nR'] = True
m_tot.fixed['alphaL'] = True
m_tot.fixed['nL'] = True
m_tot.fixed["fraction_dscb"]=True



m_tot.migrad(ncall=20000, iterate=10)
m_tot.hesse()

p_tot = m_tot.values

# Generate fit curves
x_plot = np.linspace(bins[0], bins[-1], 500)
y_tot = myBkgSignalFunctions[key](x_plot, *p_tot)



ax[0].plot(x_plot, y_tot, label="Fit (Background + Z Peak)", color='red')
ax[0].fill_between(x, 0, max(c)*1.2, where=maskFit, color='green', alpha=0.2)
ax[0].set_xlim(bins[0], bins[-1])
ax[0].set_ylim(1, max(c)*1.2)

ax[1].errorbar(x[maskFit], (c-myBkgFunctions[key](x, *p_tot[myBkgParams[key]]))[maskFit], yerr=np.sqrt(c)[maskFit], fmt='o', color='black', markersize=3)
ax[1].plot(x, zPeak(x, *p_tot[[ 'normSig', 'fraction_dscb', 'mean', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR', 'sigmaG']]), color='red', linewidth=2)
ax[1].set_ylabel("Data - Background")
ax[1].set_ylim(ax[1].get_ylim())
ax[1].fill_between(x, ax[1].get_ylim()[0], ax[1].get_ylim()[1], where=maskFit, color='green', alpha=0.2)

ax[1].hist(bins[:-1], bins=bins, weights = cZ)[0]
ax[1].hist(bins[:-1], bins=bins, weights = cHiggs, bottom=cZ)[0]

hep.cms.label(lumi="%.2f" %lumi_tot, ax=ax[0])
ax[1].set_xlabel("Dijet Mass [GeV]")
ax[0].set_ylabel("Counts")
ax[0].legend(bbox_to_anchor=(1, 1))

chi2_stat = np.sum(((c[maskFit] - myBkgSignalFunctions[key](x, *p_tot)[maskFit])**2) / np.sqrt(c)[maskFit]**2)
ndof = m_tot.ndof
chi2_pvalue = 1- chi2.cdf(chi2_stat, ndof)
ax[0].text(x=0.05, y=0.75, s="$\chi^2$/ndof = %.1f/%d\np-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax[0].transAxes, ha='left')
ax[0].set_yscale('log')

#fig.savefig("/t3home/gcelotto/newFit/bkgPlusZPeak.png", bbox_inches='tight')

# %%


# Second Fit parameters free
# Minuit fits
# Plot Data
fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(18, 15), constrained_layout=True)
fig.align_ylabels([ax[0],ax[1]])
ax[0].errorbar(x[maskUnblind], c[maskUnblind], yerr=np.sqrt(c)[maskUnblind], fmt='o', color='black', markersize=3, label="Data")
# Plot Z
bot = np.zeros(len(bins)-1)
for idx, dfMC in enumerate(dfsMC_Z):
    c_=ax[0].hist(dfMC.dijet_mass, bins=bins, weights = dfMC.weight, label=dfProcessesMC.iloc[MCList_Z].process.iloc[idx], bottom=bot)[0]
    bot += c_
cumulativeMC = bot.copy()
cHiggs = np.zeros(len(bins)-1)
for idx, dfMC in enumerate(dfsMC_H):
    c_=ax[0].hist(dfMC.dijet_mass, bins=bins, weights = dfMC.weight , label=dfProcessesMC.iloc[MCList_H].process.iloc[idx], bottom=cumulativeMC)[0]
    cHiggs += c_
    cumulativeMC +=c_
least_squares = LeastSquares(x_fit, y_tofit, yerr, myBkgSignalFunctions[key])
params ={
    "normBkg":m_tot.values["normBkg"],
    "B":m_tot.values["B"],
    "b":m_tot.values["b"],
    "normSig":m_tot.values["normSig"],
    "fraction_dscb":m_tot.values["fraction_dscb"],
    "mean":m_tot.values['mean'],
    "sigma":m_tot.values['sigma'],
    "alphaL":m_tot.values['alphaL'],
    "nL":m_tot.values['nL'],
    "alphaR":m_tot.values['alphaR'],
    "nR":m_tot.values['nR'],
    #"fraction_gaussian":m_tot.values['fraction_gaussian'],
    "sigmaG":m_tot.values['sigmaG'],
    #"p1":m_tot.values['p1']
}
if key == 2:
    params["c"] = m_tot.values["c"]
elif key ==3:
    params["c"] = m_tot.values["c"]
    params["d"] = m_tot.values["d"]
m_tot_2 = Minuit(least_squares,
               **params
               )

m_tot_2.limits['sigma'] = (m_tot.values["sigma"]/2, m_tot.values["sigma"]*2)
#m_tot_2.fixed['normSig'] = True
m_tot_2.limits['sigmaG'] = (m_tot.values["sigmaG"]/2, m_tot.values["sigmaG"]*2)
m_tot_2.limits['normSig'] = (m_tot.values["normSig"]/2, 2*m_tot.values["normSig"])
#m_tot_2.limits['normSig'] = (m_tot.values["normSig"]/2, 2*m_tot.values["normSig"])
#m_tot_2.fixed['B'] = True
#m_tot_2.fixed['b'] = True
#m_tot_2.fixed['c'] = True
m_tot_2.fixed["fraction_dscb"]=True
m_tot_2.fixed['mean'] = True
m_tot_2.fixed['alphaR'] = True
m_tot_2.fixed['nR'] = True
m_tot_2.fixed['alphaL'] = True
m_tot_2.fixed['nL'] = True
m_tot_2.fixed['sigma'] = True
m_tot_2.fixed['sigmaG'] = True
#m_tot_2.fixed["fraction_gaussian"]=(m_tot.values["fraction_gaussian"]/2, 1)
#m_tot_2.fixed['p1'] = True



m_tot_2.migrad()
m_tot_2.hesse()

p_tot = m_tot_2.values

# Generate fit curves

x_plot = np.linspace(bins[0], bins[-1], 500)
y_tot = myBkgSignalFunctions[key](x_plot, *p_tot)



ax[0].plot(x_plot, y_tot, label="Fit (Background + Z Peak)", color='red')
ax[0].fill_between(x, 0, max(c)*1.2, where=maskFit, color='green', alpha=0.2)
ax[0].set_xlim(bins[0], bins[-1])
ax[0].set_ylim(1, max(c)*1.2)

ax[1].errorbar(x[maskFit], (c-myBkgFunctions[key](x, *p_tot[myBkgParams[key]]))[maskFit], yerr=np.sqrt(c)[maskFit], fmt='o', color='black', markersize=3)
ax[1].plot(x, zPeak(x, *p_tot[[ 'normSig', 'fraction_dscb', 'mean', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR', 'sigmaG']]), color='red', linewidth=2)
ax[1].set_ylabel("Data - Background")
ax[1].set_ylim(ax[1].get_ylim())
ax[1].fill_between(x, ax[1].get_ylim()[0], ax[1].get_ylim()[1], where=maskFit, color='green', alpha=0.2)

ax[1].hist(bins[:-1], bins=bins, weights = cZ)[0]
ax[1].hist(bins[:-1], bins=bins, weights = cHiggs, bottom=cZ)[0]

hep.cms.label(lumi="%.2f" %lumi_tot, ax=ax[0])
ax[1].set_xlabel("Dijet Mass [GeV]")
ax[0].set_ylabel("Counts")
ax[0].legend(bbox_to_anchor=(1, 1))

chi2_stat = np.sum(((c[maskFit] - myBkgSignalFunctions[key](x, *p_tot)[maskFit])**2) / np.sqrt(c)[maskFit]**2)
ndof = m_tot_2.ndof
chi2_pvalue = 1- chi2.cdf(chi2_stat, ndof)
ax[0].text(x=0.05, y=0.75, s="$\chi^2$/ndof = %.1f/%d\np-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax[0].transAxes, ha='left')
ax[0].set_yscale('log')

fig.savefig("/t3home/gcelotto/newFit/afterNN/cat2/cat2p1/plots/bkgPlusZPeak_160toInf.png", bbox_inches='tight')
# %%














# Plot 2
fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(14, 14), constrained_layout=True)
fig.align_ylabels([ax[0],ax[1]])
x_plot = np.linspace(bins[0], bins[-1], 500)
y_tot = myBkgSignalFunctions[key](x_plot, *p_tot)
ax[0].errorbar(x[maskUnblind], c[maskUnblind], yerr=np.sqrt(c)[maskUnblind], fmt='o', color='black', markersize=3, label="Data")
ax[0].set_ylim(ax[0].get_ylim())
ax[0].plot(x_plot, y_tot, label="Fit Sidebands", color='red')
ax[0].fill_between(x, 0, max(c)*1.2, where=maskFit, color='green', alpha=0.2, label='Fit Region')
ax[0].set_xlim(bins[0], bins[-1])
ax[0].text(x=0.95, y=0.7, s="Dijet p$_T$ > %d GeV\nJet1 bTag > %.2f\nJet2 bTag > %.2f\nNN score > %.1f\n"%(ptCut_min, jet1_btagMin, jet2_btagMin, PNN_t), transform=ax[0].transAxes, ha='right', va='top', fontsize=24)
ax[1].errorbar(x[maskFit], (c-myBkgFunctions[key](x, *p_tot[myBkgParams[key]]))[maskFit], yerr=np.sqrt(c)[maskFit], fmt='o', color='black', markersize=3)
ax[1].plot(x, zPeak(x, *p_tot[[ 'normSig', 'fraction_dscb', 'mean', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR', 'sigmaG']]), color='red', linewidth=2)
ax[1].hist(bins[:-1], bins=bins, weights=cZ, label='Z')
ax[1].set_ylabel("Data - Background")
#ax[1].set_ylim(ax[1].get_ylim())
cHiggs = np.zeros(len(bins)-1)
cHiggs_10 = np.zeros(len(bins)-1)
visFactor = 10
for idx, dfMC in enumerate(dfsMC_H):
    c_=np.histogram(dfMC.dijet_mass, bins=bins, weights = dfMC.weight)[0]
    label = dfProcessesMC.iloc[MCList_H].process.iloc[idx] + " x %d"%visFactor if visFactor != 1 else dfProcessesMC.iloc[MCList_H].process.iloc[idx]
    c10_ = ax[1].hist(dfMC.dijet_mass, bins=bins, weights = dfMC.weight*visFactor , label=label, bottom=cHiggs_10)[0]
    cHiggs += c_
    cHiggs_10 += c10_
ax[1].set_ylim(ax[1].get_ylim()[0], ax[1].get_ylim()[1]*1.5)
ax[1].legend(ncols=3)
ax[1].fill_between(x, ax[1].get_ylim()[0], ax[1].get_ylim()[1], where=maskFit, color='green', alpha=0.2)

hep.cms.label(lumi="%.2f" %lumi_tot, ax=ax[0])
ax[1].set_xlabel("Dijet Mass [GeV]")
ax[0].set_ylabel("Counts")
ax[0].set_ylim(0, ax[0].get_ylim()[1])
ax[0].legend(fontsize=24)
chi2_stat = np.sum(((c[maskFit] - myBkgSignalFunctions[key](x, *p_tot)[maskFit])**2) / np.sqrt(c)[maskFit]**2)
ndof = m_tot_2.ndof
chi2_pvalue = 1- chi2.cdf(chi2_stat, ndof)
ax[0].text(x=0.95, y=0.95, s="Fit Sidebands\n$\chi^2$/ndof = %.1f/%d\np-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax[0].transAxes, ha='right', va='top', fontsize=24)
#ax[0].set_yscale('log')
ax[0].tick_params(labelsize=24)
ax[1].tick_params(labelsize=24)
fig.savefig("/t3home/gcelotto/newFit/afterNN/cat2/cat2p1/plots/bkgPlusZPeak2_160toInf.png", bbox_inches='tight')
# End Plot 2
# %%




# Create Hist now
# x is bin center
# y_data is the values for data extracted from the fit
# cHiggs are the counts of Higgs
from hist import Hist
y_data_fit = Hist.new.Var(bins, name="mjj").Weight()
y_Higgs = Hist.new.Var(bins, name="mjj").Weight()
y_Z = Hist.new.Var(bins, name="mjj").Weight()
y_data_blind = Hist.new.Var(bins, name="mjj").Weight()
# %%
y_data_fit.values()[:] = myBkgFunctions[key](x, *p_tot[myBkgParams[key]])
y_data_fit.variances()[:] = myBkgFunctions[key](x, *p_tot[myBkgParams[key]])*0.01
y_Higgs.values()[:] = cHiggs
y_Higgs.variances()[:] = cHiggs*0.001

y_data_blind.values()[:] = c*maskFit
y_data_blind.variances()[:] = c*maskFit

y_Z.values()[:] = cZ
print("Creating ROOT histograms")
createRootHists([y_data_fit, y_Higgs, y_Z, y_data_blind], ['bkg', 'H', 'Z', 'data_obs'], bins=bins, outFolder="/t3home/gcelotto/newFit/afterNN/cat2/cat2p1/hists", suffix='cat2p1')
# %%
sig = np.sum(cHiggs[(x>100) & (x<150)])
bkg = np.sum(y_data_fit.values()[(x>100) & (x<150)])
print("Signal : %.1f"%sig)
print("Bkg : %d"%bkg)
print("Significance ", sig/np.sqrt(bkg)*np.sqrt(41.6/lumi_tot))

# %%
