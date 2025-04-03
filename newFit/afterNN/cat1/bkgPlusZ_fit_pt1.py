# %%
import numpy as np
import pandas as pd
from functions import cut, getDfProcesses_v2
import mplhep as hep
hep.style.use("CMS")
import json, sys
sys.path.append("/t3home/gcelotto/ggHbb/newFit/afterNN/")
from helpers.doFitParameterFixed import doFitParameterFixed
from helpers.doFitParameterFree import doFitParameterFree
from helpers.plotFree import plotFree
from helpers.defineFunctions import defineFunctions
from helpers.allFunctions import *
from hist import Hist
columns=['dijet_mass', 'dijet_pt']
from helpers.getBounds import getBounds
# %%
x1, x2, key, nbins, t1, t2, t0, t3 = getBounds(paramFile="/t3home/gcelotto/ggHbb/newFit/afterNN/cat1/config_1.yaml")

set_x_bounds(x1, x2)
myBkgFunctions, myBkgSignalFunctions, myBkgParams = defineFunctions()

plotFolder = "/t3home/gcelotto/ggHbb/newFit/afterNN/cat1/plots"
isDataList = [1,
              2,
              3,
              ]

modelName = "Mar21_1_0p0"
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

MCList_H = [43, 36]
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
ptCut_min = 100
ptCut_max = 160
jet1_btagMin =0.71
jet2_btagMin =0.71
PNN_t = 0.575
dfsData = cut(dfsData, 'PNN', PNN_t, None)
dfsMC_Z = cut(dfsMC_Z, 'PNN', PNN_t, None)
dfsMC_H = cut(dfsMC_H, 'PNN', PNN_t, None)
dfsData = cut(dfsData, 'dijet_pt', ptCut_min, ptCut_max)
dfsMC_Z = cut(dfsMC_Z, 'dijet_pt', ptCut_min, ptCut_max)
dfsMC_H = cut(dfsMC_H, 'dijet_pt', ptCut_min, ptCut_max)
dfsData = cut(dfsData, 'jet1_btagDeepFlavB', jet1_btagMin, None)
dfsMC_Z = cut(dfsMC_Z, 'jet1_btagDeepFlavB', jet1_btagMin, None)
dfsMC_H = cut(dfsMC_H, 'jet1_btagDeepFlavB', jet1_btagMin, None)
dfsData = cut(dfsData, 'jet2_btagDeepFlavB', jet2_btagMin, None)
dfsMC_Z = cut(dfsMC_Z, 'jet2_btagDeepFlavB', jet2_btagMin, None)
dfsMC_H = cut(dfsMC_H, 'jet2_btagDeepFlavB', jet2_btagMin, None)

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
# Blind a region for the fit
variations = ['nominal', 'jet1_btag_up', 'jet1_btag_down']

variation_map = {
    'nominal': lambda df: df['weight'],
    'jet1_btag_up': lambda df: df['weight'] * df['jet1_btag_up'] / df['jet1_btag_central'],
    'jet1_btag_down': lambda df: df['weight'] * df['jet1_btag_down'] / df['jet1_btag_central'],
}
def apply_variation(dfs, variation):
    for idx, df in enumerate(dfs):
        dfs[idx]['weight_var'] = variation_map[variation](df)
    return dfs

hists={
    'Fit':{},
    'Higgs':{},
    'Z':{},
    'data_obs':{},
}
for var in variations:
    print("variation : ", var)
    dfsMC_Z = apply_variation(dfsMC_Z, var)
    dfsMC_H = apply_variation(dfsMC_H, var)

    dfMC_Z = pd.concat(dfsMC_Z)
    dfMC_H = pd.concat(dfsMC_H)
    
    with open("/t3home/gcelotto/ggHbb/newFit/afterNN/cat1/fit_parameters_with_systematics.json", "r") as f:
        fit_parameters_zPeak = json.load(f)
    
    params={
        "B" : 24.74e-3   ,
        'C' : 0.5047  ,
        'b' : -14.03e-3	,
        'c' : 117.2e-6  ,
           }
    paramsLimits={
        "B" : (0.0, 0.05),
        "C" : (0.1, 1)}
    x_fit, y_tofit, yerr, m_tot, cZ, cZ_err = doFitParameterFixed(x, x1, x2, c, maskFit, maskUnblind, bins, dfsMC_Z, dfProcessesMC, MCList_Z, dfsMC_H, MCList_H, myBkgSignalFunctions, key, dfMC_Z, myBkgFunctions, lumi_tot, myBkgParams, plotFolder, fit_parameters_zPeak, var, params, paramsLimits)
    # Second Fitx parameters free

    m_tot_2 = doFitParameterFree(x, m_tot, c, cZ, maskFit, maskUnblind, bins, dfsMC_Z, dfProcessesMC, MCList_Z, dfsMC_H, MCList_H, myBkgSignalFunctions, key, myBkgFunctions, lumi_tot, myBkgParams, plotFolder, fitFunction=fit_parameters_zPeak["fitFunction"])

    cHiggs, cHiggs_err = plotFree(x, c, cZ, maskFit, m_tot_2, maskUnblind, bins, dfProcessesMC, dfsMC_H, MCList_H, myBkgSignalFunctions, key, myBkgFunctions, lumi_tot, myBkgParams, plotFolder, ptCut_min, ptCut_max, jet1_btagMin, jet2_btagMin, PNN_t, PNN_t_max=None, fitFunction=fit_parameters_zPeak["fitFunction"], var=var)



# Plot 2
    p_tot = m_tot_2.values
# Create Hist now
    # x is bin center
    # y_data is the values for data extracted from the fit
    # cHiggs are the counts of Higgs

    y_data_fit = Hist.new.Var(bins, name="mjj").Weight()
    y_Higgs = Hist.new.Var(bins, name="mjj").Weight()
    y_Z = Hist.new.Var(bins, name="mjj").Weight()
    y_data_blind = Hist.new.Var(bins, name="mjj").Weight()

    y_data_fit.values()[:] = np.where(x>t0, myBkgFunctions[key](x, *p_tot[myBkgParams[key]]), 0)
    y_data_fit.variances()[:] = np.where(x>t0, myBkgFunctions[key](x, *p_tot[myBkgParams[key]]), 0)
    y_Higgs.values()[:] = np.where(x>t0, cHiggs, 0)
    y_Higgs.variances()[:] = np.where(x>t0, cHiggs_err**2, 0)

    y_data_blind.values()[:] = np.where(x>t0, c, 0)
    y_data_blind.variances()[:] = np.where(x>t0, c, 0)

    y_Z.values()[:] = np.where(x>t0, cZ, 0)
    y_Z.variances()[:] = np.where(x>t0, cZ_err**2, 0)
    
    var = var.replace("_up", "_Up")
    var = var.replace("_down", "_Down")
    hists['data_obs'][var]=y_data_blind
    hists['Higgs'][var]=y_Higgs
    hists['Z'][var]=y_Z
    hists['Fit'][var]=y_data_fit

print("Creating ROOT histograms")
outFolder="/t3home/gcelotto/ggHbb/newFit/afterNN/cat1/hists"
  
# %%
import ROOT
output_file = "/t3home/gcelotto/ggHbb/newFit/afterNN/cat1/hists/counts_cat1.root"
root_file = ROOT.TFile(output_file, "RECREATE")

# Loop through histograms and save them as TH1F
for process, variations in hists.items():
    for var, hist in variations.items():
        # Convert to numpy arrays
        values = hist.values()
        errors = np.sqrt(hist.variances())  # sqrt of variance gives errors
        bins = hist.axes[0].edges  # Get bin edges

        # Create TH1F histogram
        th1 = ROOT.TH1F(f"{process}_{var}", f"{process}_{var}", len(bins)-1, bins)
        
        for i in range(len(values)):
            th1.SetBinContent(i+1, values[i])
            th1.SetBinError(i+1, errors[i])

        th1.Write()

root_file.Close()

# %%
sig = np.sum(cHiggs[(x>100) & (x<150)])
bkg = np.sum(y_data_fit.values()[(x>100) & (x<150)])
print("Signal : %.1f"%sig)
print("Bkg : %d"%bkg)
print("Significance ", sig/np.sqrt(bkg)*np.sqrt(41.6/lumi_tot))
cHiggs.sum()
# %%
