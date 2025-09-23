# %%
import numpy as np
import pandas as pd
from functions import cut, getDfProcesses_v2
import mplhep as hep
hep.style.use("CMS")
import json, sys
import os
import yaml
sys.path.append("/t3home/gcelotto/ggHbb/newFit/afterNN/")
from helpers.doFitParameterFixed import doFitParameterFixed
from helpers.doFitParameterFree import doFitParameterFree
from helpers.plotFree import plotFree
from helpers.defineFunctions import defineFunctions
from helpers.allFunctions import *
from hist import Hist
import ROOT
from helpers.computeGradient import numerical_gradient
from helpers.getBounds import getBounds
import argparse
# %%

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--idx', type=int, default=2, help='Index value (default: 2)')
parser.add_argument('-w', '--write', type=int, default=1, help='Write Root File (1) or Not (0)')
parser.add_argument('-bl', '--blinded', type=int, default=1, help='blind or unblind the fit')
parser.add_argument('-l', '--lockNorm', type=int, default=0, help='Lock Normalization True (1) or False (0)')

args = parser.parse_args()
idx = args.idx

# Open config file and extract values
config_path = ["/t3home/gcelotto/ggHbb/newFit/afterNN/cat1/bkgPlusZFit_config.yml",
               "/t3home/gcelotto/ggHbb/newFit/afterNN/cat2/cat2p0/bkgPlusZFit_config.yml",
               "/t3home/gcelotto/ggHbb/newFit/afterNN/cat2/cat2p1/bkgPlusZFit_config.yml"][idx]
# Open config file and extract values
config_path_cuts = ["/t3home/gcelotto/ggHbb/newFit/afterNN/cat1/PeakFit_config.yml",
               "/t3home/gcelotto/ggHbb/newFit/afterNN/cat2/cat2p0/PeakFit_config.yml",
               "/t3home/gcelotto/ggHbb/newFit/afterNN/cat2/cat2p1/PeakFit_config.yml"][idx]
print("Opening config file ", config_path, "...")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
print(config)
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
plotFolder = config["plotFolder"]
MCList_Z = config["MCList_Z"]
MCList_H = config["MCList_H"]
MCList_Z_sD = config["MCList_Z_sD"]
MCList_H_sD = config["MCList_H_sD"]
MCList_Z_sU = config["MCList_Z_sU"]
MCList_H_sU = config["MCList_H_sU"]
params = config["Bkg_params"][key]
paramsLimits = config["Bkg_paramsLimits"]
output_file = config["output_file"]
fitZSystematics = config["fitZSystematics"]
fitHSystematics = config["fitHSystematics"]
set_x_bounds(x1, x2)

with open(config_path_cuts, 'r') as f2:
    config_cuts = yaml.safe_load(f2)
ptCut_min = config_cuts["cuts_dict"]["dijet_pt"][0]
ptCut_max = config_cuts["cuts_dict"]["dijet_pt"][1]
jet1_btagMin = config_cuts["cuts_dict"]["jet1_btagDeepFlavB"][0]
jet2_btagMin = config_cuts["cuts_dict"]["jet2_btagDeepFlavB"][0]

jet1_btagMax = config_cuts["cuts_dict"]["jet1_btagDeepFlavB"][1]
jet2_btagMax = config_cuts["cuts_dict"]["jet2_btagDeepFlavB"][1]
PNN_t, PNN_t_max = config_cuts["cuts_dict"]["PNN"]

# Call the functions for Bkg Bkg+Z and Bkg+Z+H
myBkgFunction, myBkgSignalFunction, myBkgParams, myBkgFunctionZ_H = defineFunctions(key)

# Call the dataframes with names, xsections, paths of processes
dfProcessesMC, dfProcessesData, dfProcessesMC_JEC = getDfProcesses_v2()
dfProcessesData = dfProcessesData.iloc[isDataList]


# %%
# Open Data
dfsData = []
lumi_tot = 0.
path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/mjjDisco/%s"%modelName
for processName in dfProcessesData.process.values:
    print("Opening ", processName)
    df = pd.read_parquet(path+"/dataframes_%s_%s.parquet"%(processName, modelName))
    dfsData.append(df)
    lumi_tot = lumi_tot + np.load(path+"/lumi_%s_%s.npy"%(processName, modelName))

# Open the MC

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
# Normalize the MC to the luminosity
for idx, df in enumerate(dfsMC_H):
    dfsMC_H[idx].weight=dfsMC_H[idx].weight*lumi_tot

for idx, df in enumerate(dfsMC_Z):
    dfsMC_Z[idx].weight=dfsMC_Z[idx].weight*lumi_tot


# %%
# Apply cuts
dfsData = cut(dfsData, 'PNN', PNN_t, PNN_t_max)
dfsMC_Z = cut(dfsMC_Z, 'PNN', PNN_t, PNN_t_max)
dfsMC_H = cut(dfsMC_H, 'PNN', PNN_t, PNN_t_max)
dfsData = cut(dfsData, 'dijet_pt', ptCut_min, ptCut_max)
dfsMC_Z = cut(dfsMC_Z, 'dijet_pt', ptCut_min, ptCut_max)
dfsMC_H = cut(dfsMC_H, 'dijet_pt', ptCut_min, ptCut_max)
dfsData = cut(dfsData, 'jet1_btagDeepFlavB', jet1_btagMin, jet1_btagMax)
dfsMC_Z = cut(dfsMC_Z, 'jet1_btagDeepFlavB', jet1_btagMin, jet1_btagMax)
dfsMC_H = cut(dfsMC_H, 'jet1_btagDeepFlavB', jet1_btagMin, jet1_btagMax)
dfsData = cut(dfsData, 'jet2_btagDeepFlavB', jet2_btagMin, jet2_btagMax)
dfsMC_Z = cut(dfsMC_Z, 'jet2_btagDeepFlavB', jet2_btagMin, jet2_btagMax)
dfsMC_H = cut(dfsMC_H, 'jet2_btagDeepFlavB', jet2_btagMin, jet2_btagMax)

# Add label process in dfsMC to distinguish VBF and ggF contribution
for idx, df in enumerate(dfsMC_H):
    dfsMC_H[idx]['process'] = dfProcessesMC.iloc[MCList_H].iloc[idx].process
for idx, df in enumerate(dfsMC_Z):
    dfsMC_Z[idx]['process'] = dfProcessesMC.iloc[MCList_Z].iloc[idx].process

# Concatenate all the subprocesses
dfMC_Z = pd.concat(dfsMC_Z)
dfMC_H = pd.concat(dfsMC_H)
df = pd.concat(dfsData)


# Histogram the data
bins = np.linspace(x1, x2, nbins)
x=(bins[1:]+bins[:-1])/2
c=np.histogram(df.dijet_mass, bins=bins)[0]


# Definition of masks for blinding
maskFit = ((x>t0)&(x < t1)) | ((x<t3)&(x > t2))
if args.blinded==0:
    print("This option")
    maskUnblind = (x < t1) | (x > t2)
elif args.blinded == 1:
    maskUnblind = ((x>t0) & (x < t1)) | ((x > t2) & (x < t3))
elif args.blinded == -1:
    maskUnblind = np.full(len(x), True)
print(np.sum(maskFit), " entries in the maskFit")
##
## variation happening
## 



# %%

variations = [
    'nominal',
    #'btag_up', 'btag_down',
    #'JER_Up', 'JER_Down',
    #'JECAbsoluteMPFBias_Up',
    #'JECAbsoluteMPFBias_Down','JECAbsoluteScale_Up', 'JECAbsoluteScale_Down','JECAbsoluteStat_Up', 'JECAbsoluteStat_Down','JECFlavorQCD_Up', 'JECFlavorQCD_Down','JECFragmentation_Up', 'JECFragmentation_Down','JECPileUpDataMC_Up', 'JECPileUpDataMC_Down',
    #'JECPileUpPtBB_Up', 'JECPileUpPtBB_Down',
    #'JECPileUpPtEC1_Up', 'JECPileUpPtEC1_Down','JECPileUpPtEC2_Up', 'JECPileUpPtEC2_Down','JECPileUpPtHF_Up', 'JECPileUpPtHF_Down','JECPileUpPtRef_Up', 'JECPileUpPtRef_Down','JECRelativeBal_Up', 'JECRelativeBal_Down','JECRelativeFSR_Up', 'JECRelativeFSR_Down','JECRelativeJEREC1_Up', 'JECRelativeJEREC1_Down','JECRelativeJEREC2_Up', 'JECRelativeJEREC2_Down','JECRelativeJERHF_Up', 'JECRelativeJERHF_Down',
    #'JECRelativePtBB_Up', 
    #'JECRelativePtBB_Down',
    #'JECRelativePtEC1_Up', 'JECRelativePtEC1_Down','JECRelativePtEC2_Up', 'JECRelativePtEC2_Down','JECRelativePtHF_Up', 'JECRelativePtHF_Down','JECRelativeSample_Up', 'JECRelativeSample_Down','JECRelativeStatEC_Up', 'JECRelativeStatEC_Down','JECRelativeStatFSR_Up', 'JECRelativeStatFSR_Down','JECRelativeStatHF_Up', 'JECRelativeStatHF_Down','JECSinglePionECAL_Up', 'JECSinglePionECAL_Down','JECSinglePionHCAL_Up', 'JECSinglePionHCAL_Down','JECTimePtEta_Up', 'JECTimePtEta_Down',
]

variation_map = {
    'nominal': lambda df: df['weight'],
    'btag_up': lambda df: df['weight'] * df['btag_up'] / df['btag_central'],
    'btag_down': lambda df: df['weight'] * df['btag_down'] / df['btag_central'],
}

def apply_variation(df, variation):
    df['weight_'] = variation_map[variation](df)
    return df

# Define histograms
hists={
    'Fit':{},
    'Higgs':{},
    'Z':{},
    'data_obs':{},
    'Fit_noZ':{},
}
# %%
for idx, var in enumerate(variations):
    print("\n\nvariation %d/%d: "%((idx+1), len(variations)), var, "\n\n")
    if var=='nominal':
        dfMC_Z['dijet_mass_'] = dfMC_Z['dijet_mass'].copy()
        dfMC_Z['weight_'] = dfMC_Z['weight'].copy()
        dfMC_H['dijet_mass_'] = dfMC_H['dijet_mass'].copy()
        dfMC_H['weight_'] = dfMC_H['weight'].copy()
    elif (var=='btag_up') | (var=='btag_down'):
        
        # Btag
        
        dfMC_Z = apply_variation(dfMC_Z, var)
        dfMC_H = apply_variation(dfMC_H, var)
        
        dfMC_Z['dijet_mass_'] = dfMC_Z.dijet_mass.copy()
        dfMC_H['dijet_mass_'] = dfMC_H.dijet_mass.copy()
    elif (var=="JER_Up") | (var=="JER_Down"):

        # JER

        smearDirection = None
        if "Up" in var:
            smearDirection='Up'
        if "Down" in var:
            smearDirection='Down'
        dfsMC_Z = []
        dfsMC_H = []
        if smearDirection=='Up':
            MCList_Z_s = MCList_Z_sU
            MCList_H_s = MCList_H_sU
        elif smearDirection=='Down':
            MCList_Z_s = MCList_Z_sD
            MCList_H_s = MCList_H_sD
        else:
            assert False

        for processName in dfProcessesMC.iloc[MCList_Z_s].process.values:
            #print("Opening (JER) ", processName)
            df = pd.read_parquet(path + "/df_%s_%s.parquet" % (processName, modelName), engine='pyarrow')
            dfsMC_Z.append(df)

        for processName in dfProcessesMC.iloc[MCList_H_s].process.values:
            #print("Opening (JER) ", processName)
            df = pd.read_parquet(path + "/df_%s_%s.parquet" % (processName, modelName), engine='pyarrow')
            dfsMC_H.append(df)

        # Apply same weight scaling
        for idx, df in enumerate(dfsMC_H):
            dfsMC_H[idx]['weight'] = df.weight * lumi_tot
        for idx, df in enumerate(dfsMC_Z):
            dfsMC_Z[idx]['weight'] = df.weight * lumi_tot

        # Apply same cuts
        dfsMC_Z = cut(dfsMC_Z, 'PNN', PNN_t, PNN_t_max)
        dfsMC_H = cut(dfsMC_H, 'PNN', PNN_t, PNN_t_max)
        dfsMC_Z = cut(dfsMC_Z, 'dijet_pt', ptCut_min, ptCut_max)
        dfsMC_H = cut(dfsMC_H, 'dijet_pt', ptCut_min, ptCut_max)
        dfsMC_Z = cut(dfsMC_Z, 'jet1_btagDeepFlavB', jet1_btagMin, None)
        dfsMC_H = cut(dfsMC_H, 'jet1_btagDeepFlavB', jet1_btagMin, None)
        dfsMC_Z = cut(dfsMC_Z, 'jet2_btagDeepFlavB', jet2_btagMin, None)
        dfsMC_H = cut(dfsMC_H, 'jet2_btagDeepFlavB', jet2_btagMin, None)


        for idx, df in enumerate(dfsMC_H):
            dfsMC_H[idx]['process'] = dfProcessesMC.iloc[MCList_H_s].iloc[idx].process
        for idx, df in enumerate(dfsMC_Z):
            dfsMC_Z[idx]['process'] = dfProcessesMC.iloc[MCList_Z_s].iloc[idx].process


        dfMC_Z = pd.concat(dfsMC_Z)
        dfMC_H = pd.concat(dfsMC_H)

        dfMC_Z['dijet_mass_'] = dfMC_Z.dijet_mass.copy()
        dfMC_H['dijet_mass_'] = dfMC_H.dijet_mass.copy()
        dfMC_Z['weight_'] = dfMC_Z.weight
        dfMC_H['weight_'] = dfMC_H.weight
    elif "JEC" in var:

        # JEC

        varDirection = None
        if var.endswith('_Up'):
            jec_syst, varDirection = var[:-3], 'Up'
        elif var.endswith('_Down'):
            jec_syst, varDirection = var[:-5], 'Down'
        print("JEC systematic : %s \nDirection %s"%(jec_syst, varDirection))
        
        dfsMC_Z = []
        dfsMC_H = []
        for processName in dfProcessesMC.process.iloc[MCList_Z].values:
            #print("Opening (JEC) ", processName)
            df = pd.read_parquet(path + "/df_%s_%s_%s.parquet" % (processName,var, modelName), engine='pyarrow')
            dfsMC_Z.append(df)

        for processName in dfProcessesMC.iloc[MCList_H].process.values:
            #if processName=="GluGluHToBB_tr":
            #    processName="GluGluHToBB"
            pathExistence = os.path.exists(path + "/df_%s_%s_%s.parquet" % (processName,var,modelName))
            df = pd.read_parquet(path + "/df_%s_%s_%s.parquet" % (processName,var,modelName), engine='pyarrow')
            dfsMC_H.append(df)

        # Apply same weight scaling
        for idx, df in enumerate(dfsMC_H):
            dfsMC_H[idx]['weight'] = df.weight * lumi_tot
        for idx, df in enumerate(dfsMC_Z):
            dfsMC_Z[idx]['weight'] = df.weight * lumi_tot

        # Apply same cuts
        dfsMC_Z = cut(dfsMC_Z, 'PNN', PNN_t, PNN_t_max)
        dfsMC_H = cut(dfsMC_H, 'PNN', PNN_t, PNN_t_max)
        dfsMC_Z = cut(dfsMC_Z, 'dijet_pt', ptCut_min, ptCut_max)
        dfsMC_H = cut(dfsMC_H, 'dijet_pt', ptCut_min, ptCut_max)
        dfsMC_Z = cut(dfsMC_Z, 'jet1_btagDeepFlavB', jet1_btagMin, None)
        dfsMC_H = cut(dfsMC_H, 'jet1_btagDeepFlavB', jet1_btagMin, None)
        dfsMC_Z = cut(dfsMC_Z, 'jet2_btagDeepFlavB', jet2_btagMin, None)
        dfsMC_H = cut(dfsMC_H, 'jet2_btagDeepFlavB', jet2_btagMin, None)


        for idx, df in enumerate(dfsMC_H):
            if processName=="GluGluHToBB_tr":
                continue
            dfsMC_H[idx]['process'] = dfProcessesMC.iloc[MCList_H].iloc[idx].process
        for idx, df in enumerate(dfsMC_Z):
            dfsMC_Z[idx]['process'] = dfProcessesMC.iloc[MCList_Z].iloc[idx].process


        dfMC_Z = pd.concat(dfsMC_Z)
        dfMC_H = pd.concat(dfsMC_H)

        dfMC_Z['dijet_mass_'] = dfMC_Z.dijet_mass.copy()
        dfMC_H['dijet_mass_'] = dfMC_H.dijet_mass.copy()
        dfMC_Z['weight_'] = dfMC_Z.weight
        dfMC_H['weight_'] = dfMC_H.weight
    
    
    with open(fitZSystematics, "r") as f:
        fit_parameters_zPeak = json.load(f)
    with open(fitHSystematics, "r") as f:
        fit_parameters_hPeak = json.load(f)
    print("Fit all constrained", key)
    x_fit, y_tofit, yerr, m_tot, cZ, cZ_err = doFitParameterFixed(x, x1, x2, c, maskFit, maskUnblind, bins, myBkgFunctionZ_H, key, dfMC_Z, dfMC_H, myBkgFunction, lumi_tot, myBkgParams, plotFolder, fit_parameters_zPeak, fit_parameters_hPeak, var, params, paramsLimits)
    # Second Fit parameters free

    print("Fit Free")
    m_tot_2 = doFitParameterFree(x, m_tot, c, cZ, 
                                 maskFit, maskUnblind, 
                                 bins, dfMC_Z, dfMC_H, 
                                 myBkgSignalFunction, myBkgFunction, lumi_tot,
                                 myBkgParams,
                                 fitFunction_Z=fit_parameters_zPeak["fitFunction"],  fit_parameters_hPeak=fit_parameters_hPeak,
                                 var=var,
                                 paramsLimits=paramsLimits,
                                 lockNorm=args.lockNorm,
                                 blind=args.blinded,
                                 myBkgFunctionZ_H=myBkgFunctionZ_H)

    print("Plot only")
    print(((config_cuts["cuts_dict"]["dijet_pt"][0]!=None) & (config_cuts["cuts_dict"]["dijet_pt"][1] != None)))
    cHiggs, cHiggs_err = plotFree(x, c, cZ, maskFit, m_tot_2, maskUnblind, bins, dfProcessesMC, dfMC_H, MCList_H, myBkgSignalFunction, key, myBkgFunction, lumi_tot, myBkgParams, plotFolder, config_cuts, ptCut_min, ptCut_max, jet1_btagMin, jet2_btagMin, PNN_t, PNN_t_max=PNN_t_max, fitFunction=fit_parameters_zPeak["fitFunction"], var=var,
                                    blind=args.blinded,
                                    myBkgFunctionZ_H=myBkgFunctionZ_H)


# Plot 2
    p_tot = m_tot_2.values
    print(p_tot)
    cov = m_tot_2.covariance
    if cov is None:
        print(m_tot_2)
        
        print("WARNING: Covariance matrix is None. Using identity.")
        cov = np.eye(mygrad)
    #print(cov)
    # here compute gradient
    #print("Cov diag",  np.diag(cov))
    mygrad = numerical_gradient(myBkgFunctionZ_H, x=x, params=p_tot, eps=1e-9)
    #print("mygrad", mygrad.shape)


    from jacobi import propagate
    def model_wrapper(p):
        return myBkgFunctionZ_H(x, *p)
    y, ycov = propagate(lambda p: model_wrapper(p), m_tot_2.values, m_tot_2.covariance)
    yerr_prop = np.diag(ycov) ** 0.5
    #print(f"cov shape: {None if cov is None else cov.shape}")
    #print(p_tot)
    #print(cov)
    #print(mygrad[:,0] )
    f_var = np.array([
    mygrad[:, i].T @ cov @ mygrad[:, i]
    for i in range(len(x))
    ])
    f_err = np.sqrt(f_var)
    print("These are my yerr_prop / error of iminuit", np.mean(f_err/yerr_prop))
    #print(f_err)
    #print(np.sqrt(myBkgSignalFunctions[key](x, *p_tot)))



# Create Hist now
    # x is bin center
    # y_data is the values for data extracted from the fit
    # cHiggs are the counts of Higgs
    y_data_fit_noZ = Hist.new.Var(bins, name="mjj").Weight()
    y_data_fit = Hist.new.Var(bins, name="mjj").Weight()
    y_Higgs = Hist.new.Var(bins, name="mjj").Weight()
    y_Z = Hist.new.Var(bins, name="mjj").Weight()
    y_data_blind = Hist.new.Var(bins, name="mjj").Weight()

    y_data_fit_noZ.values()[:] = np.where((x>t0) & (x<t3), myBkgFunction(x, *p_tot[myBkgParams]), 0)
    y_data_fit_noZ.variances()[:] = np.where((x>t0) & (x<t3), c, 0)
    

    y_data_fit.values()[:] = np.where((x>t0) & (x<t3), myBkgSignalFunction(x, *p_tot[myBkgParams+["normSig", "fraction_dscb", "mean", "sigma", "alphaL", "nL", "alphaR", "nR", "sigmaG",]]), 0)
    y_data_fit.variances()[:] = np.where((x>t0) & (x<t3), c, 0)
    
    y_Higgs.values()[:] = np.where((x>t0) & (x<t3), cHiggs, 0)
    y_Higgs.variances()[:] = np.where((x>t0) & (x<t3), cHiggs_err**2, 0)

    y_data_blind.values()[:] = np.where((x>t0) & (x<t3), c, 0)
    y_data_blind.variances()[:] = np.where((x>t0) & (x<t3), c, 0)

    y_Z.values()[:] = np.where((x>t0) & (x<t3), cZ, 0)
    y_Z.variances()[:] = np.where((x>t0) & (x<t3), cZ_err**2, 0)
    
    var = var.replace("_up", "_Up")
    var = var.replace("_down", "_Down")
    hists['data_obs'][var]=y_data_blind
    hists['Higgs'][var]=y_Higgs
    hists['Z'][var]=y_Z
    hists['Fit'][var]=y_data_fit
    hists['Fit_noZ'][var]=y_data_fit_noZ


print("Creating ROOT histograms")
  
# %%



df = pd.DataFrame({
    'x':x,
    'y':np.where((x>x1) & (x<x2), myBkgSignalFunction(x, *p_tot[myBkgParams+["normSig", "fraction_dscb", "mean", "sigma", "alphaL", "nL", "alphaR", "nR", "sigmaG",]]), 0),
    'yerr':np.where((x>x1) & (x<x2), np.sqrt(f_var), 0)
})

df.to_parquet("/t3home/gcelotto/ggHbb/newFit/afterNN/cat2/cat2p1/df_%d.parquet"%key)




if args.write:
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



sig = np.sum(cHiggs[(x>110) & (x<140)])
bkgUnc = np.sqrt(np.sum((y_data_fit.variances()[(x>110) & (x<140)])))
print("Signal : %.1f"%sig, cHiggs[(x>110) & (x<140)])
print("BkgUnc : %d"%bkgUnc)
print("Significance ", sig/np.sqrt(bkgUnc**2 + sig))
print(c[0])
print(y_data_fit.values()[0])

# %%
