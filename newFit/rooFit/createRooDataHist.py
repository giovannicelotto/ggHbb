# %%
import numpy as np
import pandas as pd
from functions import cut, getDfProcesses_v2
import mplhep as hep
hep.style.use("CMS")
import sys
import yaml
sys.path.append("/t3home/gcelotto/ggHbb/newFit/afterNN/")
from helpers.plotFree import plotFree
from helpers.defineFunctions import defineFunctions
from helpers.allFunctions import *
from hist import Hist
import ROOT
from array import array
import argparse
# %%

#parser = argparse.ArgumentParser()
#parser.add_argument('-i', '--idx', type=int, default=2, help='Index value (default: 2)')
#parser.add_argument('-w', '--write', type=int, default=1, help='Write Root File (1) or Not (0)')
#
#args = parser.parse_args()
idx = 2#args.idx

# Open config file and extract values
config_path = ["/t3home/gcelotto/ggHbb/newFit/afterNN/cat1/bkgPlusZFit_config.yml",
               "/t3home/gcelotto/ggHbb/newFit/afterNN/cat2/cat2p0/bkgPlusZFit_config.yml",
               "/t3home/gcelotto/ggHbb/newFit/afterNN/cat2/cat2p1/bkgPlusZFit_config.yml"][idx]
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
MCList_Z_sD = config["MCList_Z_sD"]
MCList_H_sD = config["MCList_H_sD"]
MCList_Z_sU = config["MCList_Z_sU"]
MCList_H_sU = config["MCList_H_sU"]
params = config["Bkg_params"]
paramsLimits = config["Bkg_paramsLimits"]
output_file = config["output_file"]
fitZSystematics = config["fitZSystematics"]
fitHSystematics = config["fitHSystematics"]
PNN_t_max=config["PNN_t_max"]
set_x_bounds(x1, x2)


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
dfsData = cut(dfsData, 'jet1_btagDeepFlavB', jet1_btagMin, None)
dfsMC_Z = cut(dfsMC_Z, 'jet1_btagDeepFlavB', jet1_btagMin, None)
dfsMC_H = cut(dfsMC_H, 'jet1_btagDeepFlavB', jet1_btagMin, None)
dfsData = cut(dfsData, 'jet2_btagDeepFlavB', jet2_btagMin, None)
dfsMC_Z = cut(dfsMC_Z, 'jet2_btagDeepFlavB', jet2_btagMin, None)
dfsMC_H = cut(dfsMC_H, 'jet2_btagDeepFlavB', jet2_btagMin, None)

# Add label process in dfsMC to distinguish VBF and ggF contribution
for idx, df in enumerate(dfsMC_H):
    dfsMC_H[idx]['process'] = dfProcessesMC.iloc[MCList_H].iloc[idx].process
for idx, df in enumerate(dfsMC_Z):
    dfsMC_Z[idx]['process'] = dfProcessesMC.iloc[MCList_Z].iloc[idx].process

# Concatenate all the subprocesses
dfMC_Z = pd.concat(dfsMC_Z)
dfMC_H = pd.concat(dfsMC_H)
df = pd.concat(dfsData)

# %%

# Create histogram
hist = ROOT.TH1F("hist", "hist", nbins, x1, x2)
hist_Z = ROOT.TH1F("hist_Z", "hist_Z", nbins, x1, x2)
data = array('d', df.dijet_mass)
Z_MC = array('d', dfMC_Z.dijet_mass)



# Fast fill: FillN(n, values, weights)
weights = array('d', [1.0] * len(data))
weights_Z = array('d', dfMC_Z.weight*1.56)

hist.FillN(len(data), data, weights)
hist_Z.FillN(len(Z_MC), Z_MC, weights_Z)

x = ROOT.RooRealVar("dijet_mass", "dijet_mass", 50,200)
data_hist = ROOT.RooDataHist("data_hist", "binned data", ROOT.RooArgList(x), hist)
Z_hist = ROOT.RooDataHist("Z_hist", "binned Z", ROOT.RooArgList(x), hist_Z)
# %%
# --- Composite Analytic Model ---

# --- Bernstein polynomial B1(x) ---
b0 = ROOT.RooRealVar("b0", "b0", 3.49354e-01, 0, 1)
b1 = ROOT.RooRealVar("b1", "b1", 2.5733e-01, 0, 1)
b2 = ROOT.RooRealVar("b2", "b2", 1.6782e-01, 0, 1)
b3 = ROOT.RooRealVar("b3", "b3", 1.3139e-01, 0, 1)
bkg_QCD = ROOT.RooBernstein("bkg_QCD", "bkg_QCD", x, ROOT.RooArgList(b0, b1, b2, b3))

# --- DoubleSided CB
# Define parameters with initial values and ranges
fraction_Z = ROOT.RooRealVar("fraction_Z", "fraction_Z", 0.001, 0, 1.)
mean = ROOT.RooRealVar("mean", "mean", 90, 85, 100)
width = ROOT.RooRealVar("width", "width", 10, 0.001, 15)
alpha1 = ROOT.RooRealVar("alpha1", "alpha1", 1.5, 0.01, 5.0)
n1 = ROOT.RooRealVar("n1", "n1", 5, 0.01, 50)
alpha2 = ROOT.RooRealVar("alpha2", "alpha2", 2.0, 0.01, 5.0)
n2 = ROOT.RooRealVar("n2", "n2", 5, 0.01, 50)

# Define the double-sided Crystal Ball formula as RooGenericPdf expression
dscb_formula = """
(
  ( ((@0-@1)/@2 >= -@3) * ((@0-@1)/@2 <= @5) * exp(-0.5 * pow(((@0-@1)/@2), 2)) )
+ ( ((@0-@1)/@2 < -@3) *
    pow(@4/abs(@3), @4) * exp(-0.5 * @3*@3) /
    pow(@4/abs(@3) - ((@0-@1)/@2 + @3), @4) )
+ ( ((@0-@1)/@2 > @5) *
    pow(@6/abs(@5), @6) * exp(-0.5 * @5*@5) /
    pow(@6/abs(@5) - ((@0-@1)/@2 - @5), @6) )
)

"""


dscb = ROOT.RooGenericPdf("dscb", "Double-sided Crystal Ball PDF", dscb_formula,
                         ROOT.RooArgList(x, mean, width, alpha1, n1, alpha2, n2))
# --- Composite PDF: B1(x) * exp(-B2(x)) ---
model_SB = ROOT.RooAddPdf("model_SB", "Signal + Background",
                       ROOT.RooArgList(dscb, bkg_QCD),
                       ROOT.RooArgList(fraction_Z))
model_SB.fixCoefNormalization(ROOT.RooArgSet(x))
# %%

# --- Double-sided Crystal Ball + Gaussian (shared mean) ---

#mean = ROOT.RooRealVar("mean", "mean", 90, 88, 95)
#sigma_gauss = ROOT.RooRealVar("sigma_gauss", "sigma_gauss", 10, 5, 20)
#sigma_gauss_2 = ROOT.RooRealVar("sigma_gauss_2", "sigma_gauss_2", 10, 5, 20)
#gauss = ROOT.RooGaussian("gauss", "Gaussian", x, mean, sigma_gauss)
#gauss_2 = ROOT.RooGaussian("gauss_2", "Gaussian", x, mean, sigma_gauss_2)
#f_gauss1 = ROOT.RooRealVar("f_gauss1", "gauss1 fraction", 0.5, 0.0, 1)
#gaussian_conv = ROOT.RooAddPdf("gaussian_conv", "gaussian_conv",
#                       ROOT.RooArgList(gauss, gauss_2),
#                       ROOT.RooArgList(f_gauss1))
#sigma_cb = ROOT.RooRealVar("sigma_cb", "sigma_cb", 20, 1, 100)
#alpha1 = ROOT.RooRealVar("alpha1", "alpha1", 1.5, 0.01, 5)
#n1 = ROOT.RooRealVar("n1", "n1", 5, 0.01, 50)
#alpha2 = ROOT.RooRealVar("alpha2", "alpha2", 2.0, 0.01, 5)
#n2 = ROOT.RooRealVar("n2", "n2", 5, 0.01, 50)


#ROOT.gSystem.CompileMacro("DoubleCB.cxx", "kO")
#cb = ROOT.DoubleCB("cb", "Double Crystal Ball", x, mean, sigma_cb, alpha1, n1, alpha2, n2)



#f_cb = ROOT.RooRealVar("f_cb", "Fraction CB", 0.5, 0, 1)
#cb_plus_gauss = ROOT.RooAddPdf("cb_plus_gauss", "CB + Gauss", ROOT.RooArgList(cb, gauss), ROOT.RooArgList(f_cb))

#f_signal = ROOT.RooRealVar("f_signal", "Signal fraction", 0.01, 0.0, .1)
#f_signal.setConstant(True)
#model_SB = ROOT.RooAddPdf("model_SB", "Signal + Background",
#                       ROOT.RooArgList(gaussian_conv, composite_shape),
#                       ROOT.RooArgList(f_signal))
# %%
# --- Workspace ---

w = ROOT.RooWorkspace("w", "workspace")
f_out = ROOT.TFile("/t3home/gcelotto/ggHbb/newFit/afterNN/workspace_sig.root", "RECREATE")
w_sig = ROOT.RooWorkspace("workspace_sig","workspace_sig")
#
### Import variables and models
w_import = getattr(w_sig, "import")

# Then safely import one by one
for obj in [x, data_hist, Z_hist, bkg_QCD, model_SB]:
    name = obj.GetName()

    if isinstance(obj, ROOT.RooAbsData):
        if not w_sig.data(name):
            getattr(w_sig, "import")(obj)
        else:
            print(f"Data {name} already in workspace. Skipping.")
    else:
        if not w_sig.arg(name):
            getattr(w_sig, "import")(obj, ROOT.RooFit.RecycleConflictNodes())
        else:
            print(f"Object {name} already in workspace. Skipping.")

##
w_sig.Print()
w_sig.Write()
f_out.Close()
## Write workspace to file



# %%
