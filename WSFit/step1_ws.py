# Need to run non Interactively!

import ROOT
ROOT.gSystem.CompileMacro("/t3home/gcelotto/ggHbb/newFit/rooFit/helpersFunctions/RooDoubleCB.cc", "kf")
import numpy as np
import pandas as pd
from functions import cut, getDfProcesses_v2
import mplhep as hep
hep.style.use("CMS")
import sys
import yaml
sys.path.append("/t3home/gcelotto/ggHbb/newFit/afterNN/")
from array import array
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default="2", help='Config File')
if hasattr(sys, 'ps1') or not sys.argv[1:]:
    # Interactive mode (REPL, Jupyter) OR no args provided → use defaults
    args = parser.parse_args([])
else:
    # Normal CLI usage
    args = parser.parse_args()
#from helpers.plotFree import plotFree
#from helpers.defineFunctions import defineFunctions
#from helpers.allFunctions import *
#from hist import Hist
#"k" stands for keep the compiled shared library after the script exits.
#import argparse

sys.path.append("/t3home/gcelotto/ggHbb/newFit/rooFit/")
from getDfs import getDfs
# %%

dfMC_Z, dfMC_H, df, nbins, x1, x2 = getDfs(idx=int(args.config))
print("[INFO] Limits on RooRealVar")
print(x1, x2, nbins)

config_path_Z = "/t3home/gcelotto/ggHbb/WSFit/output/cat%d/fit_parameters_with_systematics_Z.json"%(int(args.config))
config_path_H = "/t3home/gcelotto/ggHbb/WSFit/output/cat%d/fit_parameters_with_systematics_H.json"%(int(args.config))


# %%
# Category 2
# Create histogram
hist_data_cat = ROOT.TH1F("hist_data_cat%d"%(int(args.config)), "hist_data_cat%d"%(int(args.config)), nbins-1, x1, x2)
hist_Z_cat = ROOT.TH1F("hist_Z_cat%d"%(int(args.config)), "hist_Z_cat%d"%(int(args.config)), nbins-1, x1, x2)
hist_H_cat = ROOT.TH1F("hist_H_cat%d"%(int(args.config)), "hist_H_cat%d"%(int(args.config)), nbins-1, x1, x2)
data = array('d', df.dijet_mass)
Z_MC = array('d', dfMC_Z.dijet_mass)
H_MC = array('d', dfMC_H.dijet_mass)



# Fast fill: FillN(n, values, weights)
weights = array('d', [1.0] * len(data))
weights_Z = array('d', dfMC_Z.weight)
weights_H = array('d', dfMC_H.weight)

hist_data_cat.FillN(len(data), data, weights)
hist_Z_cat.FillN(len(Z_MC), Z_MC, weights_Z)
hist_H_cat.FillN(len(H_MC), H_MC, weights_H)



# To be saved
x_cat = ROOT.RooRealVar("dijet_mass_c%d"%(int(args.config)), "dijet_mass_c%d"%(int(args.config)), x1,x2)

rooHist_data_cat = ROOT.RooDataHist("rooHist_data_cat%d"%(int(args.config)), "rooHist_data_cat%d"%(int(args.config)), ROOT.RooArgList(x_cat), hist_data_cat)
rooHist_Z_cat = ROOT.RooDataHist("rooHist_Z_cat%d"%(int(args.config)), "binned Z", ROOT.RooArgList(x_cat), hist_Z_cat)
rooHist_H_cat = ROOT.RooDataHist("rooHist_H_cat%d"%(int(args.config)), "binned H", ROOT.RooArgList(x_cat), hist_H_cat)
# --- DoubleSided CB

with open(config_path_Z, 'r') as f:
    config = yaml.safe_load(f)

mean_Z_c = ROOT.RooRealVar("mean_Z_c%d"%(int(args.config)), "mean_Z_c%d"%(int(args.config)), config['parameters']['nominal']['mean']['value'], 85, 100)

sigma_CB_c = ROOT.RooRealVar("sigma_CB_c%d"%(int(args.config)), "sigma_CB_c%d"%(int(args.config)),      config['parameters']['nominal']['sigma']['value'],   0.001, 20)
sigma_gaus_c = ROOT.RooRealVar("sigma_gaus_c%d"%(int(args.config)), "sigma_CB_c%d"%(int(args.config)),  config['parameters']['nominal']['sigmaG']['value'],  0.001, 20)

alpha1_c = ROOT.RooRealVar("alpha1_c%d"%(int(args.config)), "Alpha1_c%d"%(int(args.config)), config['parameters']['nominal']['alphaL']['value'], 0.1, 5.0)
alpha2_c = ROOT.RooRealVar("alpha2_c%d"%(int(args.config)), "Alpha2_c%d"%(int(args.config)), config['parameters']['nominal']['alphaR']['value'], 0.1, 5.0)
nL_c = ROOT.RooRealVar("nL_c%d"%(int(args.config)), "nL_c%d"%(int(args.config)), config['parameters']['nominal']['nL']['value'], 0.1, 100.0)
nR_c = ROOT.RooRealVar("nR_c%d"%(int(args.config)), "nR_c%d"%(int(args.config)), 1+config['parameters']['nominal']['nR']['value'], 0.1, 100.0)

gaus_c = ROOT.RooGaussian("gaus_c%d"%(int(args.config)), "gaus_c%d"%(int(args.config)), x_cat, mean_Z_c, sigma_gaus_c)
dscb_c = ROOT.RooDoubleCB("dscb_c%d"%(int(args.config)), "Double-Sided Crystal Ball",
                        x_cat, mean_Z_c, sigma_CB_c, alpha1_c, nL_c, alpha2_c, nR_c)
fraction_gaus_CB_c = ROOT.RooRealVar("fraction_gaus_CB_c%d"%(int(args.config)), "fraction_gaus_CB_c%d"%(int(args.config)), config['parameters']['nominal']['fraction_dscb']['value'], 0, 1)
alpha1_c.setConstant(True)
alpha2_c.setConstant(True)
nL_c.setConstant(True)
nR_c.setConstant(True)
sigma_CB_c.setConstant(True)
sigma_gaus_c.setConstant(True)
mean_Z_c.setConstant(True)
fraction_gaus_CB_c.setConstant(True)
model_Z_c = ROOT.RooAddPdf("model_Z_c%d"%(int(args.config)), "model_Z_c%d"%(int(args.config)), 
                         ROOT.RooArgList(dscb_c, gaus_c),
                         ROOT.RooArgList(fraction_gaus_CB_c))
model_Z_c.fixCoefNormalization(ROOT.RooArgSet(x_cat))


# Define the Higgs
with open(config_path_H, 'r') as f:
    config_H = yaml.safe_load(f)

mean_H_c = ROOT.RooRealVar("mean_H_c%d"%(int(args.config)), "mean_H_c%d"%(int(args.config)), config_H['parameters']['nominal']['mean']['value'], 85, 150)

sigma_CB_H_c = ROOT.RooRealVar("sigma_CB_H_c%d"%(int(args.config)), "sigma_CB_H_c%d"%(int(args.config)),      config_H['parameters']['nominal']['sigma']['value'],   0.001, 20)
sigma_gaus_H_c = ROOT.RooRealVar("sigma_gaus_H_c%d"%(int(args.config)), "sigma_gaus_H_c%d"%(int(args.config)),  config_H['parameters']['nominal']['sigmaG']['value'],  0.001, 20)

alpha1_H_c = ROOT.RooRealVar("alpha1_H_c%d"%(int(args.config)), "Alpha1_H_c%d"%(int(args.config)), config_H['parameters']['nominal']['alphaL']['value'], 0.1, 5.0)
alpha2_H_c = ROOT.RooRealVar("alpha2_H_c%d"%(int(args.config)), "Alpha2_H_c%d"%(int(args.config)), config_H['parameters']['nominal']['alphaR']['value'], 0.1, 5.0)
nL_H_c = ROOT.RooRealVar("nL_H_c%d"%(int(args.config)), "nL_H_c%d"%(int(args.config)), config_H['parameters']['nominal']['nL']['value'], 0.1, 100.0)
nR_H_c = ROOT.RooRealVar("nR_H_c%d"%(int(args.config)), "nR_H_c%d"%(int(args.config)), config_H['parameters']['nominal']['nR']['value'], 0.1, 100.0)

gaus_H_c = ROOT.RooGaussian("gaus_H_c%d"%(int(args.config)), "gaus_H_c%d"%(int(args.config)), x_cat, mean_H_c, sigma_gaus_H_c)
dscb_H_c = ROOT.RooDoubleCB("dscb_H_c%d"%(int(args.config)), "Double-Sided Crystal Ball",
                        x_cat, mean_H_c, sigma_CB_H_c, alpha1_H_c, nL_H_c, alpha2_H_c, nR_H_c)
fraction_gaus_CB_H_c = ROOT.RooRealVar("fraction_gaus_CB_H_c%d"%(int(args.config)), "fraction_gaus_CB_H_c%d"%(int(args.config)), config_H['parameters']['nominal']['fraction_dscb']['value'], 0, 1)
alpha1_H_c.setConstant(True)
alpha2_H_c.setConstant(True)
nL_H_c.setConstant(True)
nR_H_c.setConstant(True)
sigma_CB_H_c.setConstant(True)
sigma_gaus_H_c.setConstant(True)
mean_H_c.setConstant(True)
fraction_gaus_CB_H_c.setConstant(True)
model_H_c = ROOT.RooAddPdf("model_H_c%d"%(int(args.config)), "model_H_c%d"%(int(args.config)), 
                         ROOT.RooArgList(dscb_H_c, gaus_H_c),
                         ROOT.RooArgList(fraction_gaus_CB_H_c))
model_H_c.fixCoefNormalization(ROOT.RooArgSet(x_cat))


# ________ Composite model  __________
nZ_cat = ROOT.RooRealVar("nZ_cat%d"%(int(args.config)), "nZ_cat%d"%(int(args.config)), np.sum(weights_Z), 0, 5e4)
nZ_cat.setConstant(True)

nH_cat = ROOT.RooRealVar("nH_cat%d"%(int(args.config)), "nH_cat%d"%(int(args.config)), np.sum(dfMC_H.weight), 0, 5e4)
nH_cat.setConstant(True)
print("*"*30)
print("[INFO] Number of Higgs expected in category %s is %.2f"%(args.config, nH_cat.getVal()))
print("*"*30)
print("\n"*15)


mu = ROOT.RooRealVar("mu", "signal strength", 1.0, 0.0, 2.0)
mu.setConstant(False)
nZ_times_mu = ROOT.RooFormulaVar("nZ_times_mu", "@0 * @1", ROOT.RooArgList(nZ_cat, mu))
extended_Z_cat = ROOT.RooExtendPdf("extended_Z_cat%d"%(int(args.config)), "extended_Z_cat%d"%(int(args.config)), model_Z_c, nZ_times_mu)

mu_H = ROOT.RooRealVar("mu_H", "signal strength", 1.0, 0.0, 2.0)
mu_H.setConstant(False)
nH_times_mu_c = ROOT.RooFormulaVar("nH_times_mu_c%d"%(int(args.config)), "@0 * @1", ROOT.RooArgList(nH_cat, mu_H))
extended_H_cat = ROOT.RooExtendPdf("extended_H_cat%d"%(int(args.config)), "extended_H_cat%d"%(int(args.config)), model_H_c, nH_times_mu_c)

# %%
# --- Workspace ---

w = ROOT.RooWorkspace("w", "workspace")
f_out = ROOT.TFile("/t3home/gcelotto/ggHbb/WSFit/ws/step1/ws%d.root"%(int(args.config)), "RECREATE")
w_sig = ROOT.RooWorkspace("workspace_sig","workspace_sig")

### Import variables and models
w_import = getattr(w_sig, "import")

# Then safely import one by one
for obj in [x_cat, rooHist_data_cat, rooHist_Z_cat, extended_Z_cat, model_H_c, extended_H_cat, model_Z_c, rooHist_H_cat]:
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
print(np.sum(weights_Z))


# %%
