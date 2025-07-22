# %%
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
from helpers.plotFree import plotFree
from helpers.defineFunctions import defineFunctions
from helpers.allFunctions import *
from hist import Hist
#"k" stands for keep the compiled shared library after the script exits.
from array import array
import argparse


from getDfs import getDfs
# %%

dfMC_Z, dfMC_H, df, nbins, x1, x2 = getDfs(idx=2)
print(x1, x2, nbins)

#dfMC_Z[dfMC_Z.dijet_mass>50]


# %%
# Category 2
# Create histogram
hist_data_cat2 = ROOT.TH1F("hist_data_cat2", "hist_data_cat2", nbins-1, x1, x2)
hist_Z_cat2 = ROOT.TH1F("hist_Z_cat2", "hist_Z_cat2", nbins-1, x1, x2)
data = array('d', df.dijet_mass)
Z_MC = array('d', dfMC_Z.dijet_mass)



# Fast fill: FillN(n, values, weights)
weights = array('d', [1.0] * len(data))
weights_Z = array('d', dfMC_Z.weight)

hist_data_cat2.FillN(len(data), data, weights)
hist_Z_cat2.FillN(len(Z_MC), Z_MC, weights_Z)



# To be saved
x_cat2 = ROOT.RooRealVar("dijet_mass_c2", "dijet_mass_c2", x1,x2)
rooHist_data_cat2 = ROOT.RooDataHist("rooHist_data_cat2", "binned data", ROOT.RooArgList(x_cat2), hist_data_cat2)
rooHist_Z_cat2 = ROOT.RooDataHist("rooHist_Z_cat2", "binned Z", ROOT.RooArgList(x_cat2), hist_Z_cat2)




# %%
# Category 2
# --- Composite Analytic Model ---

# --- Bernstein polynomial B1(x) ---
b0_c2 = ROOT.RooRealVar("b0_c2", "b0_c2", 0.0015, 0, 0.003)
b1_c2 = ROOT.RooRealVar("b1_c2", "b1_c2", 0.09, 0.08, 0.5)
b2_c2 = ROOT.RooRealVar("b2_c2", "b2_c2", 0.9, 0.8, 1)
bern_damped_c2 = ROOT.RooBernstein("bern_damped_c2", "bern_damped_c2", x_cat2, ROOT.RooArgList(b0_c2, b1_c2, b2_c2))
# --- expo function ---
K1_c2 = ROOT.RooRealVar("K1_c2", "K1_c2", -0.08886029701243031 , -0.2, 0)
K1_c2.setConstant(True)
expo1_c2 = ROOT.RooExponential("expo1_c2", "expo1_c2", x_cat2, K1_c2)
K2_c2 = ROOT.RooRealVar("K2_c2", "K2_c2", -0.010840017280447626 , -0.2, 0)
K2_c2.setConstant(True)
expo2_c2 = ROOT.RooExponential("expo2_c2", "expo2_c2", x_cat2, K2_c2)

qcd_dampedTerm_c2 = ROOT.RooProdPdf("qcd_dampedTerm_c2", "qcd_dampedTerm_c2",
                            ROOT.RooArgList(expo1_c2, bern_damped_c2))



fraction_qcdParts_c2 = ROOT.RooRealVar("fraction_qcdParts_c2", "fraction_qcdParts_c2", 0.11618187099, 0.02, 0.9)
fraction_qcdParts_c2.setConstant(True)
qcd_total_c2 = ROOT.RooAddPdf("qcd_total_c2", "qcd_total_c2", 
                            ROOT.RooArgList(qcd_dampedTerm_c2, expo2_c2),
                            ROOT.RooArgList(fraction_qcdParts_c2))
qcd_total_c2.fixCoefNormalization(ROOT.RooArgSet(x_cat2))




# --- DoubleSided CB
config_path_Z = "/t3home/gcelotto/ggHbb/newFit/afterNN/cat2/cat2p1/fit_parameters_with_systematics_Z.json"
with open(config_path_Z, 'r') as f:
    config = yaml.safe_load(f)

mean_Z_c2 = ROOT.RooRealVar("mean_Z_c2", "mean_Z_c2", config['parameters']['nominal']['mean']['value'], 85, 100)

sigma_CB_c2 = ROOT.RooRealVar("sigma_CB_c2", "sigma_CB_c2",      config['parameters']['nominal']['sigma']['value'],   0.001, 20)
sigma_gaus_c2 = ROOT.RooRealVar("sigma_gaus_c2", "sigma_CB_c2",  config['parameters']['nominal']['sigmaG']['value'],  0.001, 20)

alpha1_c2 = ROOT.RooRealVar("alpha1_c2", "Alpha1_c2", config['parameters']['nominal']['alphaL']['value'], 0.1, 5.0)
alpha2_c2 = ROOT.RooRealVar("alpha2_c2", "Alpha2_c2", config['parameters']['nominal']['alphaR']['value'], 0.1, 5.0)
nL_c2 = ROOT.RooRealVar("nL_c2", "nL_c2", config['parameters']['nominal']['nL']['value'], 0.1, 100.0)
nR_c2 = ROOT.RooRealVar("nR_c2", "nR_c2", 1+config['parameters']['nominal']['nR']['value'], 0.1, 100.0)

gaus_c2 = ROOT.RooGaussian("gaus_c2", "gaus_c2", x_cat2, mean_Z_c2, sigma_gaus_c2)
dscb_c2 = ROOT.RooDoubleCB("dscb_c2", "Double-Sided Crystal Ball",
                        x_cat2, mean_Z_c2, sigma_CB_c2, alpha1_c2, nL_c2, alpha2_c2, nR_c2)
fraction_gaus_CB_c2 = ROOT.RooRealVar("fraction_gaus_CB_c2", "fraction_gaus_CB_c2", config['parameters']['nominal']['fraction_dscb']['value'], 0, 1)
alpha1_c2.setConstant(True)
alpha2_c2.setConstant(True)
nL_c2.setConstant(True)
nR_c2.setConstant(True)
sigma_CB_c2.setConstant(True)
sigma_gaus_c2.setConstant(True)
mean_Z_c2.setConstant(True)
fraction_gaus_CB_c2.setConstant(True)
model_Z_c2 = ROOT.RooAddPdf("model_Z_c2", "model_Z_c2", 
                         ROOT.RooArgList(dscb_c2, gaus_c2),
                         ROOT.RooArgList(fraction_gaus_CB_c2))
model_Z_c2.fixCoefNormalization(ROOT.RooArgSet(x_cat2))


# ________ Composite model  __________
nQCD_cat2 = ROOT.RooRealVar("nQCD_cat2", "QCD yield cat2", 138_125*9.5, 0, 3e6)
nZ_cat2 = ROOT.RooRealVar("nZ_cat2", "nZ_cat2", np.sum(weights_Z), 0, 5e4)
nZ_cat2.setConstant(True)


mu = ROOT.RooRealVar("mu", "signal strength", 1.0, 0.0, 2.0)
mu.setConstant(True)
nZ_times_mu_c2 = ROOT.RooFormulaVar("nZ_times_mu_c2", "@0 * @1", ROOT.RooArgList(nZ_cat2, mu))



ext_sig_cat2 = ROOT.RooExtendPdf("ext_sig_cat2", "Z sig cat2", model_Z_c2, nZ_times_mu_c2)
#ext_sig_cat2 = ROOT.RooExtendPdf("ext_sig_cat2", "Z sig cat2", model_Z_c2, nZ_cat2)
ext_qcd_cat2 = ROOT.RooExtendPdf("ext_qcd_cat2", "QCD cat2", qcd_total_c2, nQCD_cat2)
model_cat2 = ROOT.RooAddPdf("model_cat2", "total cat2", ROOT.RooArgList(ext_sig_cat2, ext_qcd_cat2))
model_cat2.fixCoefNormalization(ROOT.RooArgSet(x_cat2))





# _________________________________________________________________________________
# _________________________________________________________________________________
# _________________________________________________________________________________






### Category 1
dfMC_Z_cat1, dfMC_H, df_cat1, nbins, x1, x2 = getDfs(idx=2)
dfMC_Z_cat1=dfMC_Z_cat1[(dfMC_Z_cat1.dijet_mass>50)]
print("max dijet mass for z in cat 1", dfMC_Z_cat1.dijet_mass.max())

### Create histogram
hist_data_cat1 = ROOT.TH1F("hist_data_cat1", "hist_data_cat1", nbins, x1, x2)
hist_Z_cat1 = ROOT.TH1F("hist_Z_cat1", "hist_Z_cat1", nbins, x1, x2)
data_cat1_array = array('d', df_cat1.dijet_mass)
Z_MC_array_cat1 = array('d', dfMC_Z_cat1.dijet_mass)

### Fast fill: FillN(n, values, weights)
weights_cat1 = array('d', [1.0] * len(data_cat1_array))
weights_Z_cat1 = array('d', dfMC_Z_cat1.weight)
##
hist_data_cat1.FillN(len(data_cat1_array), data_cat1_array, weights_cat1)
hist_Z_cat1.FillN(len(Z_MC_array_cat1), Z_MC_array_cat1, weights_Z_cat1)

### To be saved
# To be saved
x_cat1 = ROOT.RooRealVar("dijet_mass_c1", "dijet_mass_c1", x1,x2)
rooHist_data_cat1 = ROOT.RooDataHist("rooHist_data_cat1", "binned data", ROOT.RooArgList(x_cat1), hist_data_cat1)
rooHist_Z_cat1 = ROOT.RooDataHist("rooHist_Z_cat1", "binned Z", ROOT.RooArgList(x_cat1), hist_Z_cat1)




# %%
# Category 1
# --- Composite Analytic Model ---
# If you provide 𝑛 coefficients, the degree is 𝑛−1
# --- Bernstein polynomial B1(x) ---


# --- Bernstein polynomial B1(x) ---
b0_c1 = ROOT.RooRealVar("b0_c1", "b0_c1", 0.0000644, 0, 1)
b1_c1 = ROOT.RooRealVar("b1_c1", "b1_c1", 0.7458296, 0, 1)
b2_c1 = ROOT.RooRealVar("b2_c1", "b2_c1", 0.9929227, 0, 1)
bern_damped_c1 = ROOT.RooBernstein("bern_damped_c1", "bern_damped_c1", x_cat1, ROOT.RooArgList(b0_c1, b1_c1, b2_c1))
# --- expo function ---
K1_c1 = ROOT.RooRealVar("K1_c1", "K1_c1", -0.0987983 , -0.2, 0)
expo1_c1 = ROOT.RooExponential("expo1_c1", "expo1_c1", x_cat1, K1_c1)
#K2_c1 = ROOT.RooRealVar("K2_c1", "K2_c1", -0.0046 , -0.1, 0)
#expo2_c1 = ROOT.RooExponential("expo2_c1", "expo2_c1", x_cat1, K2_c1)
#fraction_expos = ROOT.RooRealVar("fraction_expos", "fraction_expos", 0.5 , 0.4999, 0.50001)
#fraction_expos.setConstant(True)
#expo_part_c1 = ROOT.RooAddPdf("expo_part_c1", "expo_part_c1", ROOT.RooArgList(expo1_c1, expo2_c1), 
#                              ROOT.RooArgList(fraction_expos))
#expo_part_c1.fixCoefNormalization(ROOT.RooArgSet(x_cat1))
#
qcd_dampedTerm_c1 = ROOT.RooProdPdf("qcd_dampedTerm_c1", "qcd_dampedTerm_c1",
                            ROOT.RooArgList(expo1_c1, bern_damped_c1))
# --- Bernstein polynomial B2(x) ---
bb0_c1 = ROOT.RooRealVar("bb0_c1", "bb0_c1", 0.9992423, 0, 1)
bb1_c1 = ROOT.RooRealVar("bb1_c1", "bb1_c1", 0.1097371, 0, 1)
bb2_c1 = ROOT.RooRealVar("bb2_c1", "bb2_c1", 0.0001478, 0, 1)
bern_notDamped_c1 = ROOT.RooBernstein("bern_notDamped_c1", "bern_notDamped_c1", x_cat1, ROOT.RooArgList(bb0_c1, bb1_c1, bb2_c1))



fraction_qcdParts_c1 = ROOT.RooRealVar("fraction_qcdParts_c1", "fraction_qcdParts_c1", 0.2, 0.1, 0.9)
#fraction_qcdParts_c1.setConstant(True)
qcd_total_c1 = ROOT.RooAddPdf("qcd_total_c1", "qcd_total_c1", 
                            ROOT.RooArgList(qcd_dampedTerm_c1, bern_notDamped_c1),
                            ROOT.RooArgList(fraction_qcdParts_c1))
qcd_total_c1.fixCoefNormalization(ROOT.RooArgSet(x_cat1))









# --- DoubleSided CB
config_path_Z = "/t3home/gcelotto/ggHbb/newFit/afterNN/cat1/fit_parameters_with_systematics_Z.json"
with open(config_path_Z, 'r') as f:
    config = yaml.safe_load(f)

mean_Z_c1 = ROOT.RooRealVar("mean_Z_c1", "mean_Z_c1", config['parameters']['nominal']['mean']['value'], 85, 100)

sigma_CB_c1 = ROOT.RooRealVar("sigma_CB_c1", "sigma_CB_c1",      config['parameters']['nominal']['sigma']['value'],   0.001, 20)
sigma_gaus_c1 = ROOT.RooRealVar("sigma_gaus_c1", "sigma_CB_c1",  config['parameters']['nominal']['sigmaG']['value'],  0.001, 20)

alpha1_c1 = ROOT.RooRealVar("alpha1_c1", "Alpha1_c1", config['parameters']['nominal']['alphaL']['value'], 0.1, 5.0)
alpha2_c1 = ROOT.RooRealVar("alpha2_c1", "Alpha2_c1", config['parameters']['nominal']['alphaR']['value'], 0.1, 5.0)
nL_c1 = ROOT.RooRealVar("nL_c1", "nL_c1", config['parameters']['nominal']['nL']['value'], 0.1, 100.0)
nR_c1 = ROOT.RooRealVar("nR_c1", "nR_c1", 1+config['parameters']['nominal']['nR']['value'], 0.1, 100.0)

gaus_c1 = ROOT.RooGaussian("gaus_c1", "gaus_c1", x_cat1, mean_Z_c1, sigma_gaus_c1)
dscb_c1 = ROOT.RooDoubleCB("dscb_c1", "Double-Sided Crystal Ball",
                        x_cat1, mean_Z_c1, sigma_CB_c1, alpha1_c1, nL_c1, alpha2_c1, nR_c1)
fraction_gaus_CB_c1 = ROOT.RooRealVar("fraction_gaus_CB_c1", "fraction_gaus_CB_c1", config['parameters']['nominal']['fraction_dscb']['value'], 0, 1)
alpha1_c1.setConstant(True)
alpha2_c1.setConstant(True)
nL_c1.setConstant(True)
nR_c1.setConstant(True)
sigma_CB_c1.setConstant(True)
sigma_gaus_c1.setConstant(True)
mean_Z_c1.setConstant(True)
fraction_gaus_CB_c1.setConstant(True)
model_Z_c1 = ROOT.RooAddPdf("model_Z_c1", "model_Z_c1", 
                         ROOT.RooArgList(dscb_c1, gaus_c1),
                         ROOT.RooArgList(fraction_gaus_CB_c1))
model_Z_c1.fixCoefNormalization(ROOT.RooArgSet(x_cat1))


# ________ Composite model  __________
nQCD_cat1 = ROOT.RooRealVar("nQCD_cat1", "QCD yield cat1", 2e6, 5e5, 6e6)
nZ_cat1 = ROOT.RooRealVar("nZ_cat1", "nZ_cat1", np.sum(weights_Z_cat1), 4e3, 45e3)
nZ_cat1.setConstant(True)
print("Sum of weights ", np.sum(weights_Z_cat1))

nZ_times_mu_c1 = ROOT.RooFormulaVar("nZ_times_mu_c1", "@0 * @1", ROOT.RooArgList(nZ_cat1, mu))



ext_sig_cat1 = ROOT.RooExtendPdf("ext_sig_cat1", "Z sig cat1", model_Z_c1, nZ_times_mu_c1)
ext_qcd_cat1 = ROOT.RooExtendPdf("ext_qcd_cat1", "QCD cat1", qcd_total_c1, nQCD_cat1)
model_cat1 = ROOT.RooAddPdf("model_cat1", "total cat1", ROOT.RooArgList(ext_sig_cat1, ext_qcd_cat1))
model_cat1.fixCoefNormalization(ROOT.RooArgSet(x_cat1))


# %%
# --- Workspace ---

w = ROOT.RooWorkspace("w", "workspace")
f_out = ROOT.TFile("/t3home/gcelotto/ggHbb/newFit/rooFit/workspace_sig.root", "RECREATE")
w_sig = ROOT.RooWorkspace("workspace_sig","workspace_sig")
#
### Import variables and models
w_import = getattr(w_sig, "import")

# Then safely import one by one
for obj in [x_cat2, rooHist_data_cat2, rooHist_Z_cat2,
            x_cat1, rooHist_data_cat1, rooHist_Z_cat1,
            model_cat1, model_cat2]:
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
