# Need to run non Interactively!


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


from getDfs import getDfs
# %%

dfMC_Z, dfMC_H, df, nbins, x1, x2 = getDfs(idx=int(args.config))
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


# Repeat with broader binning
hist_data_broad_cat2 = ROOT.TH1F("hist_data_broad_cat2", "hist_data_broad_cat2", 101, x1, x2)
hist_Z_broad_cat2 = ROOT.TH1F("hist_Z_broad_cat2", "hist_Z_broad_cat2", 101, x1, x2)
hist_data_broad_cat2.FillN(len(data), data, weights)
hist_Z_broad_cat2.FillN(len(Z_MC), Z_MC, weights_Z)
rooHist_data_broad_cat2 = ROOT.RooDataHist("rooHist_data_broad_cat2", "binned data", ROOT.RooArgList(x_cat2), hist_data_broad_cat2)
rooHist_Z_broad_cat2 = ROOT.RooDataHist("rooHist_Z_broad_cat2", "binned Z", ROOT.RooArgList(x_cat2), hist_Z_broad_cat2)




# %%
# Category 2
# --- F1 ---
# --- Bernstein polynomial B1(x) ---
b0_c2 = ROOT.RooRealVar("b0_c2", "b0_c2", 0.025, 0, 1)
b1_c2 = ROOT.RooRealVar("b1_c2", "b1_c2", 0.5, 0, 1)
b2_c2 = ROOT.RooRealVar("b2_c2", "b2_c2", 1, 0, 4)
bern_damped_c2 = ROOT.RooBernstein("bern_damped_c2", "bern_damped_c2", x_cat2, ROOT.RooArgList(b0_c2, b1_c2, b2_c2))
b0_c2.setConstant(False)
b1_c2.setConstant(True)
b2_c2.setConstant(True)
# --- expo function ---
K1_c2 = ROOT.RooRealVar("K1_c2", "K1_c2", -0.0744421 , -0.1, -0.02)
K1_c2.setConstant(False)
expo1_c2 = ROOT.RooExponential("expo1_c2", "expo1_c2", x_cat2, K1_c2)
K2_c2 = ROOT.RooRealVar("K2_c2", "K2_c2", -0.0107676, -0.02, -0.005)
K2_c2.setConstant(False)
expo2_c2 = ROOT.RooExponential("expo2_c2", "expo2_c2", x_cat2, K2_c2)

qcd_dampedTerm_c2 = ROOT.RooProdPdf("qcd_dampedTerm_c2", "qcd_dampedTerm_c2",
                            ROOT.RooArgList(expo1_c2, bern_damped_c2))



fraction_qcdParts_c2 = ROOT.RooRealVar("fraction_qcdParts_c2", "fraction_qcdParts_c2", 0.12116454804437206, 0.05, 0.2)
fraction_qcdParts_c2.setConstant(False)
f1_c2 = ROOT.RooAddPdf("f1_c2", "f1_c2", 
                            ROOT.RooArgList(qcd_dampedTerm_c2, expo2_c2),
                            ROOT.RooArgList(fraction_qcdParts_c2))
f1_c2.fixCoefNormalization(ROOT.RooArgSet(x_cat2))

# --- F0 ---
# --- Bernstein polynomial B1(x) ---
xo_c2 = ROOT.RooRealVar("xo_c2", "xo_c2", 52.9, 20, 70)
k_c2 = ROOT.RooRealVar("k_c2", "k_c2", 5.33, 0.01, 10)
delta_c2 = ROOT.RooRealVar("delta_c2", "delta_c2", -26, -100, 0)
beta_c2 = ROOT.RooRealVar("beta_c2", "beta_c2", 0.03, 0.0001, 10)

# 3. Define the custom function using RooGenericPdf
# Equivalent to: f(x) = exp((x - xo_c2)/k_c2) / (1 + exp((x - xo_c2)/(k_c2 + delta_c2/100))) + exp(-beta_c2*(x - xo_c2))
func_expr = "exp((@0 - @1)/@2) / (1 + exp((@0 - @1)/(@2 + @3/100))) + exp(-@4/100*(@0 - @1))"

f0_c2 = ROOT.RooGenericPdf("f0_c2", func_expr, ROOT.RooArgList(x_cat2, xo_c2, k_c2, delta_c2, beta_c2))

# F16
# Parameters
p0_fall = ROOT.RooRealVar("p0_fall", "p0_fall", 88.800, 0, 1)
p1_fall = ROOT.RooRealVar("p1_fall", "p1_fall",-0.6654, 0, 1)
p2_fall = ROOT.RooRealVar("p2_fall", "p2_fall", 0.00141317, 0, 1)
p1_rise = ROOT.RooRealVar("p1_rise", "p1_rise", 1.5291, 0, 2)
k_sig       = ROOT.RooRealVar("k_sig", "k_sig", 0.0994353, 0.02, 0.3)
x0      = ROOT.RooRealVar("x0", "x0", 61.9, 50, 80)
p0_fall.setConstant(False)
p1_fall.setConstant(False)
p2_fall.setConstant(False)
p1_rise.setConstant(False)
k_sig.setConstant(False)
x0.setConstant(False)

#bernstein_f16_fall = ROOT.RooBernstein("bernstein_f16_fall", "bernstein_f16_fall PDF", x_cat2, ROOT.RooArgList(p0_fall, p1_fall, p2_fall))
#bernstein_f16_rise = ROOT.RooBernstein("bernstein_f16_rise", "bernstein_f16_rise PDF", x_cat2, ROOT.RooArgList(p0_rise, p1_rise))
#sigmoid_fall_pdf = ROOT.RooGenericPdf("sigmoid_fall_pdf", "1.0 / (1.0 + exp(-@1 * (@0 - @2)))", ROOT.RooArgList(x_cat2, k_sig, x0))
#sigmoid_rise_pdf = ROOT.RooGenericPdf("sigmoid_rise_pdf", "1.0 - (1.0 / (1.0 + exp(-@1 * (@0 - @2))))", ROOT.RooArgList(x_cat2, k_sig, x0))
#prod_rise = ROOT.RooProdPdf("prod_rise", "bernstein_f16_fall * sigmoid_fall", ROOT.RooArgList(bernstein_f16_fall, sigmoid_fall_pdf))
#prod_fall = ROOT.RooProdPdf("prod_fall", "bernstein_f16_rise * sigmoid_rise", ROOT.RooArgList(bernstein_f16_rise, sigmoid_rise_pdf))
#tempFraction = ROOT.RooRealVar("tempFraction", "tempFraction", 0.5, 0.4999, 0.50001)
#tempFraction.setConstant(True)
#f16_c2 = ROOT.RooAddPdf("f16_c2", "combined rise + fall", ROOT.RooArgList(prod_rise, prod_fall), ROOT.RooArgList(tempFraction))
#f16_c2.fixCoefNormalization(ROOT.RooArgSet(x_cat2))

# Final combined function
# Function expression using @ indices
#fall*sigmoid + rise*1-sigmoid
f16_expr = "(@2 * @0 * @0 + @1 * @0 + @3) * (1.0 / (1.0 + exp(-@5 * (@0 - @6)))) + (@4 * @0 + 1) * (1.0 / (1.0 + exp(@5 * (@0 - @6))))"

# Map: @0 = x_cat2
#      @1 = p1_fall
#      @2 = p2_fall
#      @3 = p0_fall
#      @4 = p1_Rise
#      @5 = k_sig
#      @6 = x0

f16_c2 = ROOT.RooGenericPdf("f16_c2", f16_expr, 
    ROOT.RooArgList(x_cat2, p1_fall, p2_fall, p0_fall, p1_rise, k_sig, x0))
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


# Define the Higgs
config_path_H = "/t3home/gcelotto/ggHbb/newFit/afterNN/cat2/cat2p1/fit_parameters_with_systematics_H.json"
with open(config_path_H, 'r') as f:
    config_H = yaml.safe_load(f)

mean_H_c2 = ROOT.RooRealVar("mean_H_c2", "mean_H_c2", config_H['parameters']['nominal']['mean']['value'], 85, 150)

sigma_CB_c2_H = ROOT.RooRealVar("sigma_CB_c2_H", "sigma_CB_c2_H",      config_H['parameters']['nominal']['sigma']['value'],   0.001, 20)
sigma_gaus_c2_H = ROOT.RooRealVar("sigma_gaus_c2_H", "sigma_gaus_c2_H",  config_H['parameters']['nominal']['sigmaG']['value'],  0.001, 20)

alpha1_c2_H = ROOT.RooRealVar("alpha1_c2_H", "Alpha1_c2_H", config_H['parameters']['nominal']['alphaL']['value'], 0.1, 5.0)
alpha2_c2_H = ROOT.RooRealVar("alpha2_c2_H", "Alpha2_c2_H", config_H['parameters']['nominal']['alphaR']['value'], 0.1, 5.0)
nL_c2_H = ROOT.RooRealVar("nL_c2_H", "nL_c2_H", config_H['parameters']['nominal']['nL']['value'], 0.1, 100.0)
nR_c2_H = ROOT.RooRealVar("nR_c2_H", "nR_c2_H", config_H['parameters']['nominal']['nR']['value'], 0.1, 100.0)

gaus_c2_H = ROOT.RooGaussian("gaus_c2_H", "gaus_c2_H", x_cat2, mean_H_c2, sigma_gaus_c2_H)
dscb_c2_H = ROOT.RooDoubleCB("dscb_c2_H", "Double-Sided Crystal Ball",
                        x_cat2, mean_H_c2, sigma_CB_c2_H, alpha1_c2_H, nL_c2_H, alpha2_c2_H, nR_c2_H)
fraction_gaus_CB_c2_H = ROOT.RooRealVar("fraction_gaus_CB_c2_H", "fraction_gaus_CB_c2_H", config_H['parameters']['nominal']['fraction_dscb']['value'], 0, 1)
alpha1_c2_H.setConstant(True)
alpha2_c2_H.setConstant(True)
nL_c2_H.setConstant(True)
nR_c2_H.setConstant(True)
sigma_CB_c2_H.setConstant(True)
sigma_gaus_c2_H.setConstant(True)
mean_H_c2.setConstant(True)
fraction_gaus_CB_c2_H.setConstant(True)
model_H_c2 = ROOT.RooAddPdf("model_H_c2", "model_H_c2", 
                         ROOT.RooArgList(dscb_c2_H, gaus_c2_H),
                         ROOT.RooArgList(fraction_gaus_CB_c2_H))
model_H_c2.fixCoefNormalization(ROOT.RooArgSet(x_cat2))


# ________ Composite model  __________
nQCD_cat2 = ROOT.RooRealVar("nQCD_cat2", "QCD yield cat2", 1381333.4236343, 1e6, 2e6)
nZ_cat2 = ROOT.RooRealVar("nZ_cat2", "nZ_cat2", np.sum(weights_Z), 0, 5e4)
nZ_cat2.setConstant(True)

nH_cat2 = ROOT.RooRealVar("nH_cat2", "nH_cat2", np.sum(dfMC_H.weight), 0, 5e4)
nH_cat2.setConstant(True)


mu = ROOT.RooRealVar("mu", "signal strength", 1.0, 0.0, 2.0)
mu.setConstant(False)
nZ_times_mu_c2 = ROOT.RooFormulaVar("nZ_times_mu_c2", "@0 * @1", ROOT.RooArgList(nZ_cat2, mu))


mu_H = ROOT.RooRealVar("mu_H", "signal strength", 1.0, 0.0, 2.0)
mu_H.setConstant(False)
nH_times_mu_c2 = ROOT.RooFormulaVar("nH_times_mu_c2", "@0 * @1", ROOT.RooArgList(nH_cat2, mu_H))
extended_H_cat2 = ROOT.RooExtendPdf("extended_H_cat2", "extended_H_cat2", model_H_c2, nH_times_mu_c2)


#ext_sig_cat2 = ROOT.RooExtendPdf("ext_sig_cat2", "Z sig cat2", model_Z_c2, nZ_times_mu_c2)
#ext_qcd_cat2 = ROOT.RooExtendPdf("ext_qcd_cat2", "QCD cat2", f1_c2, nQCD_cat2)
model_f1_cat2 =  ROOT.RooAddPdf("model_f1_cat2", "sig+bkg", ROOT.RooArgList(f1_c2, model_Z_c2), ROOT.RooArgList(nQCD_cat2, nZ_times_mu_c2))
model_f1_cat2.fixCoefNormalization(ROOT.RooArgSet(x_cat2))

model_f0_cat2 =  ROOT.RooAddPdf("model_f0_cat2", "sig+bkg", ROOT.RooArgList(f0_c2, model_Z_c2), ROOT.RooArgList(nQCD_cat2, nZ_times_mu_c2))
model_f0_cat2.fixCoefNormalization(ROOT.RooArgSet(x_cat2))

model_f16_cat2 =  ROOT.RooAddPdf("model_f16_cat2", "sig+bkg", ROOT.RooArgList(f16_c2, model_Z_c2), ROOT.RooArgList(nQCD_cat2, nZ_times_mu_c2))
model_f16_cat2.fixCoefNormalization(ROOT.RooArgSet(x_cat2))

#sum_expr = "nQCD_cat2 + nZ_times_mu_c2"
#background_norm = ROOT.RooFormulaVar("background_norm", "Sum of QCD and Z", sum_expr,
#                             ROOT.RooArgList(nQCD_cat2, nZ_times_mu_c2))


x_cat2.setRange("R1", 53,   105)
x_cat2.setRange("R2", 140,   200)
#fit_result = model_f1_cat2.fitTo(rooHist_data_cat2,
#                                           #ROOT.RooFit.IntegrateBins(1e-2),
#                            ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(0), 
#                           ROOT.RooFit.Strategy(2),
#                           ROOT.RooFit.Extended(True), # To save normalization
#                           ROOT.RooFit.SumW2Error(True), 
#                           ROOT.RooFit.Save(),
#                           Range="R1,R2",
#                           )
#
#fit_result = model_f0_cat2.fitTo(rooHist_data_cat2,
#                                           #ROOT.RooFit.IntegrateBins(1e-2),
#                            ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(0), 
#                           ROOT.RooFit.Strategy(2),
#                           ROOT.RooFit.Extended(True), # To save normalization
#                           ROOT.RooFit.SumW2Error(True), 
#                           ROOT.RooFit.Save(),
#                           Range="R1,R2",
#                           )
#fit_result = model_f16_cat2.fitTo(rooHist_data_cat2,
#                                           #ROOT.RooFit.IntegrateBins(1e-2),
#                            ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(0), 
#                           ROOT.RooFit.Strategy(2),
#                           ROOT.RooFit.Extended(True), # To save normalization
#                           ROOT.RooFit.SumW2Error(True), 
#                           ROOT.RooFit.Save(),
#                           Range="R1,R2",
#                           )






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
# If you provide  ncoefficients, the degree is n−1
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
sigma_gaus_c1 = ROOT.RooRealVar("sigma_gaus_c1", "sigma_gaus_c1",  config['parameters']['nominal']['sigmaG']['value'],  0.001, 20)

alpha1_c1 = ROOT.RooRealVar("alpha1_c1", "Alpha1_c1", config['parameters']['nominal']['alphaL']['value'], 0.1, 5.0)
alpha2_c1 = ROOT.RooRealVar("alpha2_c1", "Alpha2_c1", config['parameters']['nominal']['alphaR']['value'], 0.1, 5.0)
nL_c1 = ROOT.RooRealVar("nL_c1", "nL_c1", config['parameters']['nominal']['nL']['value'], 1., 100.0)
nR_c1 = ROOT.RooRealVar("nR_c1", "nR_c1", config['parameters']['nominal']['nR']['value'], 1., 100.0)

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
for obj in [x_cat2, rooHist_data_cat2, rooHist_Z_cat2, rooHist_data_broad_cat2,rooHist_Z_broad_cat2,
            x_cat1, rooHist_data_cat1, rooHist_Z_cat1,
            model_cat1, model_f1_cat2, model_f0_cat2, model_f16_cat2, model_H_c2]:
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
#
# Create the index category
#pdf_index = ROOT.RooCategory("pdfindex", "PDF index")
#pdf_list = ROOT.RooArgList()
#pdf_list.add(model_f0_cat2)
#pdf_list.add(model_f1_cat2)
#pdf_list.add(model_f16_cat2)
#
## Wrap into a RooMultiPdf
#from ROOT import RooFit, RooMultiPdf
#multipdf = RooMultiPdf("multipdf", "Background MultiPdf", pdf_index, pdf_list)


# %%
