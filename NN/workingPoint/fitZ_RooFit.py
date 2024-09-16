# %%
import glob, re, sys
sys.path.append('/t3home/gcelotto/ggHbb/NN')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
#from iminuit import cost
#from iminuit import Minuit
#from numba_stats import crystalball_ex, expon
#from iminuit.cost import LeastSquares
import json
from loadData import loadData
import ROOT
from ROOT import RooFit, RooRealVar, RooDataSet, RooArgSet, RooCBShape, RooAddPdf, RooArgList, RooPlot

# %%
xmin = 60
xmax = 200
bins = np.linspace(xmin, xmax, 50)
nReal = 1005


# %%
# Load Data
print("Loading Data")

dfs, YPred_QCD, numEventsList = loadData(pdgID=0, nReal=nReal)
df = dfs[0]

# Z BOSON SAMPLES
dfs, YPred_Z100, YPred_Z200, YPred_Z400, YPred_Z600, YPred_Z800, numEventsList = loadData(pdgID=23)

dfs[0]['weights'] = 5.261e+03 / numEventsList[0] * dfs[0].sf * 0.774 * 1000 * nReal /1017
dfs[1]['weights'] = 1012. / numEventsList[1] * dfs[1].sf * 0.774 * 1000 * nReal /1017
dfs[2]['weights'] = 114.2 / numEventsList[2] * dfs[2].sf * 0.774 * 1000 * nReal /1017
dfs[3]['weights'] = 25.34 / numEventsList[3] * dfs[3].sf * 0.774 * 1000 * nReal /1017
dfs[4]['weights'] = 12.99 / numEventsList[4] * dfs[4].sf * 0.774 * 1000 * nReal /1017

df_Z = pd.concat(dfs)
YPred_Z = np.concatenate([YPred_Z100, YPred_Z200, YPred_Z400, YPred_Z600, YPred_Z800])

# From now on:
# df    : dataframe for Data sample
# df_Z  : dataframe for Z sample
# YPred_QCD : prediction for real data
# YPred_Z : predictions for Z sample

# %%
# Working Points
# Define the working Points (chosen to maximize Significance) and filter the dataframes

workingPoint = (YPred_QCD[:,1] > 0.34) & (YPred_QCD[:,0] < 0.22) & (df.dijet_pt > 100)
mass = df.dijet_mass[workingPoint]

workingPoint_Z = (YPred_Z[:,1] > 0.34) & (YPred_Z[:,0] < 0.22) & (df_Z.dijet_pt > 100)
mass_Z = df_Z.dijet_mass[workingPoint_Z]
weights_Z = df_Z.weights[workingPoint_Z]

# %%
# Plotting
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
x = (bins[1:] + bins[:-1])/2
counts = np.histogram(mass,  bins=bins)[0]
errors = np.sqrt(counts)
integral = np.sum(counts * np.diff(bins))

counts_Z = np.histogram(mass_Z, bins=bins, weights=weights_Z)[0]
errors_Z = np.sqrt(np.histogram(mass_Z, bins=bins, weights=weights_Z**2)[0])

ax[0].errorbar(x, counts, xerr = np.diff(bins)/2,yerr=errors, color='black', linestyle='none')
ax[0].hist(bins[:-1], bins=bins, weights=counts_Z*10, histtype=u'step', color='red', label='Z MC')[0]
ax[0].set_xlim(xmin, xmax)
ax[0].set_xlabel("Dijet Mass [GeV]")
# Z MC
ax[1].errorbar(x, counts_Z, yerr=errors_Z, color='red', linestyle='none')
ax[1].set_xlim(xmin, xmax)
ax[1].set_xlabel("Dijet Mass [GeV]")



# %%
with open('/t3home/gcelotto/ggHbb/NN/workingPoint/fit_results.json', 'r') as f:
    fit_results = json.load(f)

norm_Z_MC_ig     = fit_results['norm']['value']
beta_left_ig     = fit_results['beta_left']['value']
m_left_ig        = fit_results['m_left']['value']
scale_left_ig    = fit_results['scale_left']['value']
beta_right_ig    = fit_results['beta_right']['value']
m_right_ig       = fit_results['m_right']['value']
scale_right_ig   = fit_results['scale_right']['value']
loc_ig           = fit_results['loc']['value']
p0_ig, p1_ig, p2_ig, p3_ig, p4_ig = 100.91e3, -2.2603e3, 22.075,-99.09e-3, 166.06e-6
norm_Z_fullFit_ig = np.sum(np.sum(counts_Z)*np.diff(bins))
initial_guess = {
    'p0': p0_ig,
    'p1': p1_ig,
    'p2': p2_ig,
    'p3': p3_ig,
    'p4': p4_ig,
    'norm_Z_fullFit': norm_Z_fullFit_ig,
    'norm_Z_MC': norm_Z_MC_ig,
    'beta_left': beta_left_ig,
    'm_left': m_left_ig,
    'scale_left': scale_left_ig,
    'beta_right': beta_right_ig,
    'm_right': m_right_ig,
    'scale_right': scale_right_ig,
    'loc': loc_ig
}


# %%
x = RooRealVar("x", "mass", 40, 200)
w = RooRealVar("w", "weight", 0.0, 5.0)

dataset = RooDataSet("dataset", "dataset with weights", RooArgSet(x, w), RooFit.WeightVar(w))

# Fill the dataset with values and corresponding weights
for val, weight in zip(mass_Z, weights_Z):
    x.setVal(val)
    w.setVal(weight)
    dataset.add(RooArgSet(x, w), weight)

# %%
mean = RooRealVar("mean", "mean", 93, 86, 98)
sigma_left = RooRealVar("sigma_left", "sigma (left)", 13.3, 9, 20)
alpha_left = RooRealVar("alpha_left", "alpha (left)", 1.17, 1, 10)
n_left = RooRealVar("n_left", "n (left)", 3.1, 0.1, 30)

# Parameters for the right Crystal Ball
sigma_right = RooRealVar("sigma_right", "sigma (right)", 15, 9, 20)
alpha_right = RooRealVar("alpha_right", "alpha (right)", -1.576, -10, -0.1)
n_right = RooRealVar("n_right", "n (right)", 1, 0.1, 30)

# Create the two Crystal Ball components
crystal_left = RooCBShape("crystal_left", "Crystal Ball (left)", x, mean, sigma_left, alpha_left, n_left)
crystal_right = RooCBShape("crystal_right", "Crystal Ball (right)", x, mean, sigma_right, alpha_right, n_right)

# Combine the two with a RooAddPdf, assuming equal fractions (0.5 for each side)
frac = RooRealVar("frac", "fraction of left CB", 0.5, 0.0, 1.0)
double_cb = RooAddPdf("double_cb", "Double-sided Crystal Ball", RooArgList(crystal_left, crystal_right), RooArgList(frac))

# Fit the model to the dataset
fit_result = double_cb.fitTo(dataset, RooFit.SumW2Error(True), RooFit.Save())
corr_matrix = fit_result.correlationMatrix()
corr_matrix.Print()

# Print the covariance matrix as well
cov_matrix = fit_result.covarianceMatrix()
cov_matrix.Print()
# %%
# Plotting cell

frame = x.frame(RooFit.Title("Double-Sided Crystal Ball Fit"))

# Plot the dataset on this frame
dataset.plotOn(frame, RooFit.MarkerColor(ROOT.kBlack))

# Plot the double-sided Crystal Ball fit result on the same frame
double_cb.plotOn(frame, RooFit.LineColor(ROOT.kRed))

# Create a canvas to draw the plot
c = ROOT.TCanvas("c", "c", 800, 600)
frame.Draw()

# Calculate chi2 and ndof
chi2 = frame.chiSquare()  # This gives chi2/ndof

# Add chi2 and ndof information on the plot
chi2_text = ROOT.TLatex()
chi2_text.SetNDC()  # Use normalized device coordinates (NDC)
chi2_text.SetTextSize(0.04)  # Text size in the plot
chi2_text.SetTextAlign(33)  # Align at top-left corner

# Text to display
chi2_label = f"#chi^{{2}}/NDOF = {chi2:.3f}"

# Specify the position of the text box (x, y)
chi2_text.DrawLatex(0.85, 0.85, chi2_label)
c.Draw()
# %%
print(f"Mean: {mean.getVal()} ± {mean.getError()}")
print(f"Sigma Left: {sigma_left.getVal()} ± {sigma_left.getError()}")
print(f"Alpha Left: {alpha_left.getVal()} ± {alpha_left.getError()}")
print(f"N Left: {n_left.getVal()} ± {n_left.getError()}")

print(f"Sigma Right: {sigma_right.getVal()} ± {sigma_right.getError()}")
print(f"Alpha Right: {alpha_right.getVal()} ± {alpha_right.getError()}")
print(f"N Right: {n_right.getVal()} ± {n_right.getError()}")

print(f"Fraction of Left CB: {frac.getVal()} ± {frac.getError()}")
# %%
