# %%
import glob, re, sys
sys.path.append('/t3home/gcelotto/ggHbb/NN')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import json
from loadData import loadData
import ROOT
from ROOT import RooFit, RooRealVar, RooDataSet, RooArgSet, RooArgList, RooGenericPdf, RooCBShape, RooAddPdf

nReal = 1000
xmin = 50
xmax = 200
bins = np.linspace(xmin, xmax, 100)





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
# df        : dataframe for Data sample
# df_Z      : dataframe for Z sample
# YPred_QCD : prediction for real data
# YPred_Z   : predictions for Z sample






# %%
# Working Points
# Define the working Points (chosen to maximize Significance) and filter the dataframes

workingPoint = (YPred_QCD[:,1] > 0.34) & (YPred_QCD[:,0] < 0.22)  & (df.dijet_mass>xmin) & (df.dijet_mass<xmax) & (df.dijet_pt>100)
mass = df.dijet_mass[workingPoint]

workingPoint_Z = (YPred_Z[:,1] > 0.34) & (YPred_Z[:,0] < 0.22) & (df_Z.dijet_mass>xmin) & (df_Z.dijet_mass<xmax) & (df_Z.dijet_pt>100)
mass_Z = df_Z.dijet_mass[workingPoint_Z]
weights_Z = df_Z.weights[workingPoint_Z]

# %%
fig, ax = plt.subplots(1, 1)
x = (bins[1:] + bins[:-1])/2
for t in [120, 150, 200]:
    counts = np.histogram(mass[df.dijet_pt>t],  bins=bins)[0]
    errors=np.sqrt(counts)
    ax.errorbar(x, counts, xerr = np.diff(bins)/2,yerr=errors, linestyle='none', label='H pt > %d'%t)

    counts_Z = np.histogram(mass_Z[(df_Z[workingPoint_Z].dijet_pt>t)],  bins=bins, weights=weights_Z[(df_Z[workingPoint_Z].dijet_pt>t)])[0]
    ax.errorbar(x, counts_Z, xerr = np.diff(bins)/2, linestyle='none', label='Z pT > %d'%t)
ax.legend()
ax.set_yscale('log')




# %%
# Plotting in python
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
x = (bins[1:] + bins[:-1])/2
counts = np.histogram(mass,  bins=bins)[0]
errors = np.sqrt(counts)
integral = np.sum(counts * np.diff(bins))

counts_Z = np.histogram(mass_Z, bins=bins, weights=weights_Z)[0]
errors_Z = np.sqrt(np.histogram(mass_Z, bins=bins, weights=weights_Z**2)[0])

ax[0].errorbar(x, counts, xerr = np.diff(bins)/2,yerr=errors, color='black', linestyle='none')
ax[0].hist(bins[:-1], bins=bins, weights=counts_Z*10, histtype=u'step', color='red', label='Z MC x 10')[0]

ax[0].set_xlim(xmin, xmax)
ax[0].set_xlabel("Dijet Mass [GeV]")
ax[0].legend()
# Z MC
ax[1].errorbar(x, counts_Z, yerr=errors_Z, color='red', linestyle='none')
ax[1].set_xlim(xmin, xmax)
ax[1].set_xlabel("Dijet Mass [GeV]")








# %%
# Create the RooDataSet

x = RooRealVar("x", "mass", xmin, xmax)
dataset = RooDataSet("dataset", "dataset", RooArgSet(x))
#xblindLow, xblindHigh = 75, 105
for val in mass:
    x.setVal(val)
    dataset.add(RooArgSet(x))

#dataset = dataset.reduce(("x < %d || x > %d"%(88, 92)))
#x.setRange("fit_low", x.getMin(), xblindLow)#; // Range below the blind region
#x.setRange("fit_high", xblindHigh, x.getMax())#; 
# %%

p0 = RooRealVar("p0", "p0", 27698, 20000, 30000)
p1 = RooRealVar("p1", "p1", 10.29, -100, 100)
p2 = RooRealVar("p2", "p2", 4.14, -10, 10)
frac_exp1 = RooRealVar("n1", "n1", 0.5, 0, 1)
exp_coef_1 = RooRealVar("exp_coef_1", "exp_coef_1", -0.0078, -0.1, 0)
exp_coef_2 = RooRealVar("exp_coef_2", "exp_coef_2", -1, -10, 0)

# Define the exp*pol2 function
exp1 = RooGenericPdf("exp1", "exp((@0 - %d) * @1)*(@2 + (@0 - %d) * @3 + (@0 - %d) *(@0 - %d) * @4) "%( xmin, xmin, xmin, xmin), #@4 * (@0 - %d)*(@0 - %d)
                         RooArgList(x, exp_coef_1, p0, p1, p2))
# Fit the model to the data
fit_result = exp1.fitTo(dataset, RooFit.Save())



# %%
# Plotting
frame = x.frame(RooFit.Title("Exp + Exp"))
dataset.plotOn(frame)
exp1.plotOn(frame)
exp1.plotOn(frame, RooFit.Components("exp1"), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kRed))
exp1.plotOn(frame, RooFit.Components("exp2"), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kRed))

# Draw the plot
canvas = ROOT.TCanvas("c", "Fit", 800, 600)
frame.Draw()
canvas.Draw()




# %%
frame = x.frame()

# Plot the dataset on the frame
dataset.plotOn(frame)

# Plot the fitted function in the unblinded region (solid line)
#exp1.plotOn(frame, RooFit.Range("fit_low,fit_high"), RooFit.LineColor(ROOT.kBlue))

# Plot the fitted function in the blinded region with a dashed line

#expexp.plotOn(frame, RooFit.Range(xblindLow, xblindHigh), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kRed))

# Plot the fitted function over the full range (both blinded and unblinded)
exp1.plotOn(frame, RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kRed), RooFit.Range(x.getMin(), x.getMax()))

# Create a residual plot (residuals over all x, including blinded region)
residuals = frame.pullHist()  # Pull hist calculates residuals

# Create a new frame for plotting the residuals
residual_frame = x.frame()
residual_frame.addPlotable(residuals, "P")  # Plot residuals on the frame

c1 = ROOT.TCanvas("c1", "Fit and Residuals", 800, 600)

# Manually create two pads with a 3:1 ratio
# Upper pad takes 3/4 of the canvas height
upper_pad = ROOT.TPad("upper_pad", "upper_pad", 0, 0.25, 1, 1)
upper_pad.SetBottomMargin(0.02)  # Reduce the bottom margin for the upper pad
upper_pad.Draw()

# Lower pad takes 1/4 of the canvas height
lower_pad = ROOT.TPad("lower_pad", "lower_pad", 0, 0, 1, 0.25)
lower_pad.SetTopMargin(0.02)  # Reduce the top margin for the lower pad
lower_pad.SetBottomMargin(0.3)  # Increase the bottom margin for labels
lower_pad.Draw()

# Switch to the upper pad and draw the frame (fit and data)
upper_pad.cd()
frame.Draw()

# Switch to the lower pad and draw the residuals frame
lower_pad.cd()
residual_frame.Draw()

# Show the canvas
c1.Draw()







# %%

frac_exp1 = RooRealVar("frac_exp1", "frac_exp1", 0.9975, 0, 1)
mean = RooRealVar("mean", "mean",                       92.41070579278578, 86, 98)
sigma_left = RooRealVar("sigma_left", "sigma (left)",   11.196446927439963, 9, 20)
alpha_left = RooRealVar("alpha_left", "alpha (left)",   1.093064054092169, 1, 10)
n_left = RooRealVar("n_left", "n (left)",               0.2, 0.1, 30)

# Parameters for the right Crystal Ball
sigma_right = RooRealVar("sigma_right", "sigma (right)",    13.384701583913221, 9, 20)
alpha_right = RooRealVar("alpha_right", "alpha (right)",    -1.13, -10, -0.1)
n_right = RooRealVar("n_right", "n (right)",                0.62, 0.1, 30)

# Create the two Crystal Ball components
crystal_left = RooCBShape("crystal_left", "Crystal Ball (left)", x, mean, sigma_left, alpha_left, n_left)
crystal_right = RooCBShape("crystal_right", "Crystal Ball (right)", x, mean, sigma_right, alpha_right, n_right)

# Combine the two with a RooAddPdf, assuming equal fractions (0.5 for each side)
frac_cb = RooRealVar("frac_cb", "fraction of left CB", 0.38, 0.0, 1.0)
mean.setConstant(True)
#frac_exp1.setConstant(False)
sigma_left.setConstant(True)
alpha_left.setConstant(True)
n_left.setConstant(True)
sigma_right.setConstant(True)
alpha_right.setConstant(True)
n_right.setConstant(True)
frac_cb.setConstant(True)
double_cb = RooAddPdf("double_cb", "Double-sided Crystal Ball", RooArgList(crystal_left, crystal_right), RooArgList(frac_cb))


exp1DSCB = RooAddPdf("exexpDSCB", "Exp1DSCB", RooArgList(exp1, double_cb), RooArgList(frac_exp1))

# Fit the model to the data
fit_result = exp1DSCB.fitTo(dataset, RooFit.Save())




# %%
frame = x.frame(RooFit.Title("Fit of Exp*pol2 + Double-sided Crystal Ball"))
dataset.plotOn(frame)
exp1DSCB.plotOn(frame)
exp1DSCB.plotOn(frame, RooFit.Components("exp1"), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kRed))
exp1DSCB.plotOn(frame, RooFit.Components("double_cb"), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kRed))

# Draw the plot
c1 = ROOT.TCanvas("c1", "Fit and Residuals", 800, 600)

# Manually create two pads with a 3:1 ratio
# Upper pad takes 3/4 of the canvas height
upper_pad = ROOT.TPad("upper_pad", "upper_pad", 0, 0.25, 1, 1)
upper_pad.SetBottomMargin(0.02)  # Reduce the bottom margin for the upper pad
upper_pad.Draw()

# Lower pad takes 1/4 of the canvas height
lower_pad = ROOT.TPad("lower_pad", "lower_pad", 0, 0, 1, 0.25)
lower_pad.SetTopMargin(0.02)  # Reduce the top margin for the lower pad
lower_pad.SetBottomMargin(0.3)  # Increase the bottom margin for labels
lower_pad.Draw()

# Switch to the upper pad and draw the frame (fit and data)
upper_pad.cd()
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

# Switch to the lower pad and draw the residuals frame
lower_pad.cd()
residual_frame.Draw()

c1.Draw()


# %%





# Model Bkg + Z

mean = RooRealVar("mean", "mean",                       92.41070579278578, 86, 98)
sigma_left = RooRealVar("sigma_left", "sigma (left)",   14.196446927439963, 9, 20)
alpha_left = RooRealVar("alpha_left", "alpha (left)",   1.093064054092169, 1, 10)
n_left = RooRealVar("n_left", "n (left)",               0.9906014009107879, 0.1, 30)

# Parameters for the right Crystal Ball
sigma_right = RooRealVar("sigma_right", "sigma (right)",    14.384701583913221, 9, 20)
alpha_right = RooRealVar("alpha_right", "alpha (right)",    -1.094762381057086, -10, -0.1)
n_right = RooRealVar("n_right", "n (right)",                1.1553553992720837, 0.1, 30)

# Create the two Crystal Ball components
crystal_left = RooCBShape("crystal_left", "Crystal Ball (left)", x, mean, sigma_left, alpha_left, n_left)
crystal_right = RooCBShape("crystal_right", "Crystal Ball (right)", x, mean, sigma_right, alpha_right, n_right)

# Combine the two with a RooAddPdf, assuming equal fractions (0.5 for each side)
frac_cb = RooRealVar("frac_cb", "fraction of left CB", 0.44194038954109766, 0.0, 1.0)
mean.setConstant(True)
sigma_left.setConstant(True)
alpha_left.setConstant(True)
n_left.setConstant(True)
sigma_right.setConstant(True)
alpha_right.setConstant(True)
n_right.setConstant(True)
frac_cb.setConstant(True)
double_cb = RooAddPdf("double_cb", "Double-sided Crystal Ball", RooArgList(crystal_left, crystal_right), RooArgList(frac_cb))

p0 = RooRealVar("p0", "p0", 125*nReal, 125*nReal*0.8, 125*nReal*1.2)
p1 = RooRealVar("p1", "p1", -1, -10, 100)
p2 = RooRealVar("p2", "p2", -2, -10, 10)
exp_coef = RooRealVar("exp_coef", "exp_coef", -0.02, -0.05, 0)

# Define the exp*pol2 function
exp_pol2 = RooGenericPdf("exp_pol2", "exp((@0 - 70) * @1) * (@2 + @3 * (@0 - 70) + @4 * (@0 - 70)*(@0 - 70))", 
                         RooArgList(x, exp_coef, p0, p1, p2))
frac_signalBkg = RooRealVar("frac_signalBkg", "fraction of left CB", 0.0033, 0.0, 0.01)
frac_signalBkg.setConstant(True)
double_cb_bkg = RooAddPdf("double_cb_bkg", "Double-sided Crystal Ball + Bkg", RooArgList(double_cb, exp_pol2), RooArgList(frac_signalBkg))

# Fit the model to the dataset
fit_result = double_cb_bkg.fitTo(dataset)
# %%
frame = x.frame(RooFit.Title("Fit of Exp*pol1 + Double-sided Crystal Ball"))
dataset.plotOn(frame)
double_cb_bkg.plotOn(frame)
double_cb_bkg.plotOn(frame, RooFit.Components("exp_pol2"), RooFit.LineStyle(ROOT.kDashed))
double_cb_bkg.plotOn(frame, RooFit.Components("double_cb"), RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kRed))

# Draw the plot
canvas = ROOT.TCanvas("c", "Fit", 800, 600)
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
canvas.Draw()
canvas.SaveAs("/t3home/gcelotto/ggHbb/NN/workingPoint/pol4Fit.png")
# %%
