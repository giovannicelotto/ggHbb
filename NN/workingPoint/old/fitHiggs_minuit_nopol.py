# %%
#from fitFunction import DoubleSidedCrystalballFunctionPDF_normalized
import glob, re, sys
sys.path.append('/t3home/gcelotto/ggHbb/NN')
import numpy as np
import pandas as pd
from functions import loadMultiParquet, cut, getXSectionBR
from helpersForNN import preprocessMultiClass, scale, unscale
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
import mplhep as hep
hep.style.use("CMS")
import math
from applyMultiClass_Hpeak import getPredictions, splitPtFunc
from iminuit import cost
from iminuit import Minuit
from numba_stats import crystalball_ex
# %%

print("Setting xmin, xmax, WP")
xmin = 60
xmax = 180
bins=np.linspace(xmin, xmax, 50)



# Define the cost function
def combined_cost(xsf, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc):
    x, sf = xsf
    return crystalball_ex.pdf(x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc) ** sf

# Fit the data using the combined PDF
def fit_with_combined_pdf(mass, weights, truth):
    costFunc = cost.UnbinnedNLL((mass, weights), combined_cost)
    
    
    # Create the Minuit object
    m = Minuit(costFunc, *truth)
    
    # Set parameter limits
    m.limits["beta_left", "beta_right"] = (1, 5)
    m.limits["m_left"] = (1, 250)
    m.limits["m_right"] = (1, 100)
    m.limits["scale_left", "scale_right"] = (10, 30)
    m.limits["loc"] = (122, 126)
    m.migrad()  # finds minimum of least_squares function
    m.hesse()   # accurately computes uncertainties

    
    
    # Output results
    print(m.params)
    return m

#
# load Data
#
print("Loading Data")
from loadData import loadData
dfs, YPred_H = loadData()
workingPoint = (YPred_H[:,1]>0.34) & (YPred_H[:,0]<0.22)
weights = np.ones(len(dfs[0]))[workingPoint]#dfs[0].sf[workingPoint] #* dfs[0].PU_SF[workingPoint]
mass = dfs[0].dijet_mass[workingPoint]



# %%
print("Fit started")
truth = (1.213,   137, 20,
             1.54,   10.7, 14.88,
             123.65 )
m = fit_with_combined_pdf(mass, weights, truth)




# %%
print(m.params) # parameter info (using str(m.params))
print(m.covariance)
#m.fval/(len(dfs[0][workingPoint])-7)


# %%
# Plotting
fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)


x = (bins[:-1]+bins[1:])/2
c = np.histogram(mass, bins=bins,weights=weights)[0]
integral = np.sum(c*np.diff(bins))
counts = ax.hist(bins[:-1], bins=bins, weights=c, density=True, histtype=u'step', color='black', label='Binned Data')[0]
ax.errorbar(x, c/integral, yerr=np.sqrt(c)/integral, color='black', linestyle='none')

ax.set_xlim(xmin, xmax)
pdf_values = crystalball_ex.pdf(x, m.values['beta_left'], m.values['m_left'], m.values['scale_left'], m.values['beta_right'], m.values['m_right'], m.values['scale_right'], m.values['loc'])
cdf_values = crystalball_ex.cdf(x, m.values['beta_left'], m.values['m_left'], m.values['scale_left'], m.values['beta_right'], m.values['m_right'], m.values['scale_right'], m.values['loc'])
integral_in_region = cdf_values[-1] - cdf_values[0]
ax.plot(x, pdf_values/integral_in_region, label='Unbinned Fit')
ax.set_ylabel("Normalized Events")

# ********************
#      KDE
# ********************

#from scipy.stats import gaussian_kde
#kde = gaussian_kde(mass, weights = weights)
#x_range = np.linspace(xmin, xmax, 1000)
#kde_values = kde(x_range)
#ax.plot(x_range, kde_values, label='KDE', color='red')
ax.legend()

ax2.errorbar(x, (counts)/crystalball_ex.pdf(x, m.values['beta_left'], m.values['m_left'],
                               m.values['scale_left'],  m.values['beta_right'],
                               m.values['m_right'], m.values['scale_right'],
                               m.values['loc']) ,yerr= np.sqrt(c)/integral,
                               color='black',
                               marker='o', linestyle='none')
ax2.set_ylim(0.5, 1.5)
ax2.set_xlabel("Dijet Mass [GeV]")
ax2.hlines(xmin=xmin, xmax=xmax, y=1, color='black')
hep.cms.label(ax=ax)
ax2.set_ylabel("Ratio Data / Fit")
ax.yaxis.set_label_coords(-0.1, 1)
ax2.yaxis.set_label_coords(-0.1, 1)

# %%