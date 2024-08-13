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

#
# load Data
#
print("Loading Data")
from loadData import loadData
dfs, YPred_Z= loadData(pdgID=23)
# %%
YPred_Z = np.concatenate([YPred_Z200, YPred_Z400, YPred_Z600, YPred_Z800])
# %%
# %%
df = pd.concat(dfs)
# %%
print("Setting xmin, xmax, WP")
xmin = 60
xmax = 180
workingPoint = (YPred_H[:,1]>0.34) & (YPred_H[:,0]<0.22)
weights = np.ones(len(dfs[0]))[workingPoint]#dfs[0].sf[workingPoint] * dfs[0].PU_SF[workingPoint]
mass = dfs[0].dijet_mass[workingPoint]


# %%
print("Fit started")
def pdf(xsf, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc):
        x, sf = xsf
        return crystalball_ex.pdf(x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc)**sf

costFunc = cost.UnbinnedNLL((mass, weights), pdf)
truth = (1.2, 28, 19, 1.6, 8.01, 14, 124)
m = Minuit(costFunc, *truth)
m.limits["beta_left", "beta_right"] = (0, 3)
m.limits["m_left"] = (0, 300)
m.limits["m_right"] = (0, 20)
m.limits["scale_left", "scale_right"] = (10, 30)
m.limits["loc"] = (115, 130)
m.migrad()  # finds minimum of least_squares function
m.hesse()   # accurately computes uncertainties

# %%
print(m.params) # parameter info (using str(m.params))
print(m.covariance)
m.fval/(len(dfs[0][workingPoint])-7)


# %%
# Plotting
fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
bins=np.linspace(xmin, xmax, 100)

x = (bins[:-1]+bins[1:])/2
c = np.histogram(mass, bins=bins,weights=weights)[0]
integral = np.sum(c*np.diff(bins))

ax.hist(bins[:-1], bins=bins, weights=c, density=True, histtype=u'step', color='black', label='Binned Data')
ax.errorbar(x, c/integral, yerr=np.sqrt(c)/integral, color='black', linestyle='none')
#import hist
#h = hist.Hist(hist.axis.Regular(100, xmin, xmax))
#h.fill(mass)
#h.plot(ax=ax, density=True, color='black', label='Binned Data')
ax.set_xlim(xmin, xmax)
ax.plot(x, crystalball_ex.pdf(x, m.values['beta_left'], m.values['m_left'],
                               m.values['scale_left'],  m.values['beta_right'],
                               m.values['m_right'], m.values['scale_right'],
                               m.values['loc']), label='Unbinned Fit')
ax.set_ylabel("Normalized Events")
from scipy.stats import gaussian_kde
kde = gaussian_kde(mass, weights = weights)
x_range = np.linspace(xmin, xmax, 1000)
kde_values = kde(x_range)
ax.plot(x_range, kde_values, label='KDE', color='red')
ax.legend()

ax2.plot(x_range, crystalball_ex.pdf(x_range, m.values['beta_left'], m.values['m_left'],
                               m.values['scale_left'],  m.values['beta_right'],
                               m.values['m_right'], m.values['scale_right'],
                               m.values['loc']) / kde_values, color='black')
ax2.set_ylim(0.5, 1.5)
ax2.set_xlabel("Dijet Mass [GeV]")
ax2.hlines(xmin=xmin, xmax=xmax, y=1, color='black')

ax2.set_ylabel("Ratio Fit / KDE")
ax.yaxis.set_label_coords(-0.1, 1)
ax2.yaxis.set_label_coords(-0.1, 1)
# %%
#print("Compute chi2 in a binned histogram")
#yval_fit = crystalball_ex.pdf(bins, m.values['beta_left'], m.values['m_left'],
#                               m.values['scale_left'],  m.values['beta_right'],
#                               m.values['m_right'], m.values['scale_right'],
#                               m.values['loc'])
#yval_hist = h.view()
#
#chi2 = np.sum(((yval_fit - yval_hist)**2/yval_hist))



# %%
#likelihood_fit = pdf((list(mass), weights), m.values['beta_left'], m.values['m_left'],
#                               m.values['scale_left'],  m.values['beta_right'],
#                               m.values['m_right'], m.values['scale_right'],
#                               m.values['loc']) 
#
#
#
#from scipy.interpolate import interp1d
#import time
#x_fine = np.linspace(xmin, xmax, 1000)
#kde_values_fine = kde(x_fine)
#
## Create an interpolation function
#kde_interp = interp1d(x_fine, kde_values_fine, bounds_error=False, fill_value="extrapolate")
#
## Apply the interpolation function to the mass data points
#start_time = time.time()
#likelihood_kde = kde_interp(mass)
#end_time = time.time()
