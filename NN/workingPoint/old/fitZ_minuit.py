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
xmin = 30
xmax = 180
bins=np.linspace(xmin, xmax, 100)
def binned_pdf(bin_centers, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc):
                return crystalball_ex.pdf(bin_centers, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc)

def pdf(xsf, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc):
                x, sf = xsf
                return crystalball_ex.pdf(x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc)**sf
# %%
def unbinnedFit(mass, weights):
        print("Fit started")
        
        costFunc = cost.UnbinnedNLL((mass, weights), pdf)
        truth = (1.2, 28, 19, 1.6, 8.01, 14, 90)
        m = Minuit(costFunc, *truth)
        m.limits["beta_left", "beta_right"] = (0, 3)
        m.limits["m_left"] = (0, 300)
        m.limits["m_right"] = (0, 20)
        m.limits["scale_left", "scale_right"] = (10, 30)
        m.limits["loc"] = (80, 100)
        m.migrad()  # finds minimum of least_squares function
        m.hesse()   # accurately computes uncertainties

        print(m.params) # parameter info (using str(m.params))
        print(m.covariance)
        m.fval/(len(df[workingPoint])-7)
        return m


def binnedFit(mass, weights):
        # Create a binned histogram from the data
        from iminuit.cost import LeastSquares
        hist, bin_edges = np.histogram(mass, bins=bins, weights=weights)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])  # Calculate bin centers
        hist = hist/np.sum(hist)
        least_squares = LeastSquares(bin_centers, hist, np.ones(len(hist))/100, binned_pdf)
        truth = (1.5, 28, 16, 0.7, 9, 14, 90)

        m = Minuit(least_squares, *truth)
        m.limits["beta_left", "beta_right"] = (0, 2)
        m.limits["m_left"] = (0, 300)
        m.limits["m_right"] = (0, 20)
        m.limits["scale_left", "scale_right"] = (10, 30)
        m.limits["loc"] = (80, 100)

        # Perform the minimization
        m.migrad()  # Minimize the cost function
        m.hesse()   # Compute uncertainties

        # Output the results
        print(m.values[:])  # Fitted parameters
        print(m.errors[:])  # Uncertainties on fitted parameters
        return m



# %%

# load Data
#
print("Loading Data")
from loadData import loadData
dfs, YPred_Z100, YPred_Z200, YPred_Z400, YPred_Z600, YPred_Z800, numEventsList = loadData(pdgID=23)

# %%
dfs[0]['weights'] = 5.261e+03/numEventsList[0]*dfs[0].sf
dfs[1]['weights'] = 1012./numEventsList[1]*dfs[1].sf
dfs[2]['weights'] = 114.2/numEventsList[2]*dfs[2].sf
dfs[3]['weights'] = 25.34/numEventsList[3]*dfs[3].sf
dfs[4]['weights'] = 12.99/numEventsList[4]*dfs[4].sf


df = pd.concat(dfs)
YPred = np.concatenate([YPred_Z100, YPred_Z200, YPred_Z400, YPred_Z600, YPred_Z800])
# %%
print("Setting xmin, xmax, WP")

workingPoint = (YPred[:,1]>0.34) & (YPred[:,0]<0.22)
#print(len(df))
#print(len(workingPoint))
weights = df.weights[workingPoint]#dfs[0].sf[workingPoint] * dfs[0].PU_SF[workingPoint]
mass = df.dijet_mass[workingPoint]
# %%


mb = binnedFit(mass, weights)
m = unbinnedFit(mass, weights)




# %%
# Plotting
fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)


x = (bins[:-1]+bins[1:])/2
c = np.histogram(mass, bins=bins,weights=weights)[0]
integral = np.sum(c*np.diff(bins))
# Binned data
ax.hist(bins[:-1], bins=bins, weights=c, density=True, histtype=u'step', color='black', label='Binned Data')
ax.errorbar(x, c/integral, yerr=None, color='black', linestyle='none')

# Unbined Fit
ax.set_xlim(xmin, xmax)
ax.plot(x, crystalball_ex.pdf(x, m.values['beta_left'], m.values['m_left'],
                               m.values['scale_left'],  m.values['beta_right'],
                               m.values['m_right'], m.values['scale_right'],
                               m.values['loc']), label='Unbinned Fit')
# Binned fit
bins_centers = (bins[1:]+bins[:-1])/2
ax.plot(bins_centers, crystalball_ex.pdf(bins_centers, mb.values['beta_left'], mb.values['m_left'],
                               mb.values['scale_left'],  mb.values['beta_right'],
                               mb.values['m_right'], mb.values['scale_right'],
                               m.values['loc']), label='Binned Fit')
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
print(mb.values[:])



# %%
