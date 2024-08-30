# %%
import glob, re, sys
sys.path.append('/t3home/gcelotto/ggHbb/NN')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
from iminuit import cost
from iminuit import Minuit
from numba_stats import crystalball_ex
from iminuit.cost import LeastSquares
# %%

xmin = 60
xmax = 180
bins=np.linspace(xmin, xmax, 50)




# New model with a floating normalization parameter
def model_with_norm(x, norm, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc):
    return norm * crystalball_ex.pdf(x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc)

#
# load Data
#
print("Loading Data")
from loadData import loadData
dfs, YPred_H = loadData()
workingPoint = (YPred_H[:,1]>0.34) & (YPred_H[:,0]<0.22)
weights = dfs[0].sf[workingPoint] #* dfs[0].PU_SF[workingPoint]
mass = dfs[0].dijet_mass[workingPoint]



# %%
beta_left, m_left, scale_left = (1.213,   137, 20)
beta_right, m_right, scale_right = (1.54,   9, 13.9)
loc = 125

counts = np.histogram(mass,  bins=bins, weights=weights)[0]
integral = np.sum(counts * np.diff(bins))
errors = np.sqrt(np.histogram(mass,  bins=bins, weights=weights**2)[0])


x = (bins[1:] + bins[:-1])/2
least_squares = LeastSquares(x, counts, errors, model_with_norm)
m = Minuit(least_squares, integral, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc)
m.migrad()  # finds minimum of least_squares function
m.hesse() 




# %%
print(m.params) # parameter info (using str(m.params))
print(m.covariance)
#m.fval/(len(dfs[0][workingPoint])-7)


# %%
# Plotting
fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
counts = ax.hist(bins[:-1], bins=bins, weights=counts, histtype=u'step', color='black', label='Binned Data')[0]
ax.errorbar(x, counts, yerr=errors, color='black', linestyle='none')

ax.set_xlim(xmin, xmax)
fit_values = m.values['norm']*crystalball_ex.pdf(x, m.values['beta_left'], m.values['m_left'],
                               m.values['scale_left'],  m.values['beta_right'],
                               m.values['m_right'], m.values['scale_right'],
                               m.values['loc'])
ax.plot(x, fit_values, label='Binned Fit')
ax.set_ylabel("Counts [a.u]")

# ********************
#      KDE
# ********************

#from scipy.stats import gaussian_kde
#kde = gaussian_kde(mass, weights = weights)
#x_range = np.linspace(xmin, xmax, 1000)
#kde_values = kde(x_range)
#ax.plot(x_range, kde_values, label='KDE', color='red')
ax.legend()
chi2= m.fval
ndof = (len(counts) - len(m.values))
ax.text(x=0.95, y=0.7, s=r"$\chi^2/n_\text{dof}$ = %.1f/%d"%(chi2, ndof), transform=ax.transAxes, ha='right')

ax2.errorbar(x, counts/fit_values, yerr= errors/fit_values,
                               color='black',
                               markersize=2,
                               marker='o', linestyle='none')
ax2.set_ylim(0.9, 1.1)
ax2.set_xlabel("Dijet Mass [GeV]")
ax2.hlines(xmin=xmin, xmax=xmax, y=1, color='black')
hep.cms.label(ax=ax)
ax2.set_ylabel("Ratio Data / Fit")
ax.yaxis.set_label_coords(-0.1, 1)
ax2.yaxis.set_label_coords(-0.1, 1)

# %%