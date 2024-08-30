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
xmin = 40
xmax = 200
bins = np.linspace(xmin, xmax, 100)


def model_with_norm(x, norm, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc):
    return  norm * crystalball_ex.pdf(x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc)




    # Set parameter limits
    #m.limits["beta_left"] = (0,5)
    #m.limits["beta_right"] = (0,3)
    #m.limits["m_left"] = (0, 300)
    #m.limits["m_right"] = (10, 18)
    #m.limits["scale_left", "scale_right"] = (10, 30)
    #m.limits["loc"] = (80, 100)
    #m.limits["p0", "p1"] = (-1, 1)  # Adjust these limits based on expected ranges
    #m.limits["f"] = (0, 1)

# %%
# Load Data
print("Loading Data")
from loadData import loadData
dfs, YPred_Z100, YPred_Z200, YPred_Z400, YPred_Z600, YPred_Z800, numEventsList = loadData(pdgID=23)


dfs[0]['weights'] = 5.261e+03 / numEventsList[0] * dfs[0].sf
dfs[1]['weights'] = 1012. / numEventsList[1] * dfs[1].sf
dfs[2]['weights'] = 114.2 / numEventsList[2] * dfs[2].sf
dfs[3]['weights'] = 25.34 / numEventsList[3] * dfs[3].sf
dfs[4]['weights'] = 12.99 / numEventsList[4] * dfs[4].sf

df = pd.concat(dfs)
YPred = np.concatenate([YPred_Z100, YPred_Z200, YPred_Z400, YPred_Z600, YPred_Z800])

# %%
print("Setting xmin, xmax, WP")

workingPoint = (YPred[:,1] > 0.34) & (YPred[:,0] < 0.22)
weights = df.weights[workingPoint]
mass = df.dijet_mass[workingPoint]

# %%
beta_left, m_left, scale_left = 2.1, 30, 15
beta_right, m_right, scale_right = 1.4, 12, 14
loc = 93
p0, p1 = 0.033, 0,

counts = np.histogram(mass,  bins=bins, weights=weights)[0]
integral = np.sum(counts * np.diff(bins))
errors = np.sqrt(np.histogram(mass,  bins=bins, weights=weights**2)[0])


x = (bins[1:] + bins[:-1])/2
least_squares = LeastSquares(x, counts, errors, model_with_norm)
m = Minuit(least_squares, integral, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc)
m.migrad()  # finds minimum of least_squares function
m.hesse() 
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
ax.set_ylabel("Counts [a.u.]")

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
ax2.set_ylim(0.6, 1.4)
ax2.set_xlabel("Dijet Mass [GeV]")
ax2.hlines(xmin=xmin, xmax=xmax, y=1, color='black')
hep.cms.label(ax=ax)
ax2.set_ylabel("Ratio Data / Fit")
ax.yaxis.set_label_coords(-0.1, 1)
ax2.yaxis.set_label_coords(-0.1, 1)


# %%
# Save in JSON
# Example best-fit parameters and their errors (replace with your actual results)
best_fit_params = {
    'norm':m.values['norm'], 
    'beta_left':m.values['beta_left'], 
    'm_left':m.values['m_left'],
    'scale_left':m.values['scale_left'],  
    'beta_right':m.values['beta_right'],
    'm_right':m.values['m_right'], 
    'scale_right':m.values['scale_right'],
    'loc':m.values['loc']

}
best_fit_errors = {
    'norm_err':m.errors['norm'], 
    'beta_left_err':m.errors['beta_left'], 
    'm_left_err':m.errors['m_left'],
    'scale_left_err':m.errors['scale_left'],  
    'beta_right_err':m.errors['beta_right'],
    'm_right_err':m.errors['m_right'], 
    'scale_right_err':m.errors['scale_right'],
    'loc_err':m.errors['loc']
}



# Combine parameters and errors into a single dictionary
fit_results = {}
for key in best_fit_params:
    fit_results[key] = {'value': best_fit_params[key], 'error': best_fit_errors[key + '_err']}

import json

# Save to a JSON file
with open('/t3home/gcelotto/ggHbb/NN/workingPoint/fit_results.json', 'w') as f:
    json.dump(fit_results, f, indent=4)

# %%
