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
from numba_stats import crystalball_ex, expon
from iminuit.cost import LeastSquares
import json
from loadData import loadData

# %%
xmin = 40
xmax = 200
bins = np.linspace(xmin, xmax, 50)
nReal = 100
def model_Z(x, norm_Z_MC, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc):
    values = norm_Z_MC* crystalball_ex.pdf(x, beta_left=beta_left, m_left=m_left, scale_left=scale_left, beta_right=beta_right, m_right=m_right, scale_right=scale_right, loc=loc)
    return values

def model_data(x, p0, p1, p2, p3, p4, norm_Z_fullFit, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc):
    values = p0 + p1*x + p2*x**2 +p3*x**3 +p4*x**4 + norm_Z_fullFit* crystalball_ex.pdf(x, beta_left=beta_left, m_left=m_left, scale_left=scale_left, beta_right=beta_right, m_right=m_right, scale_right=scale_right, loc=loc)
    return values

def getGradient(x, norm, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc):
    grad = []
    # df/d p0
    grad.append(np.ones(len(x)))

    # df/d p1
    grad.append(x)

    # df/d p2
    grad.append(x**2)

    # df/d p3
    grad.append(x**3)

    # df/d p4
    grad.append(x**4)

    # df/d norm 
    grad.append(crystalball_ex.pdf(x,
                                beta_left=beta_left, m_left=m_left, scale_left=scale_left, beta_right=beta_right, m_right=m_right, scale_right=scale_right, loc=loc))

    # df/d B_L
    epsilon = 1e-7
    var = crystalball_ex.pdf(x, beta_left=beta_left+ epsilon, m_left=m_left, scale_left=scale_left, beta_right=beta_right, m_right=m_right, scale_right=scale_right, loc=loc)
    nom = crystalball_ex.pdf(x, beta_left=beta_left, m_left=m_left, scale_left=scale_left, beta_right=beta_right, m_right=m_right, scale_right=scale_right, loc=loc)
    grad.append(norm * (var - nom)/epsilon)

    # df/d m_L
    var = crystalball_ex.pdf(x, beta_left=beta_left, m_left=m_left+epsilon, scale_left=scale_left, beta_right=beta_right, m_right=m_right, scale_right=scale_right, loc=loc)
    grad.append(norm * (var - nom)/epsilon)

    # df/d sigma_L
    var = crystalball_ex.pdf(x, beta_left=beta_left, m_left=m_left, scale_left=scale_left+epsilon, beta_right=beta_right, m_right=m_right, scale_right=scale_right, loc=loc)
    grad.append(norm * (var - nom)/epsilon)

    # df/d B_R
    var = crystalball_ex.pdf(x, beta_left=beta_left, m_left=m_left, scale_left=scale_left, beta_right=beta_right+epsilon, m_right=m_right, scale_right=scale_right, loc=loc)
    grad.append(norm * (var - nom)/epsilon)

    # df/d m_R
    var = crystalball_ex.pdf(x, beta_left=beta_left, m_left=m_left, scale_left=scale_left, beta_right=beta_right, m_right=m_right+epsilon, scale_right=scale_right, loc=loc)
    grad.append(norm * (var - nom)/epsilon)

    # df/d sigma_R
    var = crystalball_ex.pdf(x, beta_left=beta_left, m_left=m_left, scale_left=scale_left, beta_right=beta_right, m_right=m_right, scale_right=scale_right+epsilon, loc=loc)
    grad.append(norm * (var - nom)/epsilon)

    # df/d mu
    var = crystalball_ex.pdf(x, beta_left=beta_left, m_left=m_left, scale_left=scale_left, beta_right=beta_right, m_right=m_right, scale_right=scale_right, loc=loc+epsilon)
    grad.append(norm * (var - nom)/epsilon)
    
    return grad

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

workingPoint = (YPred_QCD[:,1] > 0.34) & (YPred_QCD[:,0] < 0.22) 
mass = df.dijet_mass[workingPoint]

workingPoint_Z = (YPred_Z[:,1] > 0.34) & (YPred_Z[:,0] < 0.22) 
mass_Z = df_Z.dijet_mass[workingPoint_Z]
weights_Z = df_Z.weights[workingPoint_Z]
# %%
fig, ax = plt.subplots(1, 1)
x = (bins[1:] + bins[:-1])/2
for t in [20, 50, 80, 100,120]:
    counts = np.histogram(mass[df.dijet_pt>t],  bins=bins)[0]
    errors=np.sqrt(counts)
    ax.errorbar(x, counts, xerr = np.diff(bins)/2,yerr=errors, linestyle='none', label='H pt > %d'%t)
ax.legend()
# %%
workingPoint = (YPred_QCD[:,1] > 0.34) & (YPred_QCD[:,0] < 0.22) & (df.dijet_pt>100)
mass = df.dijet_mass[workingPoint]

workingPoint_Z = (YPred_Z[:,1] > 0.34) & (YPred_Z[:,0] < 0.22) & (df_Z.dijet_pt>100)
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

ax[0].errorbar(x, counts, xerr = np.diff(bins)/2,yerr=errors, color='black', linestyle='none', label='no pt cut')
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



def getXCountsErrors():
    return x, counts, counts_Z, errors, errors_Z 
def combined_least_squares(p0, p1, p2, p3, p4, norm_Z_fullFit, norm_Z_MC, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc):

    x, counts_data, counts_Z, errors_data, errors_Z = getXCountsErrors()
    model_data_values = model_data(x, p0, p1, p2, p3, p4, norm_Z_fullFit, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc)
    model_Z_values = model_Z(x, norm_Z_MC, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc)

    residuals_data = (counts_data - model_data_values) / errors_data
    residuals_Z = (counts_Z - model_Z_values) / errors_Z
    print("Residuals Data :", np.sum(residuals_data))
    print("Residuals Z :", np.sum(residuals_Z))
    return np.sum(residuals_data**2) + np.sum(residuals_Z**2)
m = Minuit(combined_least_squares, **initial_guess)
m.limits['norm_Z_MC'] = (0,None)
m.limits['norm_Z_fullFit'] = (0,None)
m.limits['loc'] = (88,96)
m.limits['scale_left'] = (9,20)
m.limits['scale_right'] = (9,20)
m.limits['beta_left'] = (0.8,4)
m.limits['beta_right'] = (0.8,4)
#m.limits['m_left'] = (1,8)
#m.limits['m_right'] = (1,8)
m.migrad()  # finds minimum of least_squares function
m.hesse() 

# Print the results
print(m.values)  # Optimal parameters
print(m.errors)  # Uncertainties in the parameters

# %%
fig, ax = plt.subplots(1, 2, figsize=(24, 10))
x = (bins[1:] + bins[:-1])/2
# Comment because already defined
#counts = np.histogram(mass,  bins=bins)[0]
#errors = np.sqrt(counts)
#integral = np.sum(counts * np.diff(bins))

#counts_Z = np.histogram(mass_Z, bins=bins, weights=weights_Z)[0]
#errors_Z = np.sqrt(np.histogram(mass_Z, bins=bins, weights=weights_Z**2)[0])

ax[0].errorbar(x, counts, xerr = np.diff(bins)/2,yerr=errors, color='black', linestyle='none', label='Data')
ax[0].hist(bins[:-1], bins=bins, weights=counts_Z*10, histtype=u'step', color='red', label='Z MC')[0]

# Get the values extracted from the models (full fit, z only, bkg only)
model_data_values = model_data(x, m.values['p0'], m.values['p1'], m.values['p2'], m.values['p3'], m.values['p4'], m.values['norm_Z_fullFit'], m.values['beta_left'], m.values['m_left'], m.values['scale_left'], m.values['beta_right'], m.values['m_right'], m.values['scale_right'], m.values['loc'])
model_Z_values = model_Z(x, m.values['norm_Z_fullFit'], m.values['beta_left'], m.values['m_left'], m.values['scale_left'], m.values['beta_right'], m.values['m_right'], m.values['scale_right'], m.values['loc'])
model_bkg_values = m.values['p0'] + m.values['p1'] * x + m.values['p2'] * x**2 + m.values['p3'] *x**3 + m.values['p4']*x**4
    
#GCax[0].plot(x, model_bkg_values, color='gray', label='Bkg only')
#GCax[0].plot(x, model_data_values, label='Full fit')
#GCax[0].plot(x, model_Z_values*10, label='Z Fit (f1)')

# Uncertainties are computed excluding the row and column of norm_Z_MC which does not enter in the function plotted
# the corresponding gradient is zero
gradient = np.array(getGradient(x, m.values['norm_Z_fullFit'], m.values['beta_left'], m.values['m_left'], m.values['scale_left'], m.values['beta_right'], m.values['m_right'], m.values['scale_right'], m.values['loc']))
cov = np.array(m.covariance)
mask = np.ones(cov.shape, dtype=bool)
mask[6, :] = False
mask[:, 6] = False
cov = cov[mask].reshape(13, 13)
sigma_f = []
for i in range(len(x)):
    sigma_f.append(np.sqrt(np.dot(np.dot(gradient.T[i,:],cov),gradient[:,i])))
sigma_f = np.array(sigma_f)
#GCax[0].fill_between(x, model_Z_values*10-sigma_f*10, model_Z_values*10+sigma_f*10, color='C1', alpha=0.3, label='Zbb Fit ±1σ')

# Plot the chi2 of simultaneous fit
chi2= m.fval
ndof = (len(counts) + len(counts_Z) - len(m.values))
#GCax[0].text(x=0.95, y=0.7, s=r"$\chi^2/n_\text{dof}$ = %.1f/%d"%(chi2, ndof), transform=ax[0].transAxes, ha='right')

ax[0].set_xlim(xmin, xmax)
ax[0].set_xlabel("Dijet Mass [GeV]")
ax[0].legend()

# Second Pad
# Z MC
# Get the fit values with normalization of MC sample
model_Z_values_MC = model_Z(x, m.values['norm_Z_MC'], m.values['beta_left'], m.values['m_left'], m.values['scale_left'], m.values['beta_right'], m.values['m_right'], m.values['scale_right'], m.values['loc'])
#GCax[1].plot(x, model_Z_values_MC, color='green', label='f2')
ax[1].errorbar(x, counts_Z, xerr= np.diff(bins)/2, yerr=errors_Z, color='black', linestyle='none', label='Z MC')

# Uncertainties are computed excluding the row and column of norm_fullFit which does not enter in the function plotted
# the corresponding gradient is zero
gradient = np.array(getGradient(x, m.values['norm_Z_MC'], m.values['beta_left'], m.values['m_left'], m.values['scale_left'], m.values['beta_right'], m.values['m_right'], m.values['scale_right'], m.values['loc']))
cov = np.array(m.covariance)
mask = np.ones(cov.shape, dtype=bool)
mask[5, :] = False
mask[:, 5] = False
cov = cov[mask].reshape(13, 13)
sigma_f = []
for i in range(len(x)):
    sigma_f.append(np.sqrt(np.dot(np.dot(gradient.T[i,:],cov),gradient[:,i])))
sigma_f = np.array(sigma_f)
#GCax[1].fill_between(x, model_Z_values_MC-sigma_f, model_Z_values_MC+sigma_f, alpha=0.3, label='Zbb Fit ±1σ', color='green')


ax[1].set_xlim(xmin, xmax)
ax[1].set_xlabel("Dijet Mass [GeV]")
#GCax[1].text(x=0.95, y=0.7, s=r"$\chi^2/n_\text{dof}$ = %.1f/%d"%(chi2, ndof), transform=ax[1].transAxes, ha='right')
ax[1].legend()

# %%
fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
x = (bins[1:] + bins[:-1])/2
counts = np.histogram(mass,  bins=bins)[0]
errors = np.sqrt(counts)
integral = np.sum(counts * np.diff(bins))

counts_Z = np.histogram(mass_Z, bins=bins, weights=weights_Z)[0]
errors_Z = np.sqrt(np.histogram(mass_Z, bins=bins, weights=weights_Z**2)[0])

ax.errorbar(x, counts, xerr = np.diff(bins)/2,yerr=errors, color='black', linestyle='none')
ax.hist(bins[:-1], bins=bins, weights=counts_Z*10, histtype=u'step', color='red', label='Z MC')[0]
ax.set_xlim(xmin, xmax)

    
ax.plot(x, model_bkg_values, color='gray', label='Bkg only')
ax.plot(x, model_data_values, label='Full fit')
ax.plot(x, model_Z_values*10, label='Z Fit x 10')

gradient = np.array(getGradient(x, m.values['norm_Z_fullFit'], m.values['beta_left'], m.values['m_left'], m.values['scale_left'], m.values['beta_right'], m.values['m_right'], m.values['scale_right'], m.values['loc']))
cov = np.array(m.covariance)
mask = np.ones(cov.shape, dtype=bool)
mask[6, :] = False
mask[:, 6] = False
cov = cov[mask].reshape(13, 13)
sigma_f = []
for i in range(len(x)):
    sigma_f.append(np.sqrt(np.dot(np.dot(gradient.T[i,:],cov),gradient[:,i])))
sigma_f = np.array(sigma_f)
ax.fill_between(x, model_Z_values*10-sigma_f*10, model_Z_values*10+sigma_f*10, color='C1', alpha=0.3, label='(Zbb Fit ±1σ)x10')

chi2= m.fval
ndof = (len(counts) + len(counts_Z) - len(m.values))
ax.text(x=0.95, y=0.5, s=r"$\chi^2/n_\text{dof}$ = %.1f/%d"%(chi2, ndof), transform=ax.transAxes, ha='right')
ax.legend()
hep.cms.label(ax=ax)
ax2.errorbar(x, (counts-model_bkg_values), xerr=np.diff(bins)/2,
             yerr= errors,  color='black', markersize=2, marker='o', linestyle='none', label = 'Data')
ax2.plot(x, model_Z_values, color='C1', label='Zbb Fit Extracted')
ax2.fill_between(x, model_Z_values-sigma_f, model_Z_values+sigma_f, color='C1', alpha=0.3, label='(Zbb Fit ±1σ)x10')
ax2.hist(bins[:-1], bins, weights=counts_Z, color='red', label='Z MC', histtype=u'step')
ax2.set_ylim(-1e3, 1e3)
ax2.set_xlabel("Dijet Mass [GeV]")
ax2.hlines(xmin=xmin, xmax=xmax, y=1, color='black')

ax2.set_ylabel("Data - Bkg")
ax.yaxis.set_label_coords(-0.1, 1)
ax2.yaxis.set_label_coords(-0.1, 1)










# %%
























# Plot for guessing:
fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
x = (bins[1:] + bins[:-1])/2
counts = np.histogram(mass,  bins=bins)[0]
errors = np.sqrt(counts)
integral = np.sum(counts * np.diff(bins))

counts_Z = np.histogram(mass_Z, bins=bins, weights=weights_Z)[0]
errors_Z = np.sqrt(np.histogram(mass_Z, bins=bins, weights=weights_Z**2)[0])

ax.errorbar(x, counts, xerr = np.diff(bins)/2,yerr=errors, color='black', linestyle='none')
ax.hist(bins[:-1], bins=bins, weights=counts_Z*10, histtype=u'step', color='red', label='Z MC')[0]
ax.set_xlim(xmin, xmax)
ax.set_xlabel("Dijet Mass [GeV]")

model_data_values_guess =  model_data(x, m.values['p0'], m.values['p1'], m.values['p2'], m.values['p3'], m.values['p4'], m.values['norm_Z_MC'], m.values['beta_left'], m.values['m_left'], m.values['scale_left'], m.values['beta_right'], m.values['m_right'], m.values['scale_right'], m.values['loc'])
model_Z_values_guess = model_Z(x, m.values['norm_Z_MC'], m.values['beta_left'], m.values['m_left'], m.values['scale_left'], m.values['beta_right'], m.values['m_right'], m.values['scale_right'], m.values['loc'])


ax.plot(x, model_data_values_guess, label='Full fit')
ax.plot(x, model_Z_values_guess*10, label='Z Fit x 10')

#gradient = np.array(getGradient(x, m.values['norm_Z_fullFit'], m.values['beta_left'], m.values['m_left'], m.values['scale_left'], m.values['beta_right'], m.values['m_right'], m.values['scale_right'], m.values['loc']))
#cov = np.array(m.covariance)
#mask = np.ones(cov.shape, dtype=bool)
#mask[6, :] = False
#mask[:, 6] = False
#cov = cov[mask].reshape(13, 13)
#sigma_f = []
#for i in range(len(x)):
#    sigma_f.append(np.sqrt(np.dot(np.dot(gradient.T[i,:],cov),gradient[:,i])))
#sigma_f = np.array(sigma_f)
#ax.fill_between(x, model_Z_values*10-sigma_f*10, model_Z_values*10+sigma_f*10, color='C1', alpha=0.3, label='Zbb Fit ±1σ')

chi2= m.fval
ndof = (len(counts) + len(counts_Z) - len(m.values))
#ax.text(x=0.95, y=0.7, s=r"$\chi^2/n_\text{dof}$ = %.1f/%d"%(chi2, ndof), transform=ax.transAxes, ha='right')
ax.legend()
hep.cms.label(ax=ax)
ax2.errorbar(x, (counts-model_bkg_values), xerr=np.diff(bins)/2,
             yerr= errors,  color='black', markersize=2, marker='o', linestyle='none', label = 'Data')
ax2.plot(x, model_Z_values_guess, color='C1', label='Zbb Fit Extracted')
ax2.fill_between(x, model_Z_values_guess-sigma_f, model_Z_values+sigma_f, color='C1', alpha=0.3, label='(Zbb Fit ±1σ)x10')
ax2.hist(bins[:-1], bins, weights=counts_Z, color='red', label='Z MC', histtype=u'step')
ax2.set_ylim(-1e3, 1e3)
ax2.set_xlabel("Dijet Mass [GeV]")
ax2.hlines(xmin=xmin, xmax=xmax, y=1, color='black')

ax2.set_ylabel("Data - Bkg (QCD)")
ax.yaxis.set_label_coords(-0.1, 1)
ax2.yaxis.set_label_coords(-0.1, 1)

# %%
