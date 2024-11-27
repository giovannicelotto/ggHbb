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



def model(x, p0, p1, p2, p3, p4, norm_Z):
    # Bkg + Zbb fit
    with open('/t3home/gcelotto/ggHbb/NN/workingPoint/fit_results.json', 'r') as f:
        fit_results = json.load(f)
    return  p0 + p1*x + p2*x**2 + p3*x**3 + p4*x**4 +  norm_Z* crystalball_ex.pdf(x,
                                beta_left=fit_results['beta_left']['value'], m_left=fit_results['m_left']['value'], scale_left=fit_results['scale_left']['value'], beta_right=fit_results['beta_right']['value'], m_right=fit_results['m_right']['value'], scale_right=fit_results['scale_right']['value'], loc=fit_results['loc']['value'])
def pol4(x, p0, p1, p2, p3, p4):
    # Bkg only
    fit_bkg_values = p0 +  p1*x + p2*x**2 + p3*x**3 + p4*x**4
    return fit_bkg_values

def zbb_fit_extracted(x, norm_Z, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc):
    # Zbb only contribution
    zbb_values = norm_Z* crystalball_ex.pdf(x,
                                beta_left=beta_left, m_left=m_left, scale_left=scale_left, beta_right=beta_right, m_right=m_right, scale_right=scale_right, loc=loc)
    return zbb_values

def getGradient(x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc):
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

workingPoint = (YPred_QCD[:,1] > 0.34) & (YPred_QCD[:,0] < 0.22) & (df.dijet_pt>100)
mass = df.dijet_mass[workingPoint]

workingPoint_Z = (YPred_Z[:,1] > 0.34) & (YPred_Z[:,0] < 0.22) & (df_Z.dijet_pt>100)
mass_Z = df_Z.dijet_mass[workingPoint_Z]
weights_Z = df_Z.weights[workingPoint_Z]

# %%
# Fitting

x = (bins[1:] + bins[:-1])/2
counts = np.histogram(mass,  bins=bins)[0]
errors = np.sqrt(counts)
integral = np.sum(counts * np.diff(bins))

counts_Z = np.histogram(mass_Z, bins=bins, weights=weights_Z)[0]
errors_Z = np.sqrt(np.histogram(mass_Z, bins=bins, weights=weights_Z**2)[0])

least_squares = LeastSquares(x, counts, errors, model)
initial_params = {
    'p0': -1.02791e6, 
    'p1': 47.328e3, 
    'p2': -668.73, 
    'p3': 3.93099,
    'p4': -8.4152e-3,
    'norm_Z':np.sum(counts_Z*np.diff(bins))
}
m = Minuit(least_squares,
           **initial_params,
           )
m.limits['norm_Z'] = (0,None)
m.migrad()  # finds minimum of least_squares function
m.hesse() 
# %%
# Plotting
fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
# Data

ax.errorbar(x, counts, xerr = np.diff(bins)/2,yerr=errors, color='black', linestyle='none')


# Z MC
counts_Z = ax.hist(bins[:-1], bins=bins, weights=counts_Z, histtype=u'step', color='red', label='Z MC')[0]
ax.errorbar(x, counts_Z, yerr=errors_Z, color='red', linestyle='none')

# Bkg only
fit_bkg_values = pol4(x,m.values['p0'],
                        m.values['p1'],
                        m.values['p2'],
                        m.values['p3'],
                        m.values['p4']
                      )
ax.plot(x, fit_bkg_values, label='Bkg only Fit')


# Z Extracted From Fit
# Open the fit to the MC to extract the shape of Zbb
with open('/t3home/gcelotto/ggHbb/NN/workingPoint/fit_results.json', 'r') as f:
    fit_results = json.load(f)
fit_values = model(x, m.values['p0'], m.values['p1'], m.values['p2'], m.values['p3'], m.values['p4'], m.values['norm_Z'])
zbb_values = zbb_fit_extracted(x, m.values['norm_Z'], fit_results['beta_left']['value'], fit_results['m_left']['value'], fit_results['scale_left']['value'], fit_results['beta_right']['value'], fit_results['m_right']['value'], fit_results['scale_right']['value'], fit_results['loc']['value'])
ax.plot(x, zbb_values, label='Zbb Fit Extracted')
# Naive method for uncertainties
zbb_values_normUP = zbb_fit_extracted(x, m.values['norm_Z'] + m.errors['norm_Z'], fit_results['beta_left']['value'], fit_results['m_left']['value'], fit_results['scale_left']['value'], fit_results['beta_right']['value'], fit_results['m_right']['value'], fit_results['scale_right']['value'], fit_results['loc']['value'])
zbb_values_normDOWN = zbb_fit_extracted(x, m.values['norm_Z'] - m.errors['norm_Z'], fit_results['beta_left']['value'], fit_results['m_left']['value'], fit_results['scale_left']['value'], fit_results['beta_right']['value'], fit_results['m_right']['value'], fit_results['scale_right']['value'], fit_results['loc']['value'])
zbb_values_betaleftUP = zbb_fit_extracted(x, m.values['norm_Z'] , fit_results['beta_left']['value'] + fit_results['beta_left']['error'], fit_results['m_left']['value'], fit_results['scale_left']['value'], fit_results['beta_right']['value'], fit_results['m_right']['value'], fit_results['scale_right']['value'], fit_results['loc']['value'])
zbb_values_betaleftDOWN = zbb_fit_extracted(x, m.values['norm_Z'], fit_results['beta_left']['value'] - fit_results['beta_left']['error'], fit_results['m_left']['value'], fit_results['scale_left']['value'], fit_results['beta_right']['value'], fit_results['m_right']['value'], fit_results['scale_right']['value'], fit_results['loc']['value'])
zbb_values_mleftUP = zbb_fit_extracted(x, m.values['norm_Z'] , fit_results['beta_left']['value'], fit_results['m_left']['value'] + fit_results['m_left']['error'], fit_results['scale_left']['value'], fit_results['beta_right']['value'], fit_results['m_right']['value'], fit_results['scale_right']['value'], fit_results['loc']['value'])
zbb_values_mleftDOWN = zbb_fit_extracted(x, m.values['norm_Z'] , fit_results['beta_left']['value'], fit_results['m_left']['value'] - fit_results['m_left']['error'], fit_results['scale_left']['value'], fit_results['beta_right']['value'], fit_results['m_right']['value'], fit_results['scale_right']['value'], fit_results['loc']['value'])
zbb_values_scaleleftUP = zbb_fit_extracted(x, m.values['norm_Z'] , fit_results['beta_left']['value'] , fit_results['m_left']['value'], fit_results['scale_left']['value'] + fit_results['scale_left']['error'], fit_results['beta_right']['value'], fit_results['m_right']['value'], fit_results['scale_right']['value'], fit_results['loc']['value'])
zbb_values_scaleleftDOWN = zbb_fit_extracted(x, m.values['norm_Z'], fit_results['beta_left']['value'], fit_results['m_left']['value'], fit_results['scale_left']['value'] - fit_results['scale_left']['error'], fit_results['beta_right']['value'], fit_results['m_right']['value'], fit_results['scale_right']['value'], fit_results['loc']['value'])
zbb_values_betaRightUP = zbb_fit_extracted(x, m.values['norm_Z'] , fit_results['beta_left']['value'], fit_results['m_left']['value'] , fit_results['scale_left']['value'], fit_results['beta_right']['value'] + fit_results['beta_right']['error'], fit_results['m_right']['value'], fit_results['scale_right']['value'], fit_results['loc']['value'])
zbb_values_betaRightDOWN = zbb_fit_extracted(x, m.values['norm_Z'] , fit_results['beta_left']['value'], fit_results['m_left']['value'] , fit_results['scale_left']['value'], fit_results['beta_right']['value'] - fit_results['beta_right']['error'], fit_results['m_right']['value'], fit_results['scale_right']['value'], fit_results['loc']['value'])
zbb_values_mRightUP = zbb_fit_extracted(x, m.values['norm_Z'] , fit_results['beta_left']['value'] , fit_results['m_left']['value'], fit_results['scale_left']['value'], fit_results['beta_right']['value'], fit_results['m_right']['value'] + fit_results['m_right']['error'], fit_results['scale_right']['value'], fit_results['loc']['value'])
zbb_values_mRightDOWN = zbb_fit_extracted(x, m.values['norm_Z'], fit_results['beta_left']['value'], fit_results['m_left']['value'], fit_results['scale_left']['value'], fit_results['beta_right']['value'], fit_results['m_right']['value'] - fit_results['m_right']['error'], fit_results['scale_right']['value'], fit_results['loc']['value'])
zbb_values_scaleRightUP     = zbb_fit_extracted(x, m.values['norm_Z'] , fit_results['beta_left']['value'], fit_results['m_left']['value'] , fit_results['scale_left']['value'], fit_results['beta_right']['value'], fit_results['m_right']['value'], fit_results['scale_right']['value'] + fit_results['scale_right']['error'], fit_results['loc']['value'])
zbb_values_scaleRightDOWN   = zbb_fit_extracted(x, m.values['norm_Z'] , fit_results['beta_left']['value'], fit_results['m_left']['value'] , fit_results['scale_left']['value'], fit_results['beta_right']['value'], fit_results['m_right']['value'], fit_results['scale_right']['value'] - fit_results['scale_right']['error'], fit_results['loc']['value'])
zbb_values_locUP    = zbb_fit_extracted(x, m.values['norm_Z'], fit_results['beta_left']['value'], fit_results['m_left']['value'], fit_results['scale_left']['value'], fit_results['beta_right']['value'], fit_results['m_right']['value'], fit_results['scale_right']['value'], fit_results['loc']['value'] + fit_results['loc']['error'])
zbb_values_locDOWN  = zbb_fit_extracted(x, m.values['norm_Z'], fit_results['beta_left']['value'], fit_results['m_left']['value'], fit_results['scale_left']['value'], fit_results['beta_right']['value'], fit_results['m_right']['value'], fit_results['scale_right']['value'], fit_results['loc']['value'] - fit_results['loc']['error'])
zbb_variations = [

    zbb_values_normUP,
    zbb_values_normDOWN,
    zbb_values_betaleftUP,
    zbb_values_betaleftDOWN,
    zbb_values_mleftUP,
    zbb_values_mleftDOWN,
    zbb_values_scaleleftUP,
    zbb_values_scaleleftDOWN,
    zbb_values_betaRightUP,
    zbb_values_betaRightDOWN,
    zbb_values_mRightUP,
    zbb_values_mRightDOWN,
    zbb_values_scaleRightUP,
    zbb_values_scaleRightDOWN,
    zbb_values_locUP,
    zbb_values_locDOWN

]
sigma_f_naive = []
for zbb_var in zbb_variations:
    sigma_f_naive.append(zbb_var-zbb_values)
sigma_f_naive = np.array(sigma_f_naive)
sigma_f_naive = np.sqrt(np.sum(sigma_f_naive**2,axis=0))
ax.fill_between(x, zbb_values - sigma_f_naive , zbb_values + sigma_f_naive , color='C1', alpha=0.3, label='Zbb Fit ±1σ')




#gradient = np.array(getGradient(x, fit_results['beta_left']['value'], fit_results['m_left']['value'], fit_results['scale_left']['value'], fit_results['beta_right']['value'], fit_results['m_right']['value'], fit_results['scale_right']['value'], fit_results['loc']['value']))
#cov = np.array(m.covariance)
#sigma_f = []
#for i in range(len(x)):
#    sigma_f.append(np.sqrt(np.dot(np.dot(gradient.T[i,:],cov),gradient[:,i])))
#sigma_f = np.array(sigma_f)
#ax.fill_between(x, zbb_values-sigma_f, zbb_values+sigma_f, color='C1', alpha=0.3, label='Zbb Fit ±1σ')


hep.cms.label(ax=ax)
ax.set_xlim(xmin, xmax)
ax.set_yscale('log')
ax.legend()
ax.set_ylabel("Counts [a.u.]")
chi2= m.fval
ndof = (len(counts) - len(m.values))
ax.text(x=0.95, y=0.7, s=r"$\chi^2/n_\text{dof}$ = %.1f/%d"%(chi2, ndof), transform=ax.transAxes, ha='right')

ax2.errorbar(x, (counts-fit_bkg_values), xerr=np.diff(bins)/2,
             yerr= errors,  color='black', markersize=2, marker='o', linestyle='none', label = 'Data')
ax2.plot(x, fit_values-fit_bkg_values, color='C1', label='Zbb Fit Extracted')
ax2.hist(bins[:-1], bins, weights=counts_Z, color='red', label='Z MC', histtype=u'step')
ax2.set_ylim(-1e3, 1e3)
ax2.set_xlabel("Dijet Mass [GeV]")
ax2.hlines(xmin=xmin, xmax=xmax, y=1, color='black')

ax2.set_ylabel("Data - Bkg (QCD)")
ax.yaxis.set_label_coords(-0.1, 1)
ax2.yaxis.set_label_coords(-0.1, 1)

























# %%
sys.exit()


# Plotting for guessing parameters
fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
counts = ax.hist(bins[:-1], bins=bins, weights=counts, histtype=u'step', color='black', label='Binned Data')[0]
ax.errorbar(x, counts, yerr=errors, color='black', linestyle='none')

ax.set_xlim(xmin, xmax)
fit_values = model(x, 5e4, 50, -4, .012, 3e5)
ax.plot(x, fit_values, label='Binned Fit')
ax.set_ylabel("Counts [a.u.]")

ax.legend()

ax2.errorbar(x, counts/fit_values, yerr= errors/np.abs(fit_values),
                               color='black',
                               markersize=2,
                               marker='o', linestyle='none')
ax2.set_ylim(0.95, 1.05)
ax2.set_xlabel("Dijet Mass [GeV]")
ax2.hlines(xmin=xmin, xmax=xmax, y=1, color='black')
hep.cms.label(ax=ax)
ax2.set_ylabel("Ratio Data / Fit")
ax.yaxis.set_label_coords(-0.1, 1)
ax2.yaxis.set_label_coords(-0.1, 1)
# %%


def model_normConstrained(x, p0, p1, p2, p3, p4):
    # model with the norm of Z fixed to the integral of the histogram of the MC Zqq Sample
    with open('/t3home/gcelotto/ggHbb/NN/workingPoint/fit_results.json', 'r') as f:
        fit_results = json.load(f)
    norm_Z=np.sum(counts_Z*np.diff(bins))
    return  p0 + p1*x + p2*x**2 + p3*x**3 + p4*x**4 +  norm_Z* crystalball_ex.pdf(x,
                                beta_left=fit_results['beta_left']['value'], m_left=fit_results['m_left']['value'], scale_left=fit_results['scale_left']['value'], beta_right=fit_results['beta_right']['value'], m_right=fit_results['m_right']['value'], scale_right=fit_results['scale_right']['value'], loc=fit_results['loc']['value'])