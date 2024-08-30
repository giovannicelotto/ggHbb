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
from scipy import odr

# %%
xmin = 60
xmax = 200
bins = np.linspace(xmin, xmax, 50)


def model(x, p0, p1, p2, p3, p4, norm_Z):
    
    with open('/t3home/gcelotto/ggHbb/NN/workingPoint/fit_results.json', 'r') as f:
        fit_results = json.load(f)
    return  p0 + p1*x + p2*x**2 + p3*x**3 + p4*x**4 +  norm_Z* crystalball_ex.pdf(x,
                                beta_left=fit_results['beta_left']['value'], m_left=fit_results['m_left']['value'], scale_left=fit_results['scale_left']['value'], beta_right=fit_results['beta_right']['value'], m_right=fit_results['m_right']['value'], scale_right=fit_results['scale_right']['value'], loc=fit_results['loc']['value'])




# %%
# Load Data
print("Loading Data")
from loadData import loadData
nReal = 1005
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

# %%
# Working Points
workingPoint = (YPred_QCD[:,1] > 0.34) & (YPred_QCD[:,0] < 0.22) & (df.dijet_pt>120)
mass = df.dijet_mass[workingPoint]

workingPoint_Z = (YPred_Z[:,1] > 0.34) & (YPred_Z[:,0] < 0.22) & (df_Z.dijet_pt>120)
mass_Z = df_Z.dijet_mass[workingPoint_Z]
weights_Z = df_Z.weights[workingPoint_Z]

# %%
# Fitting

x = (bins[1:] + bins[:-1])/2
counts = np.histogram(mass,  bins=bins)[0]
errors = np.sqrt(counts)
integral = np.sum(counts * np.diff(bins))





counts_Z = np.histogram(mass_Z, bins=bins, weights=weights_Z)[0]

def getNorm_Z():
    return np.sum(counts_Z*np.diff(bins))

def model_normConstrained(x, p0, p1, p2, p3, p4):
    with open('/t3home/gcelotto/ggHbb/NN/workingPoint/fit_results.json', 'r') as f:
        fit_results = json.load(f)
    norm_Z=getNorm_Z()
    return  p0 + p1*x + p2*x**2 + p3*x**3 + p4*x**4 +  norm_Z* crystalball_ex.pdf(x,
                                beta_left=fit_results['beta_left']['value'], m_left=fit_results['m_left']['value'], scale_left=fit_results['scale_left']['value'], beta_right=fit_results['beta_right']['value'], m_right=fit_results['m_right']['value'], scale_right=fit_results['scale_right']['value'], loc=fit_results['loc']['value'])


#least_squares = LeastSquares(x, counts, errors, model_normConstrained)
#initial_params = {
#    'p0': -1.02791e6, 
#    'p1': 47.328e3, 
#    'p2': -668.73, 
#    'p3': 3.93099,
#    'p4': -8.4152e-3,
#}
#m = Minuit(least_squares,
#           **initial_params,
#           )
#m.migrad()  # finds minimum of least_squares function
#m.hesse() 


least_squares = LeastSquares(x, counts, errors, model)
initial_params = {
    'p0': -1.02791e6, 
    'p1': 47.328e3, 
    'p2': -668.73, 
    'p3': 3.93099,
    'p4': -8.4152e-3,
    'norm_Z':getNorm_Z()
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
counts = ax.hist(bins[:-1], bins=bins, weights=counts, histtype=u'step', color='black', label='Binned Data')[0]
ax.errorbar(x, counts, yerr=errors, color='black', linestyle='none')

ax.set_xlim(xmin, xmax)

#fit_values = model_normConstrained(x, m.values['p0'], m.values['p1'], m.values['p2'], m.values['p3'], m.values['p4'], m.values['norm_Z'])
fit_values = model_normConstrained(x, m.values['p0'], m.values['p1'], m.values['p2'], m.values['p3'], m.values['p4'])
ax.plot(x, fit_values, label='Binned Fit')
with open('/t3home/gcelotto/ggHbb/NN/workingPoint/fit_results.json', 'r') as f:
    fit_results = json.load(f)
#fit_Z_values = m.values['norm_Z'] *crystalball_ex.pdf(
#    x, fit_results['beta_left']['value'], fit_results['m_left']['value'],
#    fit_results['scale_left']['value'],  fit_results['beta_right']['value'],
#    fit_results['m_right']['value'], fit_results['scale_right']['value'],
#    fit_results['loc']['value'])
#ax.plot(x, fit_Z_values, label='Z Peak')


counts_Z = ax.hist(bins[:-1], bins=bins, weights=counts_Z, histtype=u'step', color='red', label='Z MC')[0]
ax.errorbar(x, counts, yerr=errors, color='black', linestyle='none')

fit_bkg_values = m.values['p0'] +  m.values['p1']*x + m.values['p2']*x**2 + m.values['p3']*x**3 + m.values['p4']*x**4
ax.plot(x, fit_bkg_values, label='Bkg only')
ax.set_ylabel("Counts [a.u.]")

ax.legend()
chi2= m.fval
ndof = (len(counts) - len(m.values))
ax.text(x=0.95, y=0.7, s=r"$\chi^2/n_\text{dof}$ = %.1f/%d"%(chi2, ndof), transform=ax.transAxes, ha='right')

ax2.errorbar(x, counts/fit_values, yerr= errors/fit_values,
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
