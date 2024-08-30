# %%
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
xmin = 40
xmax = 200
bins = np.linspace(xmin, xmax, 100)

# Define the standard polynomial (canonical basis) PDF
def polynomial_pdf(x, p0, p1):
    # Define the polynomial in the standard basis (degree 2 polynomial)
    return p0 + p1*(x)


# Define the background polynomial normalized to 1 over the range [xmin, xmax]
def normalized_polynomial_pdf(x, p0, p1, xmin, xmax):
    # Compute the normalization factor
    integral = (p0 * (xmax - xmin) + p1 * (xmax**2 - xmin**2)/2)
    # Compute the polynomial value
    polynomial = polynomial_pdf(x, p0, p1)
    # Normalize the polynomial
    return polynomial / integral

# Define the combined PDF
def combined_pdf(x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc, p0, p1, f):
    signal = crystalball_ex.pdf(x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc)
    bkg = normalized_polynomial_pdf(x, p0, p1, xmin, xmax)  # Normalize polynomial over the range
    return f * signal + (1 - f) * bkg

# Define the cost function
def combined_cost(xsf, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc, p0, p1, f):
    x, sf = xsf
    return combined_pdf(x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc, p0, p1, f) ** sf

# Fit the data using the combined PDF
def fit_with_combined_pdf(mass, weights):
    costFunc = cost.UnbinnedNLL((mass, weights), combined_cost)
    
    # Initial parameters: you may need to tune these for your specific data
    truth = (2.1, 30, 15, 1.4, 12, 14, 93,
             0.005, 0,
             0.8)
    
    # Create the Minuit object
    m = Minuit(costFunc, *truth)
    
    # Set parameter limits
    m.limits["beta_left"] = (0,5)
    m.limits["beta_right"] = (0,3)
    m.limits["m_left"] = (0, 300)
    m.limits["m_right"] = (10, 18)
    m.limits["scale_left", "scale_right"] = (10, 30)
    m.limits["loc"] = (80, 100)
    m.limits["p0", "p1"] = (-1, 1)  # Adjust these limits based on expected ranges
    m.limits["f"] = (0, 1)
    

    # Perform the fit
    m.migrad()
    m.hesse()
    
    # Output results
    print(m.params)
    return m

# %%
# Load Data
print("Loading Data")
from loadData import loadData
dfs, YPred_Z100, YPred_Z200, YPred_Z400, YPred_Z600, YPred_Z800, numEventsList = loadData(pdgID=23)

# %%
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
# Fit the mass distribution
m = fit_with_combined_pdf(mass, weights)

# %%
# Plot the results
fig, ax = plt.subplots(figsize=(10, 7))
x = np.linspace(xmin, xmax, 1000)
ax.hist(mass, bins, weights=weights, histtype='step', color='black', label='Data', density=True)
ax.plot(x, combined_pdf(x, *m.values), label='Fit', color='blue')
ax.set_xlabel('Dijet Mass [GeV]')
ax.set_ylabel('Normalized Events')
ax.legend()
plt.show()

# %%
