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
xmin = 40
xmax = 200
bins=np.linspace(xmin, xmax, 100)

# Define the standard polynomial (canonical basis) PDF
def polynomial_pdf(x, p0, p1):
    poly_value = p0 + p1 * x
    return poly_value


# Define the background polynomial normalized to 1 over the range [xmin, xmax]
def normalized_polynomial_pdf(x, p0, p1, xmin, xmax):
    # Compute the normalization factor
    integral = (p0 * (xmax - xmin) + p1 * (xmax**2 - xmin**2)/2)
    #print("Integral : %.2f"%integral)
    # Compute the polynomial value
    polynomial = polynomial_pdf(x, p0, p1)
    #print(np.array(polynomial).shape, np.array(integral).shape)
    # Normalize the polynomial
    return polynomial / (integral)

def normalized_polynomial_pdf_forFit(xweights, p0, p1):
    x, weights = xweights
    # Compute the normalization factor
    integral = (p0 * (xmax - xmin) + p1 * (xmax**2 - xmin**2)/2)
    #print("Integral : %.2f"%integral)
    # Compute the polynomial value
    polynomial = polynomial_pdf(x, p0, p1)**weights
    # Normalize the polynomial
    return polynomial / integral

# Define the combined PDF
def combined_pdf(x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc, p0, p1, f):
    signal = crystalball_ex.pdf(x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc)
    #print("Params passed")
    #print(beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc)
    #print(p0, p1)
    bkg = normalized_polynomial_pdf(x, p0, p1, xmin, xmax)  # Normalize polynomial over the range
    return f * signal + (1 - f) * bkg

# Define the cost function
def combined_cost(xsf, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc, p0, p1, f):
    x, sf = xsf
    return combined_pdf(x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc, p0, p1, f) ** sf

# Fit the data using the combined PDF
def fit_with_combined_pdf(mass, weights):
    costFunc = cost.UnbinnedNLL((mass, weights), combined_cost)
    
    p0, p1, f =  3.5602e-5, -1.2052e-7 , 0.947
    truth = (1.59, 150, 21.9, 
             1.67,  11.7, 13.3, 123,
             p0, p1, f)
    
    # Create the Minuit object
    m = Minuit(costFunc, *truth)
    
    # Set parameter limits
    m.limits["beta_left", "beta_right"] = (1, 20)
    m.limits["m_left"] = (1, 500)
    m.limits["m_right"] = (1, 500)
    m.limits["scale_left", "scale_right"] = (10, 25)
    m.limits["loc"] = (122, 126)
    m.limits["p0"] = (0, 0.001)
    m.limits["p1"] = (-1e-5, -1e-7)
    m.limits["f"] = (0.5, 1)
    m.migrad()  # finds minimum of least_squares function
    m.hesse()   # accurately computes uncertainties
    #m.limits["p0", "p1"] = (-1, 1)  # Adjust these limits based on expected ranges
    

    # Perform the fit
    m.migrad()
    m.hesse()
    
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
weights = np.ones(len(dfs[0]))[workingPoint]   #dfs[0].sf[workingPoint] #* dfs[0].PU_SF[workingPoint]
mass = dfs[0].dijet_mass[workingPoint]



# %%
print("Fit started")
m = fit_with_combined_pdf(mass, weights)




# %%
print(m.params) # parameter info (using str(m.params))
print(m.covariance)
m.fval/(len(dfs[0][workingPoint])-7)


# %%
# Plotting
fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)


x = (bins[:-1]+bins[1:])/2
c = np.histogram(mass, bins=bins,weights=weights)[0]
integral = np.sum(c*np.diff(bins))

ax.hist(bins[:-1], bins=bins, weights=c, density=True, histtype=u'step', color='black', label='Binned Data')
ax.errorbar(x, c/integral, yerr=np.sqrt(c)/integral, color='black', linestyle='none')

ax.set_xlim(xmin, xmax)
bkg_values = normalized_polynomial_pdf(x, m.values['p0'], m.values['p1'], xmin, xmax)
ax.plot(x, bkg_values*(1-m.values['f']), label='Bkg only')
pdf_values = combined_pdf(x, m.values['beta_left'], m.values['m_left'], m.values['scale_left'], m.values['beta_right'], m.values['m_right'], m.values['scale_right'], m.values['loc'], m.values['p0'], m.values['p1'], m.values['f'])
ax.plot(x, pdf_values, label='Unbinned Fit')
ax.set_ylabel("Normalized Events")


from scipy.stats import gaussian_kde
kde = gaussian_kde(mass, weights = weights)
x_range = np.linspace(xmin, xmax, 1000)
kde_values = kde(x_range)
ax.plot(x_range, kde_values, label='KDE', color='red')


ax.legend()

#ax2.plot(x_range, crystalball_ex.pdf(x_range, m.values['beta_left'], m.values['m_left'],
#                               m.values['scale_left'],  m.values['beta_right'],
#                               m.values['m_right'], m.values['scale_right'],
#                               m.values['loc']) / kde_values, color='black')
ax2.set_ylim(0.5, 1.5)
ax2.set_xlabel("Dijet Mass [GeV]")
ax2.hlines(xmin=xmin, xmax=xmax, y=1, color='black')

ax2.set_ylabel("Ratio Fit / KDE")
ax.yaxis.set_label_coords(-0.1, 1)
ax2.yaxis.set_label_coords(-0.1, 1)
# %%
# %%





fig, ax = plt.subplots(figsize=(10, 7))
x = np.linspace(xmin, xmax, 100)
p0, p1, f =  3.5602e-5, -1.2052e-7 , 0.96
if p1<-p0/200:
    print("negative pol")
else:
    print("alright")
truth = (1.213, 137, 20, 
             1.54,  11.7, 14.88, 123,
             p0, p1, f)
beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc, p0, p1, f = truth
bkg_values = normalized_polynomial_pdf(x, p0, p1, xmin, xmax)
ax.plot(x, bkg_values*(1-f))
# Calculate the combined PDF over the range
pdf_values = combined_pdf(x, beta_left, m_left, scale_left, beta_right, m_right, scale_right, loc, p0, p1, f)
ax.plot(x, pdf_values)
ax.hist(mass, bins, weights=weights, histtype='step', color='black', label='Data', density=True)
#ax.plot(x, combined_pdf(x, *m.values), label='Fit', color='blue')
ax.set_xlabel('Dijet Mass [GeV]')
ax.set_ylabel('Normalized Events')
ax.set_ylim(0, 0.002)
ax.legend()



# %%
