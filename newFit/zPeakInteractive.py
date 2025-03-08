# Required libraries
# %%
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import ipywidgets as widgets
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.special import erf
from IPython.display import display
from scipy.stats import shapiro, kstest, norm, chi2
hep.style.use("CMS")

# Load data
from functions import loadMultiParquet_v2, loadMultiParquet_Data_new, cut, getDfProcesses_v2

# Load Monte Carlo data
MCList = [1, 19, 20, 21, 22]
dfsMC, sumw = loadMultiParquet_v2(
    paths=MCList, 
    nMCs=[2, -1, 10, 1, 1], 
    columns=['dijet_mass', 'dijet_pt', 'muon_dxySig', 'sf', 'PU_SF', 'genWeight'], 
    returnNumEventsTotal=True
)
dfsMC = cut(dfsMC, 'dijet_pt', 100, None)
dfProcesses = getDfProcesses_v2()[0].iloc[MCList]

# Define histogram bins
bins = np.linspace(40, 300, 51)
cTot = np.zeros(len(bins)-1)
err = np.zeros(len(bins)-1)

# Compute histogram and errors
for idx, df in enumerate(dfsMC):
    c = np.histogram(df.dijet_mass, bins=bins, 
                     weights=df.genWeight * df.sf * df.PU_SF * dfProcesses.xsection.iloc[idx] / sumw[idx])[0]
    cerr = np.histogram(df.dijet_mass, bins=bins, 
                        weights=(df.genWeight * df.sf * df.PU_SF * dfProcesses.xsection.iloc[idx] / sumw[idx])**2)[0]
    err += cerr
    cTot += c
err = np.sqrt(err)

# Define fit region
x = (bins[1:] + bins[:-1]) / 2
x1, x2 = 40, 300
x3, x4 = 302, 303
fitregion = ((x > x1) & (x < x2) | (x > x3) & (x < x4))

# Define the Double-sided Crystal Ball function
def dscb(x, norm, mean, sigma, alphaL, nL, alphaR, nR):
    """Double-sided Crystal Ball function."""
    t = (x - mean) / sigma
    A = (nL / abs(alphaL)) ** nL * np.exp(-0.5 * alphaL ** 2)
    B = (nR / abs(alphaR)) ** nR * np.exp(-0.5 * alphaR ** 2)
    
    left = A * (nL / abs(alphaL) - abs(alphaL) - t) ** (-nL)  
    right = B * (nR / abs(alphaR) - abs(alphaR) + t) ** (-nR)  
    central = np.exp(-0.5 * t ** 2)  
    
    return np.where(t < -alphaL, left, np.where(t > alphaR, right, central)) * norm

# Background and Gaussian components
def background(x, p0, p1):
    return p0 + p1 * x 

def gaussian(x, normG, mean, sigmaG):
    return normG * np.exp(-0.5 * ((x - mean) / sigmaG) ** 2)

# Total model combining all components
def total_model(x, norm, mean, sigma, alphaL, nL, alphaR, nR, normG, sigmaG, p0, p1):
    return dscb(x, norm, mean, sigma, alphaL, nL, alphaR, nR) + gaussian(x, normG, mean, sigmaG) + background(x, p0, p1)

# Define least squares function
least_squares = LeastSquares(x[fitregion], cTot[fitregion], err[fitregion], total_model)

# Interactive plotting function
def update_fit(norm, mean, sigma, alphaL, nL, alphaR, nR, normG, sigmaG, p0, p1):
    m = Minuit(least_squares, norm=norm, mean=mean, sigma=sigma, alphaL=alphaL, nL=nL, alphaR=alphaR, nR=nR, normG=normG, sigmaG=sigmaG, p0=p0, p1=p1)
    m.migrad()
    print(m.values)
    
    # Generate fit curve
    fit_x = np.linspace(40, 300, 300)
    fit_y = total_model(fit_x, **m.values.to_dict())
    
    # Plot data and fit
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar((bins[1:] + bins[:-1]) / 2, cTot, err, marker='o', color='black', linestyle='none', label="Data")
    ax.plot(fit_x, fit_y, color='red', label="Fit")
    y_values = total_model((bins[1:] + bins[:-1]) / 2, **m.values.to_dict())
    
    chi2_stat = np.sum(((cTot[fitregion] - y_values[fitregion])**2) / err[fitregion]**2)
    ndof = len(x[fitregion]) - len(m.parameters)
    chi2_pvalue = 1- chi2.cdf(chi2_stat, ndof)
    ax.text(x=0.95, y=0.85, s="$\chi^2$/ndof = %.1f/%d\np-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax.transAxes, ha='right')


    ax.set_xlabel("Dijet Mass")
    ax.set_ylabel("Counts")
    ax.legend()
    plt.show()

# Create interactive widgets
interactive_fit = widgets.interactive(
    update_fit,
    norm=widgets.FloatSlider(value=3.6e-1, min=0.1, max=1, step=0.01, description="norm"),
    mean=widgets.FloatSlider(value=91, min=80, max=100, step=0.5, description="mean"),
    sigma=widgets.FloatSlider(value=13, min=5, max=20, step=0.5, description="sigma"),
    alphaL=widgets.FloatSlider(value=3, min=1, max=10, step=0.5, description="alphaL"),
    nL=widgets.FloatSlider(value=6, min=1, max=10, step=1, description="nL"),
    alphaR=widgets.FloatSlider(value=10, min=1, max=20, step=0.5, description="alphaR"),
    nR=widgets.FloatSlider(value=6, min=1, max=20, step=1, description="nR"),
    sigmaG=widgets.FloatSlider(value=5, min=1, max=20, step=0.5, description="sigmaG"),
    normG=widgets.FloatSlider(value=1e-1, min=1e-3, max=1, step=1e-2, description="normG"),
    p0=widgets.FloatSlider(value=0, min=-0.1, max=0.1, step=0.01, description="p0"),
    p1=widgets.FloatSlider(value=0, min=-0.01, max=0.01, step=0.001, description="p1"),
)

# Display the interactive plot
display(interactive_fit)

# %%
