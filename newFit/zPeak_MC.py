# %%
from functions import loadMultiParquet_v2, loadMultiParquet_Data_new, cut, getDfProcesses_v2
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
from scipy.stats import chi2
import json
# %%
MCList = [1, 3, 4, 19, 20, 21, 22]
dfsMC, sumw = loadMultiParquet_v2(paths=MCList, nMCs=-1, columns=['jet1_pt', 'dijet_mass', 'dijet_pt', 'muon_dxySig', 'sf', 'PU_SF', 'genWeight', 'jet1_btagDeepFlavB', 'jet2_btagDeepFlavB', 'muon_eta'], returnNumEventsTotal=True)
dfsMC = cut(dfsMC, 'dijet_pt', 100, None)
dfsMC = cut(dfsMC, 'jet1_btagDeepFlavB', 0.71, None)
dfsMC = cut(dfsMC, 'jet2_btagDeepFlavB', 0.71, None)
# %%
dfProcesses = getDfProcesses_v2()[0].iloc[MCList]
# %%
bins = np.linspace(40, 300, 101)
cTot = np.zeros(len(bins)-1)
err = np.zeros(len(bins)-1)
fig, ax = plt.subplots(1, 1)
for idx, df in enumerate(dfsMC):
    c=ax.hist(df.dijet_mass, bins=bins, bottom=cTot, weights=df.genWeight*df.sf*df.PU_SF*dfProcesses.xsection.iloc[idx]/sumw[idx], label=dfProcesses.process.iloc[idx])[0]
    cerr=np.sqrt(np.histogram(df.dijet_mass, bins=bins, weights=(df.genWeight*df.sf*df.PU_SF*dfProcesses.xsection.iloc[idx]/sumw[idx])**2)[0])
    err = np.sqrt(err**2 + cerr**2)

    cTot=cTot+c
#err = np.sqrt(err)
ax.errorbar((bins[1:]+bins[:-1])/2, cTot, err, marker='o', color='black', linestyle='none')
ax.legend()


from iminuit.cost import LeastSquares
x1, x2 = 40, 300
x3, x4 = 302, 303
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.special import erf

def dscb(x, norm, mean, sigma, alphaL, nL, alphaR, nR):
    """Double-sided Crystal Ball function."""
    t = (x - mean) / sigma
    A = (nL / abs(alphaL)) ** nL * np.exp(-0.5 * alphaL ** 2)
    B = (nR / abs(alphaR)) ** nR * np.exp(-0.5 * alphaR ** 2)
    
    left = A * (nL / abs(alphaL) - abs(alphaL) - t) ** (-nL)  # Left tail
    right = B * (nR / abs(alphaR) - abs(alphaR) + t) ** (-nR)  # Right tail
    
    central = np.exp(-0.5 * t ** 2)  # Gaussian core
    return np.where(t < -alphaL, left, np.where(t > alphaR, right, central)) * norm
def background(x, p0, p1):#, p2):
    return p0 + p1 * x #+ p2 * x**2
def gaussian(x, normG, mean, sigmaG):
    return normG * np.exp(-0.5 * ((x - mean) / sigmaG) ** 2)

def total_model(x, norm, mean, sigma, alphaL, nL, alphaR, nR, normG, sigmaG, p0, p1):
    return dscb(x, norm, mean, sigma, alphaL, nL, alphaR, nR) + gaussian(x, normG, mean, sigmaG)+ background(x, p0, p1)

x = (bins[1:] + bins[:-1]) / 2
fitregion = ((x > x1) & (x < x2) | (x > x3) & (x < x4))

least_squares = LeastSquares(x[fitregion], cTot[fitregion], err[fitregion], total_model)
m = Minuit(least_squares, norm=0.15116, mean=91.93, sigma=12.415, alphaL=0.67, nL=12.18, alphaR=1.35, nR=0.62, sigmaG=11.63, normG=0.21, p0=0.00348, p1=-1.71e-5)#, p2=0)
#m.limits['sigmaG'] = (None, 20)
#m.limits['mean'] = (83, 97)

m.migrad()
m.hesse()

x_draw = np.linspace(40, 300, 1001)
y_draw = total_model(x_draw, *[m.values[p] for p in m.parameters])
y_values = total_model(x, *[m.values[p] for p in m.parameters])

chi2_stat = np.sum(((cTot[fitregion] - y_values[fitregion])**2) / err[fitregion]**2)
ndof = len(x[fitregion]) - len(m.parameters)
chi2_pvalue = 1- chi2.cdf(chi2_stat, ndof)
ax.text(x=0.95, y=0.85, s="$\chi^2$/ndof = %.1f/%d\np-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax.transAxes, ha='right')

#ax.hist(x, bins=bins, weights=cTot, histtype='step', label='Data', linewidth=3, color='black')
ax.plot(x_draw, y_draw, 'r-', label='DSCB + Guas +  pol1 Fit', color='blue')
ax.legend()
fig.savefig("/t3home/gcelotto/newFit/plots/zPeakFit.png", bbox_inches='tight')
fit_params = {name: {"value": m.values[name], "error": m.errors[name]} for name in m.parameters}
with open("/t3home/gcelotto/newFit/fit_parameters_Z100_bT.json", "w") as f:
    json.dump(fit_params, f, indent=4)

print("Cross check\nMy Chi2: %.1f\nMinuit Chi2: %.1f"%(chi2_stat, m.fval))
# %%

y_background = background(x_draw, *[m.values[k] for k in ['p0', 'p1']])
y_gaussian =  y_background+gaussian(x_draw, *[m.values[k] for k in ['normG', 'mean', 'sigmaG']])
y_dscb = y_gaussian + dscb(x_draw, *[m.values[k] for k in ['norm', 'mean', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR']])
y_total = total_model(x_draw, *[m.values[k] for k in m.parameters])

fig2, ax2 = plt.subplots(1, 1)
#ax2.fill_between(x_draw, y_background, 0, color='gray', alpha=0.5, label='Background (pol2)')
ax2.fill_between(x_draw, 0, y_gaussian, color='blue', alpha=0.5, label='Gaussian + pol1')
ax2.fill_between(x_draw, y_gaussian, y_dscb, color='red', alpha=0.5, label='DSCB')
ax2.errorbar(x, cTot, yerr=err, fmt='ko', label='Data')
ax2.text(x=0.95, y=0.85, s="$\chi^2$/ndof = %.1f/%d\np-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax2.transAxes, ha='right')
ax2.legend()
ax2.set_xlabel("Dijet Mass [GeV]")
ax2.set_ylabel("Normalized Events")
fig2.savefig("/t3home/gcelotto/newFit/plots/zPeakFit_components.png", bbox_inches='tight')


# %%
import pandas as pd
fig, ax = plt.subplots(1, 1)
ax.hist(pd.concat(dfsMC).jet1_pt, bins=np.linspace(0, 40, 101))
# %%
