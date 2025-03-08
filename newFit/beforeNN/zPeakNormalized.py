# %%
from functions import loadMultiParquet_v2, loadMultiParquet_Data_new, cut, getDfProcesses_v2
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
from scipy.stats import chi2
import json
# %%
# Load data and cut
MCList = [1, 3, 4, 19, 20, 21, 22]
dfsMC, sumw = loadMultiParquet_v2(paths=MCList, nMCs=-1, columns=['dijet_mass', 'dijet_pt', 'muon_dxySig', 'sf', 'PU_SF', 'genWeight', 'jet1_btagDeepFlavB', 'jet2_btagDeepFlavB'], returnNumEventsTotal=True)
dfsMC = cut(dfsMC, 'dijet_pt', 100, None)
dfsMC = cut(dfsMC, 'jet1_btagDeepFlavB', 0.71, None)
dfsMC = cut(dfsMC, 'jet2_btagDeepFlavB', 0.71, None)
# %%
dfProcesses = getDfProcesses_v2()[0].iloc[MCList]
# %%
x1, x2 = 40, 300
bins = np.linspace(x1, x2, 101)
cTot = np.zeros(len(bins)-1)
err = np.zeros(len(bins)-1)
fig, ax = plt.subplots(1, 1)
for idx, df in enumerate(dfsMC):
    c=ax.hist(df.dijet_mass, bins=bins, bottom=cTot, weights=df.genWeight*df.sf*df.PU_SF*dfProcesses.xsection.iloc[idx]/sumw[idx], label=dfProcesses.process.iloc[idx])[0]
    cerr=np.histogram(df.dijet_mass, bins=bins, weights=(df.genWeight*df.sf*df.PU_SF*dfProcesses.xsection.iloc[idx]/sumw[idx])**2)[0]
    err = err + cerr

    cTot=cTot+c
err = np.sqrt(err)
ax.errorbar((bins[1:]+bins[:-1])/2, cTot, err, marker='o', color='black', linestyle='none')
ax.legend()


from iminuit.cost import LeastSquares

from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.special import erf

import scipy.integrate as integrate

def dscb(x, mean, sigma, alphaL, nL, alphaR, nR):
    t = (x - mean) / sigma
    A = (nL / abs(alphaL)) ** nL * np.exp(-0.5 * alphaL ** 2)
    B = (nR / abs(alphaR)) ** nR * np.exp(-0.5 * alphaR ** 2)

    left = A * (nL / abs(alphaL) - abs(alphaL) - t) ** (-nL)  # Left tail
    right = B * (nR / abs(alphaR) - abs(alphaR) + t) ** (-nR)  # Right tail

    central = np.exp(-0.5 * t ** 2)


    integralLeft =  -A*sigma/(-nL+1)*(
        (nL / abs(alphaL) - abs(alphaL) - (-alphaL)) ** (-nL+1) - 
        (nL / abs(alphaL) - abs(alphaL) - (x1-mean)/sigma) ** (-nL+1))
    integralRight =  B*sigma/(-nR+1)*(
        (nR / abs(alphaR) - abs(alphaR) + (x2-mean)/sigma) ** (-nR+1) - 
        (nR / abs(alphaR) - abs(alphaR) + (alphaR)) ** (-nR+1))
    integralCentral = 1/2*(erf(alphaR/np.sqrt(2))  -    erf(-alphaL/np.sqrt(2)))*np.sqrt(np.pi)*(np.sqrt(2)*sigma)
    totalIntegral = integralLeft + integralRight + integralCentral
    return np.where(t < -alphaL, left, np.where(t > alphaR, right, central))/totalIntegral


def background(x, p1):
    result = 1 + p1 * x
    integral = (x2-x1) + p1*(x2**2-x1**2)/2

    return  result/integral
def gaussianN(x, mean, sigmaG):
    integralGauss = 1/2*(erf((x2-mean)/(sigmaG*np.sqrt(2)))  -    erf((x1-mean)/(sigmaG*np.sqrt(2))))
    return  1/(np.sqrt(2*np.pi)*sigmaG)* np.exp(-0.5 * ((x - mean) / sigmaG) ** 2)/integralGauss

def total_model(x, normTotal, fraction_dscb, mean, sigma, alphaL, nL, alphaR, nR, fraction_gaussian, sigmaG, p1):
    return normTotal*(fraction_dscb*dscb(x, mean, sigma, alphaL, nL, alphaR, nR) + fraction_gaussian*gaussianN(x, mean, sigmaG)+ (1-fraction_gaussian-fraction_dscb)*background(x, p1))

x = (bins[1:] + bins[:-1]) / 2
fitregion = ((x > x1) & (x < x2))

least_squares = LeastSquares(x[fitregion], cTot[fitregion], err[fitregion], total_model)
m = Minuit(least_squares,
            normTotal=cTot.sum()*(bins[1]-bins[0]),
            fraction_dscb=0.48,
           mean=92.31,
           sigma=11.3,
           alphaL=0.66,
           nL=126,
           alphaR=1.45,
           nR=0.8,
           fraction_gaussian=0.48,
           sigmaG=11.29,
           p1=-2.92e-3)
m.limits['fraction_gaussian'] = (0, 1)
m.limits['fraction_dscb'] = (0, 1)
#m.fixed['alphaL'] = True
#m.fixed['alphaR'] = True
#m.fixed['nL'] = True
#m.fixed['nR'] = True
#m.fixed['sigma'] = True
#m.fixed['mean'] = True
#m.fixed['fraction_dscb'] = True
#m.fixed['fraction_gaussian'] = True
m.limits['mean'] = (83, 97)

m.migrad()
m.hesse()

x_draw = np.linspace(x1, x2, 1001)
y_draw = total_model(x_draw, *[m.values[p] for p in m.parameters])
y_values = total_model(x, *[m.values[p] for p in m.parameters])

chi2_stat = np.sum(((cTot[fitregion] - y_values[fitregion])**2) / err[fitregion]**2)
ndof = len(x[fitregion]) - len(m.parameters)
chi2_pvalue = 1- chi2.cdf(chi2_stat, ndof)
ax.text(x=0.95, y=0.85, s="$\chi^2$/ndof = %.1f/%d\np-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax.transAxes, ha='right')

#ax.hist(x, bins=bins, weights=cTot, histtype='step', label='Data', linewidth=3, color='black')
ax.plot(x_draw, y_draw, 'r-', label='DSCB + Guas +  pol1 Fit', color='red', linewidth=5)
ax.legend()
fig.savefig("/t3home/gcelotto/newFit/plots/zPeakFit.png", bbox_inches='tight')
fit_params = {name: {"value": m.values[name], "error": m.errors[name]} for name in m.parameters}
# %%
with open("/t3home/gcelotto/newFit/fit_parameters_Z100_bT.json", "w") as f:
    json.dump(fit_params, f, indent=4)
# %%

y_background = m.values["normTotal"]*(1-m.values["fraction_gaussian"]-m.values["fraction_dscb"])*background(x_draw, *[m.values[k] for k in ['p1']])
y_gaussian =  y_background+m.values["normTotal"]*m.values["fraction_gaussian"]*gaussianN(x_draw, *[m.values[k] for k in ['mean', 'sigmaG']])
y_dscb = y_gaussian + m.values["normTotal"]*m.values["fraction_dscb"]*dscb(x_draw, *[m.values[k] for k in ['mean', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR']])
y_total = total_model(x_draw, *[m.values[k] for k in m.parameters])

fig2, ax2 = plt.subplots(1, 1)
ax2.fill_between(x_draw, 0, y_background, color='gray', alpha=0.5, label='Background (pol1)')
ax2.fill_between(x_draw, y_background, y_gaussian, color='blue', alpha=0.5, label='Gaussian')
ax2.fill_between(x_draw, y_gaussian, y_dscb, color='red', alpha=0.5, label='DSCB')
ax2.errorbar(x, cTot, yerr=err, fmt='ko', label='Data')
ax2.text(x=0.95, y=0.85, s="$\chi^2$/ndof = %.1f/%d\np-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax2.transAxes, ha='right')
ax2.legend()
ax2.set_xlabel("Dijet Mass [GeV]")
ax2.set_ylabel("Normalized Events")
fig2.savefig("/t3home/gcelotto/newFit/plots/zPeakFit_components.png", bbox_inches='tight')



# %%

integrate.quad(lambda x: gaussianN(x, *[m.values[k] for k in ['mean', 'sigmaG']]), x1, x2)
# %%
integrate.quad(lambda x: background(x, *[m.values[k] for k in ['p1']]), x1, x2)
# %%
integrate.quad(lambda x: dscb(x, *[m.values[k] for k in [
'mean',
'sigma',
'alphaL',
'nL',
'alphaR',
'nR'
]]), x1, x2)
# %%
