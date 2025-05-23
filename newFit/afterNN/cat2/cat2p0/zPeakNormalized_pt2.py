# %%
from functions import loadMultiParquet_v2, loadMultiParquet_Data_new, cut, getDfProcesses_v2
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
from scipy.stats import chi2
import json, sys
import pandas as pd
from iminuit.cost import LeastSquares
sys.path.append("/t3home/gcelotto/ggHbb/newFit/afterNN/")
from helpers.allFunctions import *
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.special import erf
import scipy.integrate as integrate
# %%
# Load data and cut
MCList = [1, 3, 4, 19, 20, 21, 22]
x1, x2 = 40, 300
set_x_bounds(x1, x2)
modelName = "Mar06_2_0p0"
path = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/abcd_df/mjjDisco/%s"%modelName
dfProcesses = getDfProcesses_v2()[0].iloc[MCList]
outFolder = "/t3home/gcelotto/ggHbb/newFit/afterNN/cat2/cat2p0"
# %%
dfsMC = []
for processName in dfProcesses.process.values:
    print("Opening ", processName)
    df = pd.read_parquet(path+"/df_%s_%s.parquet"%(processName, modelName))
    dfsMC.append(df)

# %%
# Play with cuts
#dfsMC = cut(dfsMC, 'PNN', 0.6, None)
dfsMC = cut(dfsMC, 'dijet_pt', 160, None)
dfsMC = cut(dfsMC, 'PNN', 0.45, 0.7)

# %%

bins = np.linspace(x1, x2, 51)
cTot = np.zeros(len(bins)-1)
err = np.zeros(len(bins)-1)
for idx, df in enumerate(dfsMC):
    c=np.histogram(df.dijet_mass, bins=bins, weights=df.weight)[0]
    cerr=np.histogram(df.dijet_mass, bins=bins, weights=(df.weight)**2)[0]
    err = err + cerr

    cTot=cTot+c
err = np.sqrt(err)



x = (bins[1:] + bins[:-1]) / 2
fitregion = ((x > x1) & (x < x2))

least_squares = LeastSquares(x[fitregion], cTot[fitregion], err[fitregion], zPeak3)
m = Minuit(least_squares,
            normSig=cTot.sum()*(bins[1]-bins[0]),
            fraction_dscb=0.58,
           mean=91.61,
           sigma=15.6,
           alphaR=2,
           nR=0.58,
            sigmaG=9,
            #fractionG=0.4,
           )
m.print_level=2
m.limits['mean'] = (83, 97)
m.limits['nR'] = (1e-7, 3)
m.limits['sigma'] = (5, 50)
m.limits['sigmaG'] = (5, 15)
#m.limits['p1'] = (-0.1, 0.1)
m.limits['normSig'] = (cTot.sum()*(bins[1]-bins[0])*0.7, cTot.sum()*(bins[1]-bins[0])*1.3)
m.errors["alphaR"] = 1
m.errors["sigma"] = 3
m.errors["nR"] = 1
#m.errors["normSig"]=50


m.migrad(ncall=20000, iterate=40)
#m.simplex()  
#m.minos()
#m.hesse(ncall=20000)
# %%
cTot = np.zeros(len(bins)-1)
err = np.zeros(len(bins)-1)
fig, ax = plt.subplots(1, 1)
for idx, df in enumerate(dfsMC):
    c=ax.hist(df.dijet_mass, bins=bins, bottom=cTot, weights=df.weight, alpha=0.4)[0]
    cerr=np.histogram(df.dijet_mass, bins=bins, weights=(df.weight)**2)[0]
    err = err + cerr

    cTot=cTot+c
err = np.sqrt(err)
ax.errorbar((bins[1:]+bins[:-1])/2, cTot, err, marker='o', color='black', linestyle='none')

x_draw = np.linspace(x1, x2, 1001)
y_draw = zPeak3(x_draw, *[m.values[p] for p in m.parameters])
y_values = zPeak3(x, *[m.values[p] for p in m.parameters])

chi2_stat = np.sum(((cTot[fitregion] - y_values[fitregion])**2) / err[fitregion]**2)
ndof = len(x[fitregion]) - len(m.parameters)
chi2_pvalue = 1- chi2.cdf(chi2_stat, ndof)
ax.text(x=0.95, y=0.85, s="$\chi^2$/ndof = %.1f/%d\np-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax.transAxes, ha='right')

#ax.hist(x, bins=bins, weights=cTot, histtype='step', label='Data', linewidth=3, color='black')
ax.plot(x_draw, y_draw, label='DSCB + Gaus Fit', color='red', linewidth=1)
ax.legend()
fig.savefig(outFolder+"/plots/zPeakFit_cat2p0.png", bbox_inches='tight')
fit_params = {name: {"value": m.values[name], "error": m.errors[name]} for name in m.parameters}
# %%
with open(outFolder+"/fit_parameters_Z_cat2p0.json", "w") as f:
    json.dump(fit_params, f, indent=4)
# %%

#y_background = m.values["normSig"]*(1-m.values["fraction_gaussian"]-m.values["fraction_dscb"])*background(x_draw, *[m.values[k] for k in ['p1']])
y_gaussian =  m.values["normSig"]*(1-m.values["fraction_dscb"])*gaussianN(x_draw, *[m.values[k] for k in ['mean', 'sigmaG']])
y_dscb = y_gaussian + m.values["normSig"]*m.values["fraction_dscb"]*rscb(x_draw, *[m.values[k] for k in ['mean', 'sigma','alphaR', 'nR']])
y_total = zPeak3(x_draw, *[m.values[k] for k in m.parameters])


fig2, ax2 = plt.subplots(1, 1)
#ax2.fill_between(x_draw, 0, y_background, color='gray', alpha=0.5, label='Background (pol1)')
ax2.fill_between(x_draw, 0, y_gaussian, color='blue', alpha=0.5, label='Gaussian')
ax2.fill_between(x_draw, y_gaussian, y_dscb, color='red', alpha=0.5, label='DSCB')
ax2.errorbar(x, cTot, yerr=err, fmt='ko', label='Data')
ax2.text(x=0.95, y=0.85, s="$\chi^2$/ndof = %.1f/%d\np-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax2.transAxes, ha='right')
ax2.legend()
ax2.set_xlabel("Dijet Mass [GeV]")
ax2.set_ylabel("Normalized Events")
fig2.savefig(outFolder+"/plots/zPeakFit_components_cat2p0.png", bbox_inches='tight')



# %%

integrate.quad(lambda x: gaussianN(x, *[m.values[k] for k in ['mean', 'sigmaG']]), x1, x2)
# %%
#integrate.quad(lambda x: background(x, *[m.values[k] for k in ['p1']]), x1, x2)
# %%
integrate.quad(lambda x: rscb(x, *[m.values[k] for k in [
'mean',
'sigma',
'alphaR',
'nR'
]]), x1, x2)
# %%
