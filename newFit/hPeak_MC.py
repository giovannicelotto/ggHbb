# %%
from functions import loadMultiParquet_v2, loadMultiParquet_Data_new, cut, getDfProcesses_v2
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
from scipy.stats import shapiro, kstest, norm, chi2

# %%
MCList = [0,36]
dfsMC, sumw = loadMultiParquet_v2(paths=MCList, nMCs=[-1, -1], columns=['dijet_mass', 'dijet_pt', 'muon_dxySig', 'sf', 'PU_SF', 'genWeight'], returnNumEventsTotal=True)
dfsMC = cut(dfsMC, 'dijet_pt', 100, None)
# %%
dfProcesses = getDfProcesses_v2()[0].iloc[MCList]
# %%
bins = np.linspace(40, 300, 51)
cTot = np.zeros(len(bins)-1)
err = np.zeros(len(bins)-1)
fig, ax = plt.subplots(1, 1)
for idx, df in enumerate(dfsMC):
    c=ax.hist(df.dijet_mass, bins=bins, bottom=cTot, weights=df.genWeight*df.sf*df.PU_SF*dfProcesses.xsection.iloc[idx]/sumw[idx], label=dfProcesses.process.iloc[idx])[0]
    cerr = np.histogram(df.dijet_mass, bins=bins, 
                        weights=(df.genWeight * df.sf * df.PU_SF * dfProcesses.xsection.iloc[idx] / sumw[idx])**2)[0]
    err += cerr

    cTot=cTot+c
err = np.sqrt(err)
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
def background(x, p0, p1):
    return p0 + p1 * x 

def gaussian(x, normG, mean, sigmaG):
    return normG * np.exp(-0.5 * ((x - mean) / sigmaG) ** 2)

def total_model(x, norm, mean, sigma, alphaL, nL, alphaR, nR, normG, sigmaG, p0, p1):
    return dscb(x, norm, mean, sigma, alphaL, nL, alphaR, nR) + gaussian(x, normG, mean, sigmaG) + background(x, p0, p1)

x = (bins[1:] + bins[:-1]) / 2
fitregion = ((x > x1) & (x < x2) | (x > x3) & (x < x4))

least_squares = LeastSquares(x[fitregion], cTot[fitregion], err[fitregion], total_model)
m = Minuit(least_squares, norm=0.013442275427352348, mean=122.42859549377883, sigma=13.671979058247818,
           alphaL=1.1318785558036752, nL=68.59369897698369, alphaR=50.0, nR=6,
           sigmaG=33.63720814649845, normG=0.0005112412936100314,
           p0=0.0001444, p1=-2.821098953086552e-07)

#m.migrad()
#m.hesse()

x_draw = np.linspace(40, 300, 1001)
y_draw = total_model(x_draw, *[m.values[p] for p in m.parameters])
y_values = total_model(x, *[m.values[p] for p in m.parameters])

chi2_stat = np.sum(((cTot[fitregion] - y_values[fitregion])**2) / err[fitregion]**2)
ndof = len(x[fitregion]) - len(m.parameters)
chi2_pvalue = 1- chi2.cdf(chi2_stat, ndof)
ax.text(x=0.95, y=0.85, s="$\chi^2$/ndof = %.1f/%d\np-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax.transAxes, ha='right')

#ax.hist(x, bins=bins, weights=cTot, histtype='step', label='Data', linewidth=3, color='black')
ax.plot(x_draw, y_draw, 'r-', label='DSCB + Gaus + pol1 Fit')
ax.legend()
plt.show()

# %%

y_gaussian =  background(x_draw, *[m.values[k] for k in ['p0', 'p1']]) + gaussian(x_draw, *[m.values[k] for k in ['normG', 'mean', 'sigmaG']])
y_dscb = y_gaussian + dscb(x_draw, *[m.values[k] for k in ['norm', 'mean', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR']])
y_total = total_model(x_draw, *[m.values[k] for k in m.parameters])
fig2, ax2 = plt.subplots(1, 1)
ax2.fill_between(x_draw, 0, y_gaussian, color='blue', alpha=0.5, label='Gaussian + pol1')
ax2.fill_between(x_draw, y_gaussian, y_dscb, color='red', alpha=0.5, label='DSCB')
ax2.errorbar(x, cTot, yerr=err, fmt='ko', label='MC Z')
ax2.text(x=0.95, y=0.85, s="$\chi^2$/ndof = %.1f/%d\np-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax2.transAxes, ha='right')
ax2.legend()
ax2.set_xlabel("Dijet Mass [GeV]")
ax2.set_ylabel("Normalized Events")
plt.show()
# %%
# %%
