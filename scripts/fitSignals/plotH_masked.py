
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import loadMultiParquet, cut
import mplhep as hep
hep.style.use("CMS")
from array import array
import ROOT
from scipy.optimize import curve_fit
def fitFunction(x, par):
    mean = par[0]
    sigma = par[1]
    norm = par[2]
    p0 = par[3]
    p1 = par[4]
    p2 = par[5]
    result = 0.0    
    pol = p0 + p1 * x + p1 * x * x
    sig = norm *np.exp((x-mean)/(2*sigma))
    return sig + pol

def pol2(x, p0, p1, p2, p3): 
    pol = p0 + p1 * x + p2 * x * x + p3*x*x*x
    return pol

paths = [ "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/**/"]

xsections = [1]
labels = ['GluGluHToBB']

dfs, numEventsList, fileNumberList = loadMultiParquet(paths=paths, nReal=-2, nMC=-1, columns=['sf', 'dijet_mass', 'dijet_pt', 'jet1_pt', 'jet2_pt'], returnNumEventsTotal=True, selectFileNumberList=None, returnFileNumberList=True)

dfs = cut(dfs, 'jet1_pt', 20, None)
dfs = cut(dfs, 'jet2_pt', 20, None)

bins = np.linspace(0, 350, 80)



df=dfs[0]
counts = np.histogram(df.dijet_mass, bins=bins, weights=df.sf)[0]
errors = np.sqrt(counts)
fig, ax = plt.subplots(1, 1)
ax.hist(bins[:-1], bins=bins, weights=counts)


x = (bins[1:]+bins[:-1])/2
assert len(x)==len(counts), "Check lenght of x and y"

ax.errorbar(x, counts, yerr=errors, xerr=np.diff(bins)/2, linestyle='none', color='red')

m = (counts>0) & ((x>220) | (x<50))


popt, pcov = curve_fit(f=pol2, xdata=x[m], ydata=counts[m],
            p0=[0.1, 0.01, 0.001, 0.0001], sigma=errors[m], full_output=True)[:2]

print(np.diag(pcov))
for (par, err) in zip(popt, np.diag(pcov)):
    print("%.2f +- %.2f"%(par, np.sqrt(err)))

ax.plot(x[m], pol2(x[m], popt[0], popt[1], popt[2], popt[3]))
fig.savefig("/t3home/gcelotto/ggHbb/outputs/plots/fits/higgs_masked.png")
