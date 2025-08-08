# %%
from functions import loadMultiParquet_Data_new, loadMultiParquet_v2, getCommonFilters, getDfProcesses_v2
# %%
filter1 = list(getCommonFilters(btagTight=True)[0]) + [('dijet_pt', '>', 100) , ('dijet_mass', '>', 100) , ('dijet_mass', '<', 150)]#, ('nElectrons', '<', 1)]
filter2 = list(getCommonFilters(btagTight=True)[1]) + [('dijet_pt', '>', 100) , ('dijet_mass', '>', 100) , ('dijet_mass', '<', 150)]#, ('nElectrons', '<', 1)]
newFilters = [filter1,filter2]
dfProcesses=getDfProcesses_v2()[0]
# %%
totalEvents = 0
lumi_tot = 0
for i in [0]:
    dfs, lumi =loadMultiParquet_Data_new(dataTaking=[i], nReals=-1, columns=['dijet_mass','dijet_pt'], selectFileNumberList=None, returnFileNumberList=False, filters=newFilters, training=False)
    totalEvents+=len(dfs[0])
    lumi_tot = lumi_tot + lumi
    print("%d at %.2f"%(totalEvents*41.6/lumi_tot, lumi_tot))
# %%
sigEvents = 0
for i in [37]:
    dfsMC, gensumw =loadMultiParquet_v2(paths=[i], nMCs=-1, columns=['dijet_mass','dijet_pt', 'dijet_dPhi', 'sf', 'PU_SF', 'btag_central', 'genWeight', 'jet_pileupId_SF_nom'], returnNumEventsTotal=True, selectFileNumberList=None, returnFileNumberList=False, filters=newFilters, training=False, isJEC=0)
    print(dfProcesses.xsection[i], " pb")
    dfsMC[0]['weight'] = dfsMC[0].genWeight * dfsMC[0].sf * dfsMC[0].PU_SF * dfsMC[0].btag_central * dfsMC[0].jet_pileupId_SF_nom * dfProcesses.xsection[i] * 1000/gensumw
    sigEvents+=dfsMC[0].weight.sum()
    print(sigEvents * 41.6)

# %%
print("S/B", sigEvents/totalEvents)
# %%
import matplotlib.pyplot as plt
import numpy as np
fig,ax = plt.subplots(1, 1)
bins=np.linspace(0, 3, 101)
c=ax.hist(dfsMC[0].dijet_dPhi, bins=bins, density=True, histtype='step')[0]





#ax.hist(dfs[0].dijet_mass, bins=bins, density=True, histtype='step')
# %%
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define Gaussian
def gauss(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Recompute bin centers from your bins
bin_edges = bins
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Mask zero or near-zero bins (to avoid log(0) or bad fits)
mask = c > 0

# Fit
popt, pcov = curve_fit(gauss, bin_centers[mask], c[mask], p0=[1, np.mean(bin_centers), np.std(bin_centers)])

# Plot
plt.figure()
plt.hist(dfsMC[0].dijet_mass, bins=bins, density=True, histtype='step', label='Histogram')
x_fit = np.linspace(bin_centers[0], bin_centers[-1], 1000)
plt.plot(x_fit, gauss(x_fit, *popt), label='Gaussian fit', color='red')
plt.legend()
plt.xlabel('Dijet mass')
plt.ylabel('Density')
plt.title('Gaussian fit to histogram counts')
plt.show()

# Optional: print fit parameters
print(f"Fit results: A = {popt[0]:.3f}, mu = {popt[1]:.3f}, sigma = {popt[2]:.3f}")

# %%
