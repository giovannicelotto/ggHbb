# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iminuit import cost
from iminuit import Minuit
from functions import loadMultiParquet, cut
import mplhep as hep
hep.style.use("CMS")
from matplotlib.patches import Rectangle
# %%
paths = [
    '/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/**/*',
        '/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-100to200',
         '/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-200to400',
         '/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-400to600',
         '/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-600to800',
         '/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-800toInf',
         ]
columns = ['dijet_pt', 'dijet_mass', 'sf', 'jet1_pt', 'jet2_pt', 'jet2_btagDeepFlavB']
nReal = 1000
dfs, numEventsList = loadMultiParquet(paths=paths, nReal=nReal, nMC=-1, columns=columns, returnNumEventsTotal=True, selectFileNumberList=None, returnFileNumberList=False)

dfs = cut(dfs, 'jet1_pt', 20, None)
dfs = cut(dfs, 'jet2_pt', 20, None)
dfs = cut(dfs, 'jet2_btagDeepFlavB', 0.2, None)
xsections = [#5.261e+03,
    1012, 114.2, 25.34, 12.99, 9.8, ]
print(dfs[1].sf)
for idx, df in enumerate(dfs):
    if idx==0:
        continue
    df.sf = df.sf*xsections[idx-1]/numEventsList[idx]*nReal*0.774/1017*1000
print(dfs[1].sf)
# %%
for dijetptT in [90,95, 100, 105, 110, 115, 120, 125, 130, 135, 140]:
    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
    hep.cms.label(lumi=round(float(0.774*nReal/1017), 4), ax=ax[0])

    dfs = cut(dfs, 'dijet_pt', dijetptT, None)
    ax[0].text(x=0.9, y=0.5, s=r"Dijet p$_{T}$ > %d GeV"%dijetptT, transform=ax[0].transAxes, ha='right')
    bins = np.linspace(50, 200, 25)
    counts = np.histogram(dfs[0].dijet_mass, bins=bins)[0]
    ax[0].hist(bins[:-1], bins=bins, weights=counts,histtype=u'step', color='black')
    ax[0].set_xlim(bins[0], bins[-1])


    ax[0].set_ylabel("Counts / %.1f GeV"%(bins[1]-bins[0]))

    ax[1].set_ylabel(r"Data - Fit")
    ax[1].set_xlabel("Dijet Mass [GeV]")

    x1, x2 = 50, 70
    x3, x4 = 110, 190


    assert (x1>= bins[0])
    x = (bins[1:]+bins[:-1])/2
    fitregion = ((x>x1) & (x<x2) | (x>x3)  & (x<x4))
    from iminuit.cost import LeastSquares
    def mypol(x, p0, p1, p2, p3, p4):
        return p0 + p1*x +p2*x*x + p3*x**3 + p4*x**4
    least_squares = LeastSquares(x[fitregion], (counts)[fitregion], np.sqrt(counts[fitregion]), mypol)
    m = Minuit(least_squares, p0=1e6, p1=-1, p2=0, p3=0, p4=0)


    m.migrad()  # finds minimum of least_squares function
    m.hesse() 
    y_values = mypol(x, p0=m.params['p0'].value,
                     p1=m.params['p1'].value, p2=m.params['p2'].value,
                     p3=m.params['p3'].value, p4=m.params['p4'].value)

    ax[0].errorbar(x[(x>x1) & (x<x4)], y_values[(x>x1) & (x<x4)], color='red', label='Fitted Background')


    # plot data - fit
    dfs = [dfs[0], pd.concat(dfs[1:])]
    counts_sig = np.histogram(dfs[1].dijet_mass, bins=bins, weights=dfs[1].sf)[0]
    visibilityFactor = 10
    ax[0].errorbar(x=(bins[:-1]+bins[1:])/2, y=counts_sig*visibilityFactor, color='blue', label=r'ZJets MC $\times %d$'%visibilityFactor)
    ax[0].errorbar(x, (counts), yerr=np.sqrt(counts), color='black', marker='o', linestyle='none')
    ax[1].errorbar(x, (counts - y_values), yerr=np.sqrt(counts), color='black', marker='o', linestyle='none')

    ax[1].errorbar(x=(bins[:-1]+bins[1:])/2, y=counts_sig, color='blue', label='ZJets MC')
    ax[1].hlines(y=0 ,xmin=bins[0], xmax = bins[-1], color='red')


    ax[0].set_ylim(ax[0].get_ylim())
    ax[1].set_ylim(ax[1].get_ylim())
    rect1 = Rectangle((x1, ax[0].get_ylim()[0]), x2 - x1, ax[0].get_ylim()[1] - ax[0].get_ylim()[0], alpha=0.1, color='green', label='Fit region')
    rect2 = Rectangle((x3, ax[0].get_ylim()[0]), x4 - x3, ax[0].get_ylim()[1] - ax[0].get_ylim()[0], alpha=0.1, color='green')
    ax[0].add_patch(rect1)
    ax[0].add_patch(rect2)
    rect1 = Rectangle((x1, ax[1].get_ylim()[0]), x2 - x1, ax[1].get_ylim()[1] - ax[1].get_ylim()[0], alpha=0.1, color='green')
    rect2 = Rectangle((x3, ax[1].get_ylim()[0]), x4 - x3, ax[1].get_ylim()[1] - ax[1].get_ylim()[0], alpha=0.1, color='green')

    ax[1].add_patch(rect1)
    ax[1].add_patch(rect2)
    ax[0].legend()
    fig.savefig("/t3home/gcelotto/ggHbb/scripts/fitSignals/fit_bkg_Z_1A/fitResult_%d_%d.png"%(nReal/100, dijetptT))
    print("saving")

# %%
