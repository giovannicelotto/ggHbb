import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functions import loadMultiParquet, getXSectionBR
import sys
sys.path.append("/t3home/gcelotto/ggHbb/scripts/plotScripts")
from plotFeatures import plotNormalizedFeatures, cut
import mplhep as hep
hep.style.use("CMS")
def main():
    paths = [
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/**",
            #5.261e+03 +/- 1.432e+02
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-100to200",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-200to400",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-400to600",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-600to800",
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-800toInf",
            #"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/EWKZJets"
            ]
    nReal=1000
    visibilityFactor=100
    dfs, numEventsList = loadMultiParquet(paths=paths, nReal=nReal, nMC=-1, returnNumEventsTotal=True, columns=
                                          ['dijet_mass','sf', 'jet1_pt', 'jet2_pt', 'dijet_pt'  ])

    
    dfs[1].sf = 5.261e+03/numEventsList[1]*dfs[1].sf*1000*0.774*nReal/1017
    dfs[2].sf = 1012/numEventsList[2]*dfs[2].sf*1000*0.774*nReal/1017
    dfs[3].sf = 114.2/numEventsList[3]*dfs[3].sf*1000*0.774*nReal/1017
    dfs[4].sf = 25.34/numEventsList[4]*dfs[4].sf*1000*0.774*nReal/1017
    dfs[5].sf = 12.99/numEventsList[5]*dfs[5].sf*1000*0.774*nReal/1017
    #dfs[5].sf = 9.8/numEventsList[5]*dfs[5].sf*1000*0.774*nReal/1017

    df_Z1 = dfs[1].copy()
    df_Z2 = dfs[2].copy()
    df_Z3 = dfs[3].copy()
    df_Z4 = dfs[4].copy()
    df_Z5 = dfs[5].copy()
    

    dfZBoson = pd.concat((dfs[1], dfs[2], dfs[3], dfs[4], dfs[5]))
    dfs = [dfs[0], dfZBoson, df_Z1, df_Z2, df_Z3, df_Z4, df_Z5]
    



    #dfs=cut(dfs, 'dijet_pt', minPt, None)
    dfs=cut(dfs, 'jet1_pt', 10, None)
    dfs=cut(dfs, 'jet2_pt', 10, None)
    #dfs=cut(dfs, 'dijet_pt', 150, None)
    #dfs=cut(dfs, 'jet2_btagDeepFlavB', None, None)
    
    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [7, 3]}, figsize=(10, 10))
    hep.cms.label(lumi=round(float(0.774*nReal/1017), 4), ax=ax[0])
    bins = np.linspace(40, 200, 80)
    
    counts = np.histogram(dfs[0].dijet_mass, bins=bins)[0]
    ax[0].errorbar(x=(bins[:-1]+bins[1:])/2, y=counts/1., yerr=np.sqrt(counts)/1., color='black', marker='o', linestyle='none', markersize=5)

    counts_sig = np.histogram(dfs[1].dijet_mass, bins=bins, weights=dfs[1].sf)[0]
    counts_lowEnergy1 = np.histogram(dfs[2].dijet_mass, bins=bins, weights=dfs[2].sf)[0]
    counts_lowEnergy2 = np.histogram(dfs[3].dijet_mass, bins=bins, weights=dfs[3].sf)[0]
    counts_lowEnergy3 = np.histogram(dfs[4].dijet_mass, bins=bins, weights=dfs[4].sf)[0]
    counts_lowEnergy4 = np.histogram(dfs[5].dijet_mass, bins=bins, weights=dfs[5].sf)[0]
    counts_lowEnergy5 = np.histogram(dfs[6].dijet_mass, bins=bins, weights=dfs[6].sf)[0]

    ax[0].hist(bins[:-1], bins=bins, weights=counts_lowEnergy1*visibilityFactor/1., label='ZJets HT100-200')
    ax[0].hist(bins[:-1], bins=bins, weights=counts_lowEnergy2*visibilityFactor/1., bottom=counts_lowEnergy1*visibilityFactor/1., label='ZJets HT200-400')
    ax[0].hist(bins[:-1], bins=bins, weights=counts_lowEnergy3*visibilityFactor/1., bottom=counts_lowEnergy1*visibilityFactor/1.+counts_lowEnergy2*visibilityFactor/1., label='ZJets HT400-600')
    ax[0].hist(bins[:-1], bins=bins, weights=counts_lowEnergy4*visibilityFactor/1., bottom=counts_lowEnergy1*visibilityFactor/1.+counts_lowEnergy2*visibilityFactor/1.+counts_lowEnergy3*visibilityFactor/1., label='ZJets HT600-800')
    ax[0].hist(bins[:-1], bins=bins, weights=counts_lowEnergy5*visibilityFactor/1., bottom=counts_lowEnergy1*visibilityFactor/1.+counts_lowEnergy2*visibilityFactor/1.+counts_lowEnergy3*visibilityFactor/1.+counts_lowEnergy4*visibilityFactor/1., label='ZJets HT800-Inf')
    ax[0].errorbar(x=(bins[:-1]+bins[1:])/2, y=counts_sig*visibilityFactor/1., color='blue', label=r'ZJets MC $\times %d$'%visibilityFactor)
    #ax[0].set_yscale('log')

    # do fit
    x1, x2 = 60, 75
    x3, x4 = 115, 170
    x = (bins[1:]+bins[:-1])/2
    fitregion = ((x>x1) & (x<x2) | (x>x3)  & (x<x4))
    coefficients = np.polyfit(x=x[fitregion],
                              y=(counts/1.)[fitregion], deg=8,
                              w=1/(np.sqrt(counts[fitregion])+0.0000001))
    fitted_polynomial = np.poly1d(coefficients)
    y_values = fitted_polynomial(x)


    ax[0].errorbar(x[(x>x1) & (x<x4)], y_values[(x>x1) & (x<x4)], color='red', label='Fitted Background')
    # subtract the fit

    #plot the data - fit
    ax[1].errorbar(x, (counts/1. - y_values), yerr=np.sqrt(counts)/1., color='black', marker='o', linestyle='none')
    ylim  = np.max(abs((counts/1.)[(x>x1) & (x<x4)]-y_values[(x>x1) & (x<x4)]))*1.1
    ax[1].errorbar(x=(bins[:-1]+bins[1:])/2, y=counts_sig/1., color='blue', label='ZJets MC')
    ax[1].hlines(y=0 ,xmin=bins[0], xmax = bins[-1], color='red')
    ax[1].set_ylim(-ylim, ylim)

    ax[0].set_ylabel("Counts / %.1f GeV"%(bins[1]-bins[0]))
    #ax[1].set_ylabel(r"$\frac{\text{Data - Fit }}{\sqrt{\text{Data}}}$")
    ax[1].set_ylabel(r"Data - Fit")
    ax[1].set_xlabel("Dijet Mass [GeV]")


    from matplotlib.patches import Rectangle
    rect1 = Rectangle((x1, ax[0].get_ylim()[0]), x2 - x1, ax[0].get_ylim()[1] - ax[0].get_ylim()[0], alpha=0.1, color='green', label='Fit region')
    rect2 = Rectangle((x3, ax[0].get_ylim()[0]), x4 - x3, ax[0].get_ylim()[1] - ax[0].get_ylim()[0], alpha=0.1, color='green')

    ax[0].add_patch(rect1)
    ax[0].add_patch(rect2)
    ax[0].legend(bbox_to_anchor=(1.05, 1))


    rect1 = Rectangle((x1, ax[1].get_ylim()[0]), x2 - x1, ax[1].get_ylim()[1] - ax[1].get_ylim()[0], alpha=0.1, color='green')
    rect2 = Rectangle((x3, ax[1].get_ylim()[0]), x4 - x3, ax[1].get_ylim()[1] - ax[1].get_ylim()[0], alpha=0.1, color='green')

    ax[1].add_patch(rect1)
    ax[1].add_patch(rect2)
    #plt.gca().add_patch(rect2)




    outName = "/t3home/gcelotto/ggHbb/outputs/plots/massSpectrum/zPeak.png"
    fig.savefig(outName, bbox_inches='tight')
    print("Saving in ",outName )





    return


if __name__ == "__main__":
    main()