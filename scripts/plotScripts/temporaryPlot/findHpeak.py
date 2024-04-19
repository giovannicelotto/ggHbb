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
            "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/**",
            #"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-200to400",
            #"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-400to600",
            #"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-600to800",
            #"/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-800toInf"
            ]
    nReal=1017
    dfs, numEventsList = loadMultiParquet(paths=paths, nReal=nReal, nMC=-1, returnNumEventsTotal=True, columns=
    ['dijet_mass', 'ht', 'dijet_dR', 'sf', 'jet1_pt', 'jet2_pt', 'dijet_pt', 'jet1_btagDeepFlavB', 'jet2_btagDeepFlavB', 'dijet_cs', 'dijet_dEta' ])

    
    dfs[1].sf = getXSectionBR()/numEventsList[1]*dfs[1].sf*1000*0.774

    #dfs=cut(dfs, 'dijet_mass', 90, 160)
    dfs=cut(dfs, 'jet2_btagDeepFlavB', 0.2, None)
    dfs=cut(dfs, 'jet2_pt', 20, None)
    dfs=cut(dfs, 'dijet_dEta', None, 3)
    dfs=cut(dfs, 'dijet_cs', -0.6, 0.6)
    #dfs=cut(dfs, 'ht', 300, None)
    #dfs=cut(dfs, 'dijet_dR', None, 2.5)


    #dfs=cut(dfs, 'dijet_pt', 80, None)
    #dfs=cut(dfs, 'jet1_qgl', 0.1, None)
    #dfs=cut(dfs, 'jet2_qgl', 0.15, None)
    #plotNormalizedFeatures(dfs, outFile="/t3home/gcelotto/ggHbb/outputs/plots/features/HvsData.png", legendLabels=['Data', 'ZJets'], colors=['red', 'blue'], figsize=(15, 30))
    print(dfs[1].sf.sum()/np.sqrt(dfs[0].sf.sum()))

    if (True):
        fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10))
        hep.cms.label(lumi=round(float(0.774*nReal/1017), 4), ax=ax[0])
        bins = np.linspace(0, 450, 100)
        counts = np.histogram(dfs[0].dijet_mass, bins=bins)[0]
        ax[0].errorbar(x=(bins[:-1]+bins[1:])/2, y=counts, yerr=np.sqrt(counts), color='black', marker='o', linestyle='none', markersize=5)

        counts_sig = np.histogram(dfs[1].dijet_mass, bins=bins, weights=dfs[1].sf)[0]
        ax[0].errorbar(x=(bins[:-1]+bins[1:])/2, y=counts_sig, color='blue', label='ggH MC')
        ax[0].set_yscale('log')

        # do fit
        x1, x2 = 55, 75
        x3, x4 = 160, 300
        #x1, x2 = 90, 150
        #x3, x4 = 200, 300
        x = (bins[1:]+bins[:-1])/2
        fitregion = ((x>x1) & (x<x2) | (x>x3)  & (x<x4))
        coefficients = np.polyfit(x[fitregion], counts[fitregion], 12)
        fitted_polynomial = np.poly1d(coefficients)
        y_values = fitted_polynomial(x)

        print(y_values[fitregion])

        ax[0].errorbar(x[(x>x1) & (x<x4)], y_values[(x>x1) & (x<x4)], color='red', label='Fitted Background')
        # subtract the fit

        #plot the data - fit
        #ax[1].errorbar(x, (counts - y_values)/(np.sqrt(counts)+0.0000001), color='black', marker='o', linestyle='none')
        ax[1].errorbar(x, (counts - y_values), yerr=np.sqrt(counts), color='black', marker='o', linestyle='none')
        ax[1].set_ylim(-10000, 10000)

        ax[0].legend()
        ax[0].set_ylabel("Counts / %.1f GeV"%(bins[1]-bins[0]))
        #ax[1].set_ylabel(r"$\frac{\text{Data - Fit }}{\sqrt{\text{Data}}}$")
        ax[1].set_ylabel(r"Data - Fit")
        ax[1].set_xlabel("Dijet Mass [GeV]")


        from matplotlib.patches import Rectangle
        rect1 = Rectangle((x1, ax[0].get_ylim()[0]), x2 - x1, ax[0].get_ylim()[1] - ax[0].get_ylim()[0], alpha=0.1, color='green')
        rect2 = Rectangle((x3, ax[0].get_ylim()[0]), x4 - x3, ax[0].get_ylim()[1] - ax[0].get_ylim()[0], alpha=0.1, color='green')

        ax[0].add_patch(rect1)
        ax[0].add_patch(rect2)


        rect1 = Rectangle((x1, ax[1].get_ylim()[0]), x2 - x1, ax[1].get_ylim()[1] - ax[1].get_ylim()[0], alpha=0.1, color='green')
        rect2 = Rectangle((x3, ax[1].get_ylim()[0]), x4 - x3, ax[1].get_ylim()[1] - ax[1].get_ylim()[0], alpha=0.1, color='green')

        ax[1].add_patch(rect1)
        ax[1].add_patch(rect2)
    #plt.gca().add_patch(rect2)





        fig.savefig("/t3home/gcelotto/ggHbb/outputs/plots/massSpectrum/hPeak.png", bbox_inches='tight')





    return


if __name__ == "__main__":
    main()