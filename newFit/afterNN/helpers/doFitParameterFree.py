import numpy as np
import matplotlib.pyplot as plt
import json
from iminuit.cost import LeastSquares
from iminuit import Minuit
from scipy.stats import chi2
import mplhep as hep
import sys
sys.path.append("/t3home/gcelotto/ggHbb/newFit/afterNN/helpers")
from allFunctions import *
hep.style.use("CMS")

def doFitParameterFree(x,m_tot, c, cZ, maskFit, maskUnblind, bins, dfMC_Z, dfProcessesMC, MCList_Z, dfMC_H, MCList_H, myBkgSignalFunctions, key, myBkgFunctions, lumi_tot, myBkgParams, plotFolder, fitFunction, paramsLimits):
    x_fit = x[maskFit]
    y_tofit = c[maskFit]
    yerr = np.sqrt(y_tofit)

    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(18, 15), constrained_layout=True)
    fig.align_ylabels([ax[0],ax[1]])
    ax[0].errorbar(x[maskUnblind], c[maskUnblind], yerr=np.sqrt(c)[maskUnblind], fmt='o', color='black', markersize=3, label="Data")
    # Plot Z
    bot = np.zeros(len(bins)-1)
    c_=ax[0].hist(dfMC_Z.dijet_mass_, bins=bins, weights = dfMC_Z.weight_, label='Zbb', bottom=bot)[0]
    bot += c_
    cumulativeMC = bot.copy()
    cHiggs = np.zeros(len(bins)-1)
    for process in np.unique(dfMC_H.process):
        maskProcess = dfMC_H.process==process

        c_=ax[0].hist(dfMC_H.dijet_mass_[maskProcess], bins=bins, weights = dfMC_H.weight_ [maskProcess], label=process, bottom=cumulativeMC)[0]
        cHiggs += c_
        cumulativeMC +=c_

    least_squares = LeastSquares(x_fit, y_tofit, yerr, myBkgSignalFunctions[key])
    params = {}
    for par in m_tot.parameters:
        params[par] = m_tot.values[par]
    m_tot_2 = Minuit(least_squares,
                   **params
                   )
    for par in m_tot_2.parameters:
        if par in paramsLimits:
            m_tot_2.limits[par] = paramsLimits[par]  # Assign limits from the dictionary
        else:
            m_tot_2.limits[par] = None  # No limits if not specified
    m_tot_2.limits['normSig'] = (m_tot.values["normSig"]/3, 2*m_tot.values["normSig"])
    for par in m_tot_2.parameters:
        #print(par)
        #if ((par=="normSig") | (par in myBkgParams[key])):
        if ( par in myBkgParams[key]):
        #if ( par in myBkgParams[key]) | (par=="normSig"):
            #print("Excluding!")
            continue
        else:
            m_tot_2.fixed[par]=True
    




    m_tot_2.migrad()
    m_tot_2.hesse()

    p_tot = m_tot_2.values

    # Generate fit curves

    x_plot = np.linspace(bins[0], bins[-1], 500)
    y_tot = myBkgSignalFunctions[key](x_plot, *p_tot)



    ax[0].plot(x_plot, y_tot, label="Fit (Background + Z Peak)", color='red')
    ax[0].fill_between(x, 0, max(c)*1.2, where=maskFit, color='green', alpha=0.2)
    ax[0].set_xlim(bins[0], bins[-1])
    ax[0].set_ylim(1, max(c)*1.2)

    ax[1].errorbar(x[maskFit], (c-myBkgFunctions[key](x, *p_tot[myBkgParams[key]]))[maskFit], yerr=np.sqrt(c)[maskFit], fmt='o', color='black', markersize=3)
    ax[1].set_ylim(ax[1].get_ylim())
    if fitFunction=="zPeak_dscb":
        ax[1].plot(x, zPeak_dscb(x, *p_tot[[ 'normSig', 'fraction_dscb', 'mean', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR', 'sigmaG']]), color='red', linewidth=2)
    elif fitFunction=="zPeak_rscb":
        ax[1].plot(x, zPeak_rscb(x, *p_tot[[ 'normSig', 'fraction_dscb', 'mean', 'sigma', 'alphaR', 'nR', 'sigmaG']]), color='red', linewidth=2)
    ax[1].set_ylabel("Data - Background")
    ax[1].fill_between(x, ax[1].get_ylim()[0], ax[1].get_ylim()[1], where=maskFit, color='green', alpha=0.2)

    ax[1].hist(bins[:-1], bins=bins, weights = cZ)[0]
    ax[1].hist(bins[:-1], bins=bins, weights = cHiggs, bottom=cZ)[0]

    hep.cms.label(lumi="%.2f" %lumi_tot, ax=ax[0])
    ax[1].set_xlabel("Dijet Mass [GeV]")
    ax[0].set_ylabel("Counts")
    ax[0].legend(bbox_to_anchor=(1, 1))

    chi2_stat = np.sum(((c[maskFit] - myBkgSignalFunctions[key](x, *p_tot)[maskFit])**2) / np.sqrt(c)[maskFit]**2)
    ndof = m_tot_2.ndof
    chi2_pvalue = 1- chi2.cdf(chi2_stat, ndof)
    ax[0].text(x=0.05, y=0.75, s="$\chi^2$/ndof = %.1f/%d\np-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax[0].transAxes, ha='left')
    ax[0].set_yscale('log')

    #fig.savefig(plotFolder+"/mjjFit_cat1.png", bbox_inches='tight')

    plt.close()
    return m_tot_2