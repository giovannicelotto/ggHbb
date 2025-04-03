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
def plotFree(x, c, cZ, maskFit, m_tot_2, maskUnblind, bins, dfProcessesMC, dfsMC_H, MCList_H, myBkgSignalFunctions, key, myBkgFunctions, lumi_tot, myBkgParams, plotFolder, ptCut_min, ptCut_max, jet1_btagMin, jet2_btagMin, PNN_t, PNN_t_max=None, fitFunction='zPeak', var=None):
    p_tot = m_tot_2.values
    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(14, 14), constrained_layout=True)
    fig.align_ylabels([ax[0],ax[1]])
    x_plot = np.linspace(bins[0], bins[-1], 500)
    y_tot = myBkgSignalFunctions[key](x_plot, *p_tot)
    ax[0].errorbar(x[maskUnblind], c[maskUnblind], yerr=np.sqrt(c)[maskUnblind], fmt='o', color='black', markersize=3, label="Data")
    ax[0].set_ylim(ax[0].get_ylim())
    ax[0].plot(x_plot, y_tot, label="Fit Region", color='red')
    ax[0].fill_between(x, 0, max(c)*1.2, where=maskFit, color='green', alpha=0.2, label='Fit Region')
    ax[0].set_xlim(bins[0], bins[-1])

    ax[1].errorbar(x[maskFit], (c-myBkgFunctions[key](x, *p_tot[myBkgParams[key]]))[maskFit], yerr=np.sqrt(c)[maskFit], fmt='o', color='black', markersize=3)
    print("FitFunction = ", fitFunction)
    if fitFunction=='zPeak_dscb':
        print(fitFunction)
        ax[1].plot(x, zPeak_dscb(x, *p_tot[[ 'normSig', 'fraction_dscb', 'mean', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR', 'sigmaG']]), color='red', linewidth=2)
    elif fitFunction=='zPeak_rscb':
        ax[1].plot(x, zPeak_rscb(x, *p_tot[[ 'normSig', 'fraction_dscb', 'mean', 'sigma', 'alphaR', 'nR', 'sigmaG']]), color='red', linewidth=2)
    else:
        print(fitFunction)
        assert False
    ax[1].hist(bins[:-1], bins=bins, weights=cZ, label='Z')
    ax[1].set_ylabel("Data - Background")
    #ax[1].set_ylim(ax[1].get_ylim())
    cHiggs = np.zeros(len(bins)-1)
    cHiggs_err = np.zeros(len(bins)-1)
    cHiggs_10 = np.zeros(len(bins)-1)
    visFactor = 10
    for idx, dfMC in enumerate(dfsMC_H):
        c_=np.histogram(dfMC.dijet_mass, bins=bins, weights = dfMC.weight_var)[0]
        label = dfProcessesMC.iloc[MCList_H].process.iloc[idx] + " x %d"%visFactor if visFactor != 1 else dfProcessesMC.iloc[MCList_H].process.iloc[idx]
        c10_ = ax[1].hist(dfMC.dijet_mass, bins=bins, weights = dfMC.weight_var*visFactor , label=label, bottom=cHiggs_10)[0]
        cHiggs += c_
        cerr_=np.histogram(dfMC.dijet_mass, bins=bins, weights = dfMC.weight_var**2)[0]
        cHiggs_err = cHiggs_err + cerr_
        cHiggs_10 += c10_
    cHiggs_err = np.sqrt(cHiggs_err)
    ax[1].set_ylim(ax[1].get_ylim()[0], ax[1].get_ylim()[1]*1.5)
    ax[1].fill_between(x, ax[1].get_ylim()[0], ax[1].get_ylim()[1], where=maskFit, color='green', alpha=0.2)

    hep.cms.label(lumi="%.2f" %lumi_tot, ax=ax[0])
    ax[1].set_xlabel("Dijet Mass [GeV]")
    ax[0].set_ylabel("Counts")
    ax[0].set_ylim(0, ax[0].get_ylim()[1])
    chi2_stat = np.sum(((c[maskFit] - myBkgSignalFunctions[key](x, *p_tot)[maskFit])**2) / np.sqrt(c)[maskFit]**2)
    print("My Chi2 vs iminuit  %.3f vs %.3f"%(chi2_stat, m_tot_2.fval))
    ndof = m_tot_2.ndof
    chi2_pvalue = 1- chi2.cdf(chi2_stat, ndof)
    ax[0].text(x=0.95, y=0.75, s="Fit Sidebands\n$\chi^2$/ndof = %.1f/%d\np-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax[0].transAxes, ha='right', va='top', fontsize=24)
    if ptCut_min is not None and ptCut_max is not None:
        ax[0].text(x=0.95, y=0.5, s="%d < Dijet p$_T$ < %d GeV\nJet1 bTag > %.2f\nJet2 bTag > %.2f\nNN score > %.3f\n"%(ptCut_min, ptCut_max, jet1_btagMin, jet2_btagMin, PNN_t), transform=ax[0].transAxes, ha='right', va='top', fontsize=24)
    else:
        if PNN_t_max is not None:
            ax[0].text(x=0.95, y=0.5, s="Dijet p$_T$ > %d GeV\nJet1 bTag > %.2f\nJet2 bTag > %.2f\n%.3f< NN score < %.2f\n"%(ptCut_min, jet1_btagMin, jet2_btagMin, PNN_t, PNN_t_max), transform=ax[0].transAxes, ha='right', va='top', fontsize=24)
        else:
            ax[0].text(x=0.95, y=0.5, s="Dijet p$_T$ > %d GeV\nJet1 bTag > %.2f\nJet2 bTag > %.2f\nNN score > %.2f\n"%(ptCut_min, jet1_btagMin, jet2_btagMin, PNN_t), transform=ax[0].transAxes, ha='right', va='top', fontsize=24) 
    ax[0].tick_params(labelsize=24)
    ax[1].tick_params(labelsize=24)
    ax[0].legend(fontsize=24)
    ax[1].legend(ncols=3)
    if var=='nominal':
        
        fig.savefig(plotFolder+"/mjjFit2_cat1.png", bbox_inches='tight')
    return cHiggs, cHiggs_err




    