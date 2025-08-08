import numpy as np
import matplotlib.pyplot as plt
import json
from iminuit.cost import LeastSquares
from iminuit import Minuit
from scipy.stats import chi2
import mplhep as hep
hep.style.use("CMS")
import sys
sys.path.append("/t3home/gcelotto/ggHbb/newFit/afterNN/helpers")
from allFunctions import *
def plotFree(x, c, cZ, maskFit, m_tot_2, maskUnblind, bins, dfProcessesMC, dfMC_H, MCList_H, myBkgSignalFunction, key, myBkgFunction, lumi_tot, myBkgParams, plotFolder, config_cuts, ptCut_min, ptCut_max, jet1_btagMin, jet2_btagMin, PNN_t, PNN_t_max=None, fitFunction='zPeak', var=None, blind=True, myBkgFunctionZ_H=None):
    p_tot = m_tot_2.values
    fig, ax = plt.subplots(nrows=3, sharex=True, gridspec_kw={'height_ratios': [4, 1, 1]}, figsize=(14, 14), constrained_layout=True)
    fig.align_ylabels([ax[0],ax[1], ax[2]])
    x_plot = np.linspace(bins[0], bins[-1], 500)

    y_tot = myBkgFunctionZ_H(x_plot, *p_tot)
    ax[0].errorbar(x[maskUnblind], c[maskUnblind], yerr=np.sqrt(c)[maskUnblind], fmt='o', color='black', markersize=3, label="Data")
    #ax[0].errorbar(x[maskUnblind], c[maskUnblind], yerr=np.sqrt(c)[maskUnblind], fmt='o', color='black', markersize=3, label="Data")
    ax[0].set_ylim(ax[0].get_ylim())
    ax[0].plot(x_plot, y_tot, label="S+B Fit", color='red')
    ax[0].plot(x_plot, myBkgFunction(x_plot, *p_tot[myBkgParams]), label="B Fit", color='red', linestyle='dashed')
    
    ax[0].fill_between(x, 0, max(c)*1.2, where=maskFit, color='green', alpha=0.2, label='Fit Region')
    ax[0].set_xlim(bins[0], bins[-1])

    #ax[1].errorbar(x[maskFit], (c-myBkgFunction(x, *p_tot[myBkgParams]))[maskFit], yerr=np.sqrt(c)[maskFit], fmt='o', color='black', markersize=3)
    ax[1].errorbar(x[maskUnblind], (c-myBkgFunction(x, *p_tot[myBkgParams]))[maskUnblind], yerr=np.sqrt(c)[maskUnblind], fmt='o', color='black', markersize=3)
    print("FitFunction = ", fitFunction)
    visFactor = 10
    if fitFunction=='zPeak_dscb':
        if blind:
            ax[1].plot(x, zPeak_dscb(x, *p_tot[[ 'normSig', 'fraction_dscb', 'mean', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR', 'sigmaG']]), color='red', linewidth=2)
        else:
            ax[1].plot(x, zPeak_dscb(x, *p_tot[[ 'normSig', 'fraction_dscb', 'mean', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR', 'sigmaG']])+
                       visFactor*zPeak_dscb(x, *p_tot[[ 'normSig_H', 'fraction_dscb_H', 'mean_H', 'sigma_H', 'alphaL_H', 'nL_H', 'alphaR_H', 'nR_H', 'sigmaG_H']]), color='red', linewidth=2)
    elif fitFunction=='zPeak_rscb':
        ax[1].plot(x, zPeak_rscb(x, *p_tot[[ 'normSig', 'fraction_dscb', 'mean', 'sigma', 'alphaR', 'nR', 'sigmaG']]), color='red', linewidth=2)
    else:
        assert False
    ax[1].hist(bins[:-1], bins=bins, weights=cZ, label='Z')
    ax[1].set_ylabel("Data - nonRes")
    #ax[1].set_ylim(ax[1].get_ylim())
    cHiggs = np.zeros(len(bins)-1)
    cHiggs_err = np.zeros(len(bins)-1)
    cHiggs_10 = cZ
    
    for process in np.unique(dfMC_H.process):
        maskProcess = dfMC_H.process==process
        c_=np.histogram(dfMC_H[maskProcess].dijet_mass_, bins=bins, weights = dfMC_H[maskProcess].weight_)[0]
        label = process + " x %d"%visFactor if visFactor != 1 else process
        c10_ = ax[1].hist(dfMC_H[maskProcess].dijet_mass_, bins=bins, weights = dfMC_H[maskProcess].weight_*visFactor , label=label, bottom=cHiggs_10)[0]
        cHiggs += c_
        cerr_=np.histogram(dfMC_H[maskProcess].dijet_mass_, bins=bins, weights = dfMC_H[maskProcess].weight_**2)[0]
        cHiggs_err = cHiggs_err + cerr_
        cHiggs_10 += c10_
    cHiggs_err = np.sqrt(cHiggs_err)
    ax[1].set_ylim(ax[1].get_ylim()[0], ax[1].get_ylim()[1]*1.5)
    ax[1].fill_between(x, ax[1].get_ylim()[0], ax[1].get_ylim()[1], where=maskFit, color='green', alpha=0.2)


    ax[2].hist(bins[:-1], bins=bins, weights = cHiggs)
    ax[2].set_ylim(-max(cHiggs)*5, max(cHiggs)*8)
    ax[2].plot(x, zPeak_dscb(x, *p_tot[[ 'normSig_H', 'fraction_dscb_H', 'mean_H', 'sigma_H', 'alphaL_H', 'nL_H', 'alphaR_H', 'nR_H', 'sigmaG_H']]), color='blue', linewidth=2)
    ax[2].errorbar(x[maskUnblind], (c-myBkgFunction(x, *p_tot[myBkgParams])  \
                                    - zPeak_dscb(x, *p_tot[[ 'normSig', 'fraction_dscb', 'mean', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR', 'sigmaG']]))[maskUnblind], yerr=np.sqrt(c)[maskUnblind], fmt='o', color='black', markersize=3)
    ax[2].fill_between(x, -99999, 9999, where=maskFit, color='green', alpha=0.2, label='Fit Region')
    hep.cms.label(lumi="%.2f" %lumi_tot, ax=ax[0])
    ax[2].set_xlabel("Dijet Mass [GeV]")
    ax[2].set_ylabel("Data - Bkg")
    ax[0].set_ylabel("Counts / %.1f GeV"%(bins[1]-bins[0]))
    ax[0].set_ylim(0, ax[0].get_ylim()[1])

    chi2_stat = np.sum(((c[maskFit] - myBkgFunctionZ_H(x, *p_tot)[maskFit])**2) / np.sqrt(c)[maskFit]**2)
    ndof = m_tot_2.ndof
    chi2_pvalue = 1- chi2.cdf(chi2_stat, ndof)
    print("pval %.2f"%chi2_pvalue)
    #print("My Chi2 vs iminuit  %.3f vs %.3f"%(chi2_stat, m_tot_2.fval))
    ax[0].text(x=0.95, y=0.75, s="Fit Sidebands\n$\chi^2$/ndof = %.1f/%d\np-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax[0].transAxes, ha='right', va='top', fontsize=24)
    
    if config_cuts["cuts_dict"]["dijet_pt"][0] is not None and config_cuts["cuts_dict"]["dijet_pt"][1] is not None:
        s="%d < Dijet p$_T$ < %d GeV"%(config_cuts["cuts_dict"]["dijet_pt"][0], config_cuts["cuts_dict"]["dijet_pt"][1])
    elif config_cuts["cuts_dict"]["dijet_pt"][0] is not None and config_cuts["cuts_dict"]["dijet_pt"][1] is None:
        s="Dijet p$_T$ > %d GeV"%(config_cuts["cuts_dict"]["dijet_pt"][0])
    if (config_cuts["cuts_dict"]["jet1_btagDeepFlavB"][0]==0.71) & (config_cuts["cuts_dict"]["jet1_btagDeepFlavB"][1] is None):
        s=s+"\nJet1 bTag > T"
    elif (config_cuts["cuts_dict"]["jet1_btagDeepFlavB"][0]==0.2783) & (config_cuts["cuts_dict"]["jet1_btagDeepFlavB"][1]==0.71):
        s=s+"\nM < Jet1 bTag < T"
    elif (config_cuts["cuts_dict"]["jet1_btagDeepFlavB"][0]==0.2783) & (config_cuts["cuts_dict"]["jet1_btagDeepFlavB"][1] is None):
        s=s+"\nJet1 bTag > M"
    if (config_cuts["cuts_dict"]["jet2_btagDeepFlavB"][0]==0.71) & (config_cuts["cuts_dict"]["jet2_btagDeepFlavB"][1] is None):
        s=s+"\nJet2 bTag > T"
    elif (config_cuts["cuts_dict"]["jet2_btagDeepFlavB"][0]==0.2783) & (config_cuts["cuts_dict"]["jet2_btagDeepFlavB"][1]==0.71):
        s=s+"\nM < Jet2 bTag < T"
    elif (config_cuts["cuts_dict"]["jet2_btagDeepFlavB"][0]==0.2783) & (config_cuts["cuts_dict"]["jet2_btagDeepFlavB"][1] is None):
        s=s+"\nJet2 bTag > M"
    if (config_cuts["cuts_dict"]["PNN"][0] is not None) & (config_cuts["cuts_dict"]["PNN"][1] is None):
        s=s+"\nNN score > %.3f"%config_cuts["cuts_dict"]["PNN"][0]
    elif (config_cuts["cuts_dict"]["PNN"][0]is not None) & (config_cuts["cuts_dict"]["PNN"][1] is not None):
        s=s+"\n%.3f < NN score < %.3f"%(config_cuts["cuts_dict"]["PNN"][0], config_cuts["cuts_dict"]["PNN"][1])
                                  
    ax[0].text(x=0.95, y=0.5, s=s, transform=ax[0].transAxes, ha='right', va='top', fontsize=24)
    ax[0].tick_params(labelsize=24)
    ax[1].tick_params(labelsize=24)
    ax[0].legend(fontsize=24)
    ax[1].legend(ncols=3)
    #if (var=='nominal') | ("JER" in var):
    if "JEC" in var:
        fig.savefig(plotFolder+"/JEC/mjjFit_%s.png"%var, bbox_inches='tight')
    else:
        fig.savefig(plotFolder+"/mjjFit_%s_%d.png"%(var, key), bbox_inches='tight')
    plt.close()
    return cHiggs, cHiggs_err




    