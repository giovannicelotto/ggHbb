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

def doFitParameterFree(x, m_tot, c, cZ, maskFit, maskUnblind, bins, dfMC_Z, dfMC_H, myBkgSignalFunction, myBkgFunctions, lumi_tot, myBkgParams,
                       fitFunction_Z, fit_parameters_hPeak,var,
                       paramsLimits, lockNorm=True, blind=True,
                       myBkgFunctionZ_H=None):

    x_fit = x[maskFit]
    y_tofit = c[maskFit]

    
    yerr = np.sqrt(y_tofit)

    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(18, 15), constrained_layout=True)
    fig.align_ylabels([ax[0],ax[1]])
    # Plot Data
    ax[0].errorbar(x[maskUnblind], c[maskUnblind], yerr=np.sqrt(c)[maskUnblind], fmt='o', color='black', markersize=3, label="Data")
    # Plot Z
    bot = np.zeros(len(bins)-1)
    c_=ax[0].hist(dfMC_Z.dijet_mass_, bins=bins, weights = dfMC_Z.weight_, label='Zbb', bottom=bot)[0]
    bot += c_
    cumulativeMC = bot.copy()
    # Plot higgs
    cHiggs = np.zeros(len(bins)-1)
    for process in np.unique(dfMC_H.process):
        maskProcess = dfMC_H.process==process

        c_=ax[0].hist(dfMC_H.dijet_mass_[maskProcess], bins=bins, weights = dfMC_H.weight_ [maskProcess], label=process, bottom=cumulativeMC)[0]
        cHiggs += c_
        cumulativeMC +=c_

    # Initialize Fit
    least_squares = LeastSquares(x_fit, y_tofit, yerr, myBkgFunctionZ_H)
    params = {}
    for par in m_tot.parameters:
        if par=='normSig':
            params[par] = m_tot.values[par]    
        else:
            params[par] = m_tot.values[par]

    m_tot_2 = Minuit(least_squares,
                   **params)
    print("\n"*4, "Iniital Params")
    print(params)

    
    print("\n"*4)
    
    # Set limits based on config file
    for par in m_tot_2.parameters:
        if par in paramsLimits:
            m_tot_2.limits[par] = paramsLimits[par]  # Assign limits from the dictionary
        else:
            m_tot_2.limits[par] = None  # No limits if not specified

    m_tot_2.limits['normSig'] = (m_tot.values["normSig"]*0.1, 1.5*m_tot.values["normSig"])
    m_tot_2.limits['normSig_H'] = (0, 5*m_tot.values["normSig_H"])
    for par in m_tot_2.parameters:
        #if par=="B":
        #    m_tot_2.fixed[par]=True

        if (par=='normSig_H'):
        #if (par=='normSig_H'):
            m_tot_2.fixed[par]=True
        if lockNorm:
            if ( par in myBkgParams):
                m_tot_2.fixed[par]=False
                continue
            else:
                print(par, " was fixed: not in myBkgParamss")
                m_tot_2.fixed[par]=True
        else:
            if ( par in myBkgParams) | (par=="normSig") | (par=="normSig_H"):
                continue
            else:
                print(par, " was fixed")
                m_tot_2.fixed[par]=True
        #print(par)
        #if ((par=="normSig") | (par in myBkgParams)):
            #print("Excluding!")
        
    




    m_tot_2.migrad()
    m_tot_2.hesse()

    print(m_tot_2)
    print("Signal Strenght : ",m_tot_2.values["normSig"]/m_tot.values["normSig"], "\n"*5)
    p_tot = m_tot_2.values

    # Generate fit curves

    x_plot = np.linspace(bins[0], bins[-1], 500)
    y_tot = myBkgFunctionZ_H(x_plot, *p_tot)



    ax[0].plot(x_plot, y_tot, label="Fit (Background + Z Peak)", color='red')
    ax[0].fill_between(x, 0, max(c)*1.2, where=maskFit, color='green', alpha=0.2)
    ax[0].set_xlim(bins[0], bins[-1])
    ax[0].set_ylim(1, max(c)*1.2)

    ax[1].errorbar(x[maskFit], (c-myBkgFunctions(x, *p_tot[myBkgParams]))[maskFit], yerr=np.sqrt(c)[maskFit], fmt='o', color='black', markersize=3)
    ax[1].set_ylim(ax[1].get_ylim())
    if blind:
        if fitFunction_Z=="zPeak_dscb":
            ax[1].plot(x, zPeak_dscb(x, *p_tot[[ 'normSig', 'fraction_dscb', 'mean', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR', 'sigmaG']]), color='red', linewidth=2)
        elif fitFunction_Z=="zPeak_rscb":
            ax[1].plot(x, zPeak_rscb(x, *p_tot[[ 'normSig', 'fraction_dscb', 'mean', 'sigma', 'alphaR', 'nR', 'sigmaG']]), color='red', linewidth=2)
    else:
        if fitFunction_Z=="zPeak_dscb":
            ax[1].plot(x, zPeak_dscb(x, *p_tot[[ 'normSig', 'fraction_dscb', 'mean', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR', 'sigmaG']]) +
                       zPeak_dscb(x, *p_tot[[ 'normSig_H', 'fraction_dscb_H', 'mean_H', 'sigma_H', 'alphaL_H', 'nL_H', 'alphaR_H', 'nR_H', 'sigmaG_H']]), color='red', linewidth=2)
        elif fitFunction_Z=="zPeak_rscb":
            ax[1].plot(x, zPeak_rscb(x, *p_tot[[ 'normSig', 'fraction_dscb', 'mean', 'sigma', 'alphaR', 'nR', 'sigmaG']]), color='red', linewidth=2)
    ax[1].set_ylabel("Data - Background")
    ax[1].fill_between(x, ax[1].get_ylim()[0], ax[1].get_ylim()[1], where=maskFit, color='green', alpha=0.2)

    ax[1].hist(bins[:-1], bins=bins, weights = cZ)[0]
    ax[1].hist(bins[:-1], bins=bins, weights = cHiggs, bottom=cZ)[0]

    hep.cms.label(lumi="%.2f" %lumi_tot, ax=ax[0])
    ax[1].set_xlabel("Dijet Mass [GeV]")
    ax[0].set_ylabel("Counts")
    ax[0].legend(bbox_to_anchor=(1, 1))

    chi2_stat = np.sum(((c[maskFit] - myBkgFunctionZ_H(x, *p_tot)[maskFit])**2) / np.sqrt(c)[maskFit]**2)
        
    ndof = m_tot_2.ndof
    chi2_pvalue = 1- chi2.cdf(chi2_stat, ndof)
    ax[0].text(x=0.05, y=0.75, s="$\chi^2$/ndof = %.1f/%d\np-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax[0].transAxes, ha='left')
    ax[0].set_yscale('log')

    #fig.savefig(plotFolder+"/mjjFit_cat1.png", bbox_inches='tight')

    plt.close()
    return m_tot_2