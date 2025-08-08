import numpy as np
import matplotlib.pyplot as plt
import json
from iminuit.cost import LeastSquares
from iminuit import Minuit
from scipy.stats import chi2
import mplhep as hep
import sys
import inspect
sys.path.append("/t3home/gcelotto/ggHbb/newFit/afterNN/helpers")
from allFunctions import *
hep.style.use("CMS")

def doFitParameterFixed(x,x1,x2, c, maskFit, maskUnblind, bins, myBkgFunctionZ_H, key, dfMC_Z, dfMC_H, myBkgFunction, lumi_tot, myBkgParams, plotFolder, fit_parameters_zPeak, fit_parameters_hPeak, var, params, paramsLimits):
    set_x_bounds(x1, x2)
    x_fit = x[maskFit]
    y_tofit = c[maskFit]
    yerr = np.sqrt(y_tofit)

    # Plot Data
    fig, ax = plt.subplots(nrows=3, sharex=True, gridspec_kw={'height_ratios': [4, 1, 1]}, figsize=(18, 15), constrained_layout=True)
    fig.align_ylabels([ax[0],ax[1], ax[2]])
    ax[0].errorbar(x[maskUnblind], c[maskUnblind], yerr=np.sqrt(c)[maskUnblind], fmt='o', color='black', markersize=3, label="Data")
    # Plot Z

    cZ=ax[0].hist(dfMC_Z.dijet_mass_, bins=bins, weights = dfMC_Z.weight_, label='Zbb')[0]
    cZ_err=np.histogram(dfMC_Z.dijet_mass_, bins=bins, weights = (dfMC_Z.weight_)**2)[0]
    cZ_err = np.sqrt(cZ_err)
    cumulativeMC=cZ.copy()
    
    cHiggs = np.zeros(len(bins)-1)
    for process in np.unique(dfMC_H.process):
        maskProcess = dfMC_H.process==process
        c_=ax[0].hist(dfMC_H[maskProcess].dijet_mass_, bins=bins, weights = dfMC_H.weight_[maskProcess] , label=process, bottom=cumulativeMC)[0]
        cHiggs += c_
        cumulativeMC +=c_



    # Minuit fits

    params["normBkg"] = c.sum()*(bins[1]-bins[0])-cZ.sum()*(bins[1]-bins[0])
    params["normSig"] = (cZ.sum()*(bins[1]-bins[0]) -  dfMC_Z[dfMC_Z.dijet_mass_<bins[0]].weight_.sum())
    params["normSig_H"] = cHiggs.sum()*(bins[1]-bins[0]) -  dfMC_H[dfMC_H.dijet_mass_<bins[0]].weight_.sum()

    
    least_squares = LeastSquares(x_fit, y_tofit, yerr, myBkgFunctionZ_H)
    allPars = inspect.signature(myBkgFunctionZ_H)
    allPars = list(allPars.parameters.keys())
    #print("These are all params")
    #print(allPars)
    for par in allPars:
        print(par)

        if par in params.keys():
            # parameter was already initialized
            continue
        elif par in myBkgParams:
            params[par]=0
            continue
        elif par=='x':
            continue
        else:
            if par in fit_parameters_zPeak['parameters'][var].keys():
                params[par]=fit_parameters_zPeak['parameters'][var][par]['value']
            elif par.endswith('_H'):
                params[par]=fit_parameters_hPeak['parameters'][var][par[:-2]]['value']

    
    #print("Final parameters")
    #print(params)
    #print(params)
    m_tot = Minuit(least_squares,
                    **params
                    )
    #m_tot.fixed["B"]=True
    for par in m_tot.parameters:
        if par in myBkgParams:
            #print(par, "not fixed")
            continue
        else:
            m_tot.fixed[par]=True

        # Apply limits using the dictionary
    for par in m_tot.parameters:
        #m_tot.fixed[par] = True
        #print(par)
        if par in paramsLimits:
            m_tot.limits[par] = paramsLimits[par]  # Assign limits from the dictionary
        else:
            m_tot.limits[par] = None  # No limits if not specified


    
    set_x_bounds(x1, x2)
    m_tot.migrad(ncall=100000, iterate=200)
    m_tot.hesse()

    p_tot = m_tot.values
    #print(p_tot)

    # Generate fit curves
    x_plot  = np.linspace(bins[0], bins[-1], 500)
    y_tot   = myBkgFunctionZ_H(x_plot, *p_tot)



    ax[0].plot(x_plot, y_tot, label="Fit (Background + Z Peak)", color='red')
    ax[0].fill_between(x, 0, max(c)*1.2, where=maskFit, color='green', alpha=0.2)
    ax[0].set_xlim(bins[0], bins[-1])
    ax[0].set_ylim(1, max(c)*1.2)


    ax[1].errorbar(x[maskUnblind], (c-myBkgFunction(x, *p_tot[myBkgParams]))[maskUnblind], yerr=np.sqrt(c)[maskUnblind], fmt='o', color='black', markersize=3)
#    else:
#        ax[1].errorbar(x, (c-myBkgFunction(x, *p_tot[myBkgParams])), yerr=np.sqrt(c), fmt='o', color='black', markersize=3)
    ax[1].set_ylim(ax[1].get_ylim())
    if fit_parameters_zPeak["fitFunction"]=='zPeak_dscb':
        #print(fit_parameters_zPeak['parameters']['nominal']['alphaL']['value'])
        ax[1].plot(x, zPeak_dscb(x, *p_tot[[ 'normSig', 'fraction_dscb', 'mean', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR', 'sigmaG']]), color='red', linewidth=2)

        ax[1].plot(x, zPeak_dscb(x, *p_tot[[ 'normSig', 'fraction_dscb', 'mean', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR', 'sigmaG']]) +
                   1*zPeak_dscb(x, *p_tot[[ 'normSig_H', 'fraction_dscb_H', 'mean_H', 'sigma_H', 'alphaL_H', 'nL_H', 'alphaR_H', 'nR_H', 'sigmaG_H']]), color='blue', linewidth=2)

    elif fit_parameters_zPeak["fitFunction"]=='zPeak_rscb':
        ax[1].plot(x, zPeak_rscb(x, *p_tot[[ 'normSig', 'fraction_dscb', 'mean', 'sigma', 'alphaR', 'nR', 'sigmaG']]), color='red', linewidth=2)
    ax[1].set_ylabel("Data - Background")
    ax[1].fill_between(x, ax[1].get_ylim()[0], ax[1].get_ylim()[1], where=maskFit, color='green', alpha=0.2)

    ax[1].hist(bins[:-1], bins=bins, weights = cZ)[0]
    ax[1].hist(bins[:-1], bins=bins, weights = cHiggs*1, bottom=cZ)[0]
    ax[1].set_ylim(-max(np.sqrt(c))*3, max(cZ)*2)



    ax[2].errorbar(x[maskUnblind], (c-myBkgFunction(x, *p_tot[myBkgParams]) - zPeak_dscb(x, *p_tot[[ 'normSig', 'fraction_dscb', 'mean', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR', 'sigmaG']]))[maskUnblind], yerr=np.sqrt(c)[maskUnblind], fmt='o', color='black', markersize=3)
    ax[2].set_ylim(-max(np.sqrt(c))*3, max(np.sqrt(c))*3)
#    else:
#        ax[1].errorbar(x, (c-myBkgFunction(x, *p_tot[myBkgParams])), yerr=np.sqrt(c), fmt='o', color='black', markersize=3)
    ax[2].set_ylim(ax[2].get_ylim())
    if fit_parameters_zPeak["fitFunction"]=='zPeak_dscb':
        #print(fit_parameters_zPeak['parameters']['nominal']['alphaL']['value'])
        #ax[1].plot(x, zPeak_dscb(x, *p_tot[[ 'normSig', 'fraction_dscb', 'mean', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR', 'sigmaG']]), color='red', linewidth=2)

        ax[2].plot(x, zPeak_dscb(x, *p_tot[[ 'normSig_H', 'fraction_dscb_H', 'mean_H', 'sigma_H', 'alphaL_H', 'nL_H', 'alphaR_H', 'nR_H', 'sigmaG_H']]), color='blue', linewidth=2)

    #elif fit_parameters_zPeak["fitFunction"]=='zPeak_rscb':
    #    ax[2].plot(x, zPeak_rscb(x, *p_tot[[ 'normSig', 'fraction_dscb', 'mean', 'sigma', 'alphaR', 'nR', 'sigmaG']]), color='red', linewidth=2)
    ax[2].set_ylabel("Data - Background")
    ax[2].fill_between(x, ax[1].get_ylim()[0], ax[1].get_ylim()[1], where=maskFit, color='green', alpha=0.2)

    #ax[2].hist(bins[:-1], bins=bins, weights = cZ)[0]
    ax[2].hist(bins[:-1], bins=bins, weights = cHiggs)[0]

    hep.cms.label(lumi="%.2f" %lumi_tot, ax=ax[0])
    ax[2].set_xlabel("Dijet Mass [GeV]")
    ax[0].set_ylabel("Counts")
    ax[0].legend(bbox_to_anchor=(1, 1))

    chi2_stat = np.sum(((c[maskFit] - myBkgFunctionZ_H(x, *p_tot)[maskFit])**2) / np.sqrt(c)[maskFit]**2)
    ndof = m_tot.ndof
    chi2_pvalue = 1- chi2.cdf(chi2_stat, ndof)
    ax[0].text(x=0.05, y=0.75, s="$\chi^2$/ndof = %.1f/%d\np-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax[0].transAxes, ha='left')
    #ax[0].set_yscale('log')

    fig.savefig(plotFolder+"/bkgPlusZPeak_fixed.png", bbox_inches='tight')

    plt.close()
    return x_fit, y_tofit, yerr, m_tot, cZ, cZ_err
