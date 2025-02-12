import numpy as np
import matplotlib.pyplot as plt
from hist import Hist
import mplhep as hep
import sys
from scipy.stats import shapiro, kstest, norm, chi2
hep.style.use("CMS")
from scipy.stats import anderson
def plot4ABCD(regions, bins, x1, x2, t1, t2, blindPar, outName=None,suffix="",  sameWidth_flag=False):
    sameWidth=np.ones(len(bins)-1) if sameWidth_flag == True else np.diff(bins)
    # A D C are not exaclty Poisson anymore after the subctraction of MC
    epsilon = 1e-6
    print("QCD in SR estimated via ABCD")
    hB_ADC_values = regions['A'].values()*regions['D'].values()/(regions['C'].values()+epsilon)
    ADC_err = regions['A'].values()*regions['D'].values()/regions['C'].values()*np.sqrt(regions['A'].variances()/(regions['A'].values())**2 +
                                                                                        regions['D'].variances()/(regions['D'].values())**2 +
                                                                                        regions['C'].variances()/(regions['C'].values())**2)
    hB_ADC = Hist.new.Var(bins, name='mjj').Weight()
    hB_ADC.values()[:] = hB_ADC_values
    hB_ADC.variances()[:] = ADC_err**2 
    print("QCD in SR estimated via ABCD. First Bin %.1f pm %.1f"%(hB_ADC.values()[0], np.sqrt(hB_ADC.variances()[0])))

    # Define the Higgs peak and blinding range
    # Create a mask for the blinding
    blind, higgs_peak, blind_range = blindPar
    blind_mask = (~((bins[:-1] > higgs_peak - blind_range) & (bins[:-1] < higgs_peak + blind_range))) if blind else np.ones(len(bins)-1, dtype=bool)


    # Plot Data and Data and B=A*D/C estimation in SR
    fig, ax = plt.subplots(2, 2, constrained_layout=True, figsize=(15, 10))
    x=(bins[1:]+bins[:-1])/2
    ax[0,0].hist(bins[:-1], bins=bins, weights=regions["A"].values()*blind_mask/sameWidth, label='Region A')
    ax[0,1].hist(bins[:-1], bins=bins, weights=regions["B"].values()*blind_mask/sameWidth, label='Region B')
    ax[1,0].hist(bins[:-1], bins=bins, weights=regions['C'].values()*blind_mask/sameWidth, label='Region C')
    ax[1,1].hist(bins[:-1], bins=bins, weights=regions['D'].values()*blind_mask/sameWidth, label='Region D')


    ax[0,1].hist(bins[:-1], bins=bins, weights=hB_ADC.values()*blind_mask/sameWidth, histtype=u'step', linewidth=2, label=r'$A\times D / C$ ', color='red')
    ax[0,1].errorbar(x, regions["B"].values()*blind_mask/sameWidth, yerr=np.sqrt(regions["B"].variances())/sameWidth, linestyle='none', color='black', marker='o')
    
    ax[0,0].set_title("%s < %.2f, %s >= %.2f"%(x1, t1, x2, t2), fontsize=14)
    ax[0,1].set_title("%s >= %.2f, %s >= %.2f"%(x1, t1, x2, t2), fontsize=14)
    ax[1,0].set_title("%s < %.2f, %s < %.2f"%(x1, t1, x2, t2), fontsize=14)
    ax[1,1].set_title("%s >= %.2f, %s < %.2f"%(x1, t1, x2, t2), fontsize=14)
    for idx, axx in enumerate(ax.ravel()):
        axx.set_xlim(bins[0], bins[-1])
        axx.set_xlabel("Dijet Mass [GeV]")
        axx.legend(fontsize=18, loc='upper right')
    if outName is None:
        outName = "/t3home/gcelotto/ggHbb/abcd/new/plots/QCDhists_SR_CR/QCDhists_SR_CR_%s.png"%suffix
    fig.savefig(outName, bbox_inches='tight')
    print("Saved %s"%outName)
    plt.close('all')    

    return hB_ADC



def SM_SR(regions, hB_ADC, bins, dfsData, dfsMC, isMCList, dfProcesses, x1, t1, x2, t2, lumi, suffix, blindPar, sameWidth_flag=True):
    sameWidth=np.ones(len(bins)-1) if sameWidth_flag == True else np.diff(bins)

    x = (bins[1:] + bins[:-1])/2
    blind, higgs_peak, blind_range = blindPar
    blind_mask = (~((bins[:-1] > higgs_peak - blind_range) & (bins[:-1] < higgs_peak + blind_range))) if blind else np.ones(len(bins)-1, dtype=bool)

    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
    fig.align_ylabels([ax[0],ax[1]])
    hExcess = regions["B"].copy()
    hExcess.values()[:] = regions["B"].values() - hB_ADC.values()
    # hExcess has non trivial variances since it is results of subtractions
    # Most of the uncertainty comes from this part: hB_ADC.variances()
    hExcess.variances()[:] = regions["B"].variances() + hB_ADC.variances()
    ax[0].errorbar(x, hExcess.values()*blind_mask/sameWidth, yerr=np.sqrt(hExcess.variances())*blind_mask/sameWidth , marker='o', color='black', linestyle='none')
    cTot = np.zeros(len(bins)-1)


    h = Hist.new.Var(bins, name="mjj").Weight()
    countsDict_SR = {
            'Data'   : h.copy() ,
            'VV'     : h.copy() ,
            'ST'     : h.copy() ,
            'ttbar'  : h.copy() ,
            'WJets' : h.copy() ,
            'QCD'    : h.copy() ,
            'ZJets' : h.copy() ,
            'H'      : h.copy() ,
        }

    for idx, df in enumerate(dfsMC):
        isMC = isMCList[idx]
        process = dfProcesses.process[isMC]
        mB      = (df[x1]>t1 ) & (df[x2]>t2 ) 

        h = Hist.new.Var(bins, name="mjj").Weight()
        h.fill(df.dijet_mass[mB], weight=df.weight[mB])

        if 'Data' in process:
            assert False
            countsDict_SR['Data'] = countsDict_SR['Data'] + h
            print("adding data with", process)
        elif 'HToBB' in process:
            countsDict_SR['H'] = countsDict_SR['H'] + h
        elif 'ST' in process:
            countsDict_SR['ST'] = countsDict_SR['ST'] + h
        elif 'TTTo' in process:
            countsDict_SR['ttbar'] = countsDict_SR['ttbar'] + h
        elif 'QCD' in process:
            countsDict_SR['QCD'] = countsDict_SR['QCD'] + h
        elif 'ZJets' in process:
            #print(process, " in ZJets")
            countsDict_SR['ZJets'] = countsDict_SR['ZJets'] + h
        elif 'WJets' in process:
            #print(process, " in WJets")
            countsDict_SR['WJets'] = countsDict_SR['WJets'] + h
        elif (('WW' in process) | ('ZZ' in process) | ('WZ' in process)):
            countsDict_SR['VV'] = countsDict_SR['VV'] + h
        else:
            #countsDict_SR['H'] = countsDict_SR['H'] + h
            assert False, "Process not found : %s"%process

    print("\n\n", "*"*50, "\n\n")
    print("SM counts in SR")
    for key in countsDict_SR.keys():
        if countsDict_SR[key].values().sum()==0:
            continue
        print("%s      \t : %d \t %.3f"%(key, countsDict_SR[key].values().sum(), np.sqrt(countsDict_SR[key].variances().sum())/countsDict_SR[key].values().sum()))
        ax[0].hist(bins[:-1], bins=bins, weights=countsDict_SR[key].values()/sameWidth, bottom=cTot/sameWidth, label=key)
        cTot = cTot + countsDict_SR[key].values()
    ax[0].legend()
    ax[0].set_xlim(bins[0],bins[-1])

    ax[1].set_xlim(ax[1].get_xlim())    
    ax[1].hlines(y=1, xmin=ax[1].get_xlim()[0], xmax=ax[1].get_xlim()[1], color='black')

    # Define a new histogram for SM contributions
    countsDict_SR['mc'] = countsDict_SR['ZJets'] + countsDict_SR['WJets'] + countsDict_SR['ttbar'] + countsDict_SR['ST'] + countsDict_SR['H'] + countsDict_SR['VV']
    # Arguments of bar are x, height -> 2xError, bottom
    ax[0].bar(x, 2*np.sqrt(countsDict_SR['mc'].variances())/sameWidth, width=sameWidth, bottom=(countsDict_SR['mc'].values() - np.sqrt(countsDict_SR['mc'].variances()))/sameWidth, 
           color='none', edgecolor='black', hatch='///', linewidth=0, alpha=1, label="Uncertainty")

    ax[1].bar(x, 2*np.sqrt(countsDict_SR['mc'].variances())/countsDict_SR['mc'].values()/sameWidth, width=sameWidth, bottom=1 - (np.sqrt(countsDict_SR['mc'].variances())/countsDict_SR['mc'].values())/sameWidth, 
           color='none', edgecolor='black', hatch='///', linewidth=0, alpha=1, label="Uncertainty")
    
    expected = countsDict_SR['mc'].values()[blind_mask]
    observed =  hExcess.values()[blind_mask]
    chi2_stat = np.sum((observed - expected) ** 2 / (countsDict_SR['mc'].variances()[blind_mask] + hExcess.variances()[blind_mask]))
    ndof = np.sum(blind_mask)
    p_value = 1 - chi2.cdf(chi2_stat, df=ndof)
    ax[0].text(x=0.95, y=0.5, s="$\chi^2$/ndof = %.1f/%d"%(chi2_stat, ndof), ha='right', transform=ax[0].transAxes)
    ax[0].text(x=0.95, y=0.42, s="p-value = %.3f"%p_value, ha='right', transform=ax[0].transAxes)


    ax[1].set_ylim(0., 2)
    ax[1].set_xlabel("Dijet Mass [GeV]")
    ax[1].set_ylabel("Ratio")
    ylabel = "Counts" if sameWidth_flag else  "Counts / Bin Width"
    ax[0].set_ylabel(ylabel)
    ax[1].errorbar(x, hExcess.values()/countsDict_SR['mc'].values()*blind_mask, yerr=np.sqrt(hExcess.variances()*blind_mask)/countsDict_SR['mc'].values() , marker='o', color='black', linestyle='none')
    hep.cms.label(lumi=np.round(lumi, 3), ax=ax[0])
    outName = "/t3home/gcelotto/ggHbb/abcd/new/plots/SMnonQCD/SMnonQCD_closure_%s.png"%suffix
    fig.savefig(outName)
    plt.close('all')
    print("\n\n", "*"*50, "\n\n")
    return countsDict_SR




def QCD_SR(bins, hB_ADC, qcd_mc, lumi=0, blindPar=(False, 125, 20),  outName=None,suffix="", sameWidth_flag=True, corrected=False):
    sameWidth=np.ones(len(bins)-1) if sameWidth_flag == True else np.diff(bins)
    x = (bins[1:] + bins[:-1])/2
    blind, higgs_peak, blind_range = blindPar
    blind_mask = (~((bins[:-1] > higgs_peak - blind_range) & (bins[:-1] < higgs_peak + blind_range))) if blind else np.ones(len(bins)-1, dtype=bool)

    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
    fig.align_ylabels([ax[0],ax[1]])
    ax[0].hist(bins[:-1], bins=bins, weights=qcd_mc.values()*blind_mask/sameWidth, label='QCD = Data - MC')
    ax[0].hist(bins[:-1], bins=bins, weights=(hB_ADC.values())*blind_mask/sameWidth, label='QCD = ABCD estimation', histtype='step', color='red', linewidth=2)

    observed = np.array(hB_ADC.values())[blind_mask]
    expected = np.array(qcd_mc.values())[blind_mask]
    sigma = np.sqrt(hB_ADC.variances()[blind_mask] + qcd_mc.variances()[blind_mask])  
    ndof = np.sum(blind_mask)

    chi2_stat = np.sum(((observed-expected)/sigma)**2)
    # Compute pulls
    pulls = (observed - expected) / sigma
    pvalues = np.where(
    pulls > 0, 
    1 - norm.cdf(pulls),  # For pulls > 0
    norm.cdf(pulls)       # For pulls < 0
    )

    # Print results
    print("Gaussian pulls in SR between QCD via ABCD and Data - MC")
    for pull, pval in zip(pulls, pvalues):
        print(f"Gaussian Pull: {pull:.2f}, p-value: {pval:.4f}")
    
    #shapiro_stat, shapiro_p = shapiro(pulls)
    #print(f"Shapiro-Wilk Test: Statistic = {shapiro_stat:.4f}, p-value = {shapiro_p:.4f}")
    KSstat, p_value_ks = kstest(pulls, 'norm', args=(0, 1))
    print(f"KS Test: Statistic = {KSstat:.4f}, p-value = {p_value_ks:.4f}")


    # Anderson-Darling Test
    
    result = anderson(pulls, dist='norm')
    print(f"Anderson-Darling Test Statistic: {result.statistic:.4f}")
    print("Critical Values:")
    for i, cv in enumerate(result.critical_values):
        print(f"  {result.significance_level[i]}%: {cv:.4f}")
    if result.statistic > result.critical_values[2]:  # Compare with 5% critical value
        print("The data does not follow a normal distribution at 5% significance level.")
    else:
        print("The data follows a normal distribution at 5% significance level.")

    chi2_pvalue = 1- chi2.cdf(chi2_stat, ndof)
    
    ax[0].text(x=0.95, y=0.9, s="$\chi^2$/ndof = %.1f/%d, p-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax[0].transAxes, ha='right')
    ax[0].text(x=0.95, y=0.82, s="KS p-value = %.3f"%(p_value_ks), transform=ax[0].transAxes, ha='right')
    print(qcd_mc.values())
    ratios = hB_ADC.values()*blind_mask/qcd_mc.values()
    err_ratios = np.sqrt(hB_ADC.variances())/qcd_mc.values()
    ax[1].errorbar(x, ratios, yerr=err_ratios,linestyle='none', marker='o', color='red')
    ylims = (0.95, 1.05) if corrected is not False else (0.9, 1.1)
    ax[1].set_ylim(ylims)
    ax[1].set_xlim(bins[0], bins[-1])
    ax[1].hlines(y=1, xmin=bins[0], xmax=bins[-1], color='C0')
    ax[1].set_xlabel("Dijet Mass [GeV]")
    ax[1].set_ylabel("Ratio")
    ylabel = "Counts" if sameWidth_flag == True else "Counts / Bin Width"
    ax[0].set_ylabel(ylabel) 
    ax[0].legend()
    hep.cms.label(lumi=np.round(lumi, 3), ax=ax[0])
    if outName is None:
        assert False, "outName not None"
    fig.savefig(outName, bbox_inches='tight')

    return ratios, err_ratios


def QCDplusSM_SR(bins, regions, countsDict, hB_ADC, lumi, suffix, blindPar, sameWidth_flag=True, corrections=None):
    sameWidth=np.ones(len(bins)-1) if sameWidth_flag == True else np.diff(bins)
    x = (bins[1:] + bins[:-1])/2
    blind, higgs_peak, blind_range = blindPar
    blind_mask = (~((bins[:-1] > higgs_peak - blind_range) & (bins[:-1] < higgs_peak + blind_range))) if blind else np.ones(len(bins)-1, dtype=bool)
    
    labels = ['Data', 'H', 'VV', 'ST', 'ttbar', 'WJets', 'QCD', 'ZJets']

    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
    fig.align_ylabels([ax[0],ax[1]])
    
    # Data in SR
    ax[0].errorbar(x, regions['B'].values()*blind_mask/sameWidth, yerr=np.sqrt(regions['B'].variances()*blind_mask)/sameWidth , marker='o', color='black', linestyle='none')

    # Prediction QCD in SR via ABCD
    cTot = np.zeros(len(bins)-1)
    cQCD = ax[0].hist(bins[:-1], bins=bins, weights=hB_ADC.values()*blind_mask/sameWidth, bottom=cTot, label='QCD')[0]
    cTot = cTot + cQCD
    # Prediction MC in SR
    for key in labels:
        if np.sum(countsDict[key].values())==0:
            continue
        print(key, countsDict[key].sum())
        ax[0].hist(bins[:-1], bins=bins, weights=countsDict[key].values()*blind_mask/sameWidth, bottom=cTot, label=key)
        cTot = cTot + countsDict[key].values()/sameWidth


    ax[0].legend()
    ax[0].set_yscale('log')
    ylabel = "Counts" if sameWidth_flag else "Counts / Bin Width"
    ax[0].set_ylabel(ylabel)
    
    ax[1].set_xlim(ax[1].get_xlim())    
    ax[1].hlines(y=0, xmin=ax[1].get_xlim()[0], xmax=ax[1].get_xlim()[1], color='black')
    mcPlusQCD = countsDict["mc"].copy()
    print("mcPlusQCD")
    print(mcPlusQCD.values())
    mcPlusQCD= mcPlusQCD + hB_ADC
    ylims = (-3, 3) if corrections is not None else (-5, 5)
    ax[1].set_ylim(ylims)
    ax[1].set_yticks(ticks=[-3, -1, 1, 3])

    pulls = (regions["B"].values() - mcPlusQCD.values())[blind_mask] / (np.sqrt(regions["B"].variances() + mcPlusQCD.variances())[blind_mask])
    chi2_stat = np.sum(pulls**2)
    ndof = np.sum(blind_mask)
    p_value = 1 - chi2.cdf(chi2_stat, df=ndof)
    ax[0].text(x=0.95, y=0.9, s="$\chi^2$/ndof = %.1f/%d"%(chi2_stat, ndof), ha='right', transform=ax[0].transAxes)
    ax[0].text(x=0.95, y=0.82, s="p-value = %.3f"%p_value, ha='right', transform=ax[0].transAxes)
    #err_pulls = regions["B"].values()*blind_mask/mcPlusQCD.values() * np.sqrt(   ()/regions["B"].values())**2 +   (np.sqrt()/mcPlusQCD.values())**2)

    ax[1].errorbar(x[blind_mask], pulls, yerr=1 , marker='o', color='black', linestyle='none')
    print("*"*10)
    print("Ratios in SR of QCD + MC")
    for val in pulls:
        print("%.4f"%(val))
    print("*"*10)
    ax[1].set_xlabel("Dijet Mass [GeV]")
    ax[1].set_ylabel("Pulls")
    hep.cms.label(lumi=np.round(lumi, 3), ax=ax[0])
    outName = "/t3home/gcelotto/ggHbb/abcd/new/plots/ZQCDplusSM/ZQCDplusSM_%s.png"%suffix
    fig.savefig(outName)
    print(outName)

    return pulls


def pullsVsDisco(dcor_values, pulls_QCDPlusSM_SR, err_QCDPlusSM_SR, lumi=0, outName=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
    fig.align_ylabels([ax[0],ax[1]])
    ax[0].errorbar(dcor_values, abs(pulls_QCDPlusSM_SR-1), err_QCDPlusSM_SR, linestyle='none', color='black', marker='o')
    ax[1].set_xlabel("disCo")
    ax[0].set_ylabel("|Ratio - 1|")
    ax[0].set_xlim(0, ax[0].get_xlim()[1])


    def model_function(x, a, b):
        """Example model function: a simple linear model."""
        return a * np.array(x) + b
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(model_function, dcor_values, abs(pulls_QCDPlusSM_SR-1), sigma=err_QCDPlusSM_SR, absolute_sigma=True)

    # Extract fit parameters and their uncertainties
    a_fit, b_fit = popt
    a_err, b_err = np.sqrt(np.diag(pcov))
    y_fit = model_function(dcor_values, *popt)
    ax[0].plot(dcor_values, y_fit, label=f'Fit: y = {a_fit:.2f}x + {b_fit:.2f}', color='red')
    ax[1].errorbar(dcor_values, abs(pulls_QCDPlusSM_SR-1)-y_fit, err_QCDPlusSM_SR, linestyle='none', color='black', marker='o')
    ax[1].hlines(y=0, xmin=0, xmax=ax[1].get_xlim()[1], color='red')

    chi2_stat = np.sum(((abs(pulls_QCDPlusSM_SR-1)-y_fit)/err_QCDPlusSM_SR)**2)
    ndof  = len(err_QCDPlusSM_SR)
    from scipy.stats import chi2
    chi2_pvalue = 1- chi2.cdf(chi2_stat, ndof)
    ax[0].text(x=0.95, y=0.12, s="$\chi^2$/ndof = %.1f/%d, p-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax[0].transAxes, ha='right')

    if outName is not None:
        fig.savefig(outName, bbox_inches='tight')

    return popt, pcov


def pullsVsDisco_pearson(dcor_values, pulls_QCDPlusSM_SR, err_QCDPlusSM_SR, mask, lumi=0, outName=None):
    pulls_QCDPlusSM_SR = pulls_QCDPlusSM_SR[mask]
    err_QCDPlusSM_SR = err_QCDPlusSM_SR[mask]
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
    fig.align_ylabels([ax[0],ax[1]])
    ax[0].errorbar(dcor_values[mask], pulls_QCDPlusSM_SR, err_QCDPlusSM_SR, linestyle='none', color='black', marker='o', label='CR')
    ax[1].set_xlabel("Pearson R")
    ax[0].set_ylabel("1/Correction")
    


    def model_function(x, q, m):
        """Example model function: a simple linear model."""
        return q + m * np.array(x)
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(model_function, dcor_values[mask], pulls_QCDPlusSM_SR, sigma=err_QCDPlusSM_SR, absolute_sigma=True)

    # Extract fit parameters and their uncertainties
    q_fit, m_fit = popt
    a_err, b_err = np.sqrt(np.diag(pcov))
    y_fit = model_function(dcor_values, *popt)
    ax[0].plot(dcor_values, y_fit, label=f'Fit: y = {m_fit:.2f}x + {q_fit:.2f}', color='red')
    ax[0].errorbar(dcor_values[~mask], model_function(dcor_values[~mask], *popt), label=f'VR and SR', color='green', linestyle='none', marker='o')
    ax[0].set_xlim(ax[1].get_xlim()[0], ax[0].get_xlim()[1])
    ax[1].errorbar(dcor_values[mask], pulls_QCDPlusSM_SR-y_fit[mask], err_QCDPlusSM_SR, linestyle='none', color='black', marker='o')
    ax[1].hlines(y=0, xmin=ax[1].get_xlim()[0], xmax=ax[1].get_xlim()[1], color='red')

    chi2_stat = np.sum(((pulls_QCDPlusSM_SR-y_fit[mask])/err_QCDPlusSM_SR)**2)
    ndof  = np.sum(mask)
    from scipy.stats import chi2
    chi2_pvalue = 1- chi2.cdf(chi2_stat, ndof)
    ax[0].text(x=0.05, y=0.12, s="$\chi^2$/ndof = %.1f/%d, p-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax[0].transAxes, ha='left')
    ax[0].text(x=0.05, y=0.04, s="q = %.3f +- %.3f"%(popt[0], np.sqrt(pcov[0,0])) , transform=ax[0].transAxes, ha='left')
    #ax[0].text(x=0.05, y=0.10, s="m = %.3f +- %.3f"%(popt[0], np.sqrt(pcov[0,0])) , transform=ax[0].transAxes, ha='left')
    ax[0].legend()

    if outName is not None:
        fig.savefig(outName, bbox_inches='tight')

    return popt, pcov