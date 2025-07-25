import numpy as np
import matplotlib.pyplot as plt
from hist import Hist
import mplhep as hep
import sys
from scipy.stats import shapiro, kstest, norm, chi2
hep.style.use("CMS")
from scipy.stats import anderson


def plot4ABCD(regions, bins, x1, x2, t1, t2, unblinded_mask, outName=None,suffix="",  sameWidth_flag=False):
    sameWidth=np.ones(len(bins)-1) if sameWidth_flag == True else np.diff(bins)
    # A D C are not exaclty Poisson anymore after the subctraction of MC
    epsilon = 1e-6
    #print("QCD in SR estimated via ABCD")
    hB_ADC_values = regions['A'].values()*regions['D'].values()/(regions['C'].values()+epsilon)
    ADC_err = regions['A'].values()*regions['D'].values()/regions['C'].values()*np.sqrt(regions['A'].variances()/(regions['A'].values())**2 +
                                                                                        regions['D'].variances()/(regions['D'].values())**2 +
                                                                                        regions['C'].variances()/(regions['C'].values())**2)
    hB_ADC = Hist.new.Var(bins, name='mjj').Weight()
    hB_ADC.values()[:] = hB_ADC_values
    hB_ADC.variances()[:] = ADC_err**2 


    #print("QCD in SR estimated via ABCD")


    # Plot Data and Data and B=A*D/C estimation in SR
    fig, ax = plt.subplots(2, 2, constrained_layout=True, figsize=(15, 10))
    x=(bins[1:]+bins[:-1])/2
    ax[0,0].hist(bins[:-1], bins=bins, weights=regions["A"].values()/sameWidth, label='Region A')
    ax[0,1].hist(bins[:-1], bins=bins, weights=regions["B"].values()*unblinded_mask/sameWidth, label='Region B')
    ax[1,0].hist(bins[:-1], bins=bins, weights=regions['C'].values()/sameWidth, label='Region C')
    ax[1,1].hist(bins[:-1], bins=bins, weights=regions['D'].values()/sameWidth, label='Region D')


    ax[0,1].hist(bins[:-1], bins=bins, weights=hB_ADC.values()/sameWidth, histtype=u'step', linewidth=2, label=r'$A\times D / C$ ', color='red')
    ax[0,1].errorbar(x, regions["B"].values()*unblinded_mask/sameWidth, yerr=np.sqrt(regions["B"].variances())/sameWidth, linestyle='none', color='black', marker='o')
    
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
    ax[0,0].set_yscale('log')
    ax[0,1].set_yscale('log')
    ax[1,0].set_yscale('log')
    ax[1,1].set_yscale('log')
    fig.savefig(outName, bbox_inches='tight')
    print("Saved %s"%outName)
    plt.close('all')    

    return hB_ADC









def SM_CR(region, bins, dfMC, isMCList, dfProcesses, x1, t1, x2, t2, lumi, suffix, unblinded_mask, sameWidth_flag=True):
    sameWidth=np.ones(len(bins)-1) if sameWidth_flag == True else np.diff(bins)
    x = (bins[1:] + bins[:-1])/2


    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
    fig.align_ylabels([ax[0],ax[1]])
    cTot = np.zeros(len(bins)-1)


    h = Hist.new.Var(bins, name="mjj").Weight()
    countsDict_CR = {
            'Data'   : h.copy() ,
            'VV'     : h.copy() ,
            'ST'     : h.copy() ,
            'ttbar'  : h.copy() ,
            'WJets' : h.copy() ,
            'QCD'    : h.copy() ,
            'ZJets' : h.copy() ,
            'H'      : h.copy() ,
        }

    for process in np.unique(dfMC.process):
        maskProcess = dfMC.process == process
        df = dfMC[maskProcess]

        if region == 'A':
            mRegion      = (df[x1]<t1 ) & (df[x2]>t2 ) 
        elif region == 'B':
            mRegion      = (df[x1]>t1 ) & (df[x2]>t2 ) 
            
        elif region == 'C':
            mRegion      = (df[x1]<t1 ) & (df[x2]<t2 ) 
        elif region == 'D':
            mRegion      = (df[x1]>t1 ) & (df[x2]<t2 ) 

        h = Hist.new.Var(bins, name="mjj").Weight()
        h.fill(df.dijet_mass[mRegion], weight=df.weight_[mRegion])

        if 'Data' in process:
            assert False
            countsDict_CR['Data'] = countsDict_CR['Data'] + h
            print("adding data with", process)
        elif 'HToBB' in process:
            countsDict_CR['H'] = countsDict_CR['H'] + h
        elif 'ST' in process:
            countsDict_CR['ST'] = countsDict_CR['ST'] + h
        elif 'TTTo' in process:
            countsDict_CR['ttbar'] = countsDict_CR['ttbar'] + h
        elif 'QCD' in process:
            countsDict_CR['QCD'] = countsDict_CR['QCD'] + h
        elif 'ZJets' in process:
            #print(process, " in ZJets")
            countsDict_CR['ZJets'] = countsDict_CR['ZJets'] + h
        elif 'WJets' in process:
            #print(process, " in WJets")
            countsDict_CR['WJets'] = countsDict_CR['WJets'] + h
        elif (('WW' in process) | ('ZZ' in process) | ('WZ' in process)):
            countsDict_CR['VV'] = countsDict_CR['VV'] + h
        else:
            #countsDict_CR['H'] = countsDict_CR['H'] + h
            assert False, "Process not found : %s"%process

    print("\n\n", "*"*50, "\n\n")
    print("SM counts in CR")
    for key in countsDict_CR.keys():
        if countsDict_CR[key].values().sum()==0:
            continue
        print("%s      \t : %d \t %.3f"%(key, countsDict_CR[key].values().sum(), np.sqrt(countsDict_CR[key].variances().sum())/countsDict_CR[key].values().sum()))
        ax[0].hist(bins[:-1], bins=bins, weights=countsDict_CR[key].values()/sameWidth, bottom=cTot/sameWidth, label=key)
        cTot = cTot + countsDict_CR[key].values()
    ax[0].legend()
    ax[0].set_xlim(bins[0],bins[-1])
    ax[0].set_xlabel("Dijet Mass [GeV]")
    ylabel = "Counts" if sameWidth_flag else  "Counts / Bin Width"
    ax[0].set_ylabel(ylabel)
    hep.cms.label(lumi=np.round(lumi, 2), ax=ax[0])
    outName = "/t3home/gcelotto/ggHbb/abcd/new/plots/SMnonQCD_CR/SMnonQCD_CR%s_%s.png"%(region, suffix)
    fig.savefig(outName)
    plt.close('all')
    print("\n\n", "*"*50, "\n\n")
    return










def SM_SR(regions, hB_ADC, bins, dfData, dfMC, isMCList, dfProcesses, x1, t1, x2, t2, lumi, suffix, unblinded_mask, chi2_mask, sameWidth_flag=True):
    sameWidth=np.ones(len(bins)-1) if sameWidth_flag == True else np.diff(bins)

    x = (bins[1:] + bins[:-1])/2

    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
    fig.align_ylabels([ax[0],ax[1]])
    hExcess = regions["B"].copy()
    hExcess.values()[:] = regions["B"].values() - hB_ADC.values()
    # hExcess has non trivial variances since it is results of subtractions
    # Most of the uncertainty comes from this part: hB_ADC.variances()
    hExcess.variances()[:] = regions["B"].variances() + hB_ADC.variances()
    ax[0].errorbar(x, hExcess.values()*unblinded_mask/sameWidth, yerr=np.sqrt(hExcess.variances())*unblinded_mask/sameWidth , marker='o', color='black', linestyle='none')
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

    for process in np.unique(dfMC.process):
        maskProcess = dfMC.process==process
        # Select only event in the dataframe belonging to the same process
        df = dfMC[maskProcess]

        mB      = (df[x1]>t1 ) & (df[x2]>t2 ) 
        h = Hist.new.Var(bins, name="mjj").Weight()
        h.fill(df.dijet_mass[mB], weight=df.weight_[mB])

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
    ax[0].bar(x, 2*np.sqrt(countsDict_SR['mc'].variances())/sameWidth, width=np.diff(bins), bottom=(countsDict_SR['mc'].values() - np.sqrt(countsDict_SR['mc'].variances()))/sameWidth, 
           color='none', edgecolor='black', hatch='///', linewidth=0, alpha=1, label="Uncertainty")

    ax[1].bar(x, 2*np.sqrt(countsDict_SR['mc'].variances())/countsDict_SR['mc'].values()/sameWidth, width=np.diff(bins), bottom=1 - (np.sqrt(countsDict_SR['mc'].variances())/countsDict_SR['mc'].values())/sameWidth, 
           color='none', edgecolor='black', hatch='///', linewidth=0, alpha=1, label="Uncertainty")
    
    expected = countsDict_SR['mc'].values()[chi2_mask]
    observed =  hExcess.values()[chi2_mask]
    chi2_stat = np.sum((observed - expected) ** 2 / (countsDict_SR['mc'].variances()[chi2_mask] + hExcess.variances()[chi2_mask]))
    ndof = np.sum(chi2_mask)
    p_value = 1 - chi2.cdf(chi2_stat, df=ndof)
    ax[0].text(x=0.95, y=0.5, s="$\chi^2$/ndof = %.1f/%d"%(chi2_stat, ndof), ha='right', transform=ax[0].transAxes)
    ax[0].text(x=0.95, y=0.42, s="p-value = %.3f"%p_value, ha='right', transform=ax[0].transAxes)


    ax[1].set_ylim(0., 2)
    ax[1].set_xlabel("Dijet Mass [GeV]")
    ax[1].set_ylabel("Ratio")
    ylabel = "Counts" if sameWidth_flag else  "Counts / Bin Width"
    ax[0].set_ylabel(ylabel)
    ax[1].errorbar(x, hExcess.values()/countsDict_SR['mc'].values()*unblinded_mask, yerr=np.sqrt(hExcess.variances()*unblinded_mask)/countsDict_SR['mc'].values() , marker='o', color='black', linestyle='none')
    hep.cms.label(lumi=np.round(lumi, 2), ax=ax[0])
    outName = "/t3home/gcelotto/ggHbb/abcd/new/plots/SMnonQCD/SMnonQCD_closure_%s.png"%suffix
    fig.savefig(outName)
    plt.close('all')
    print("\n\n", "*"*50, "\n\n")
    return countsDict_SR




def QCD_SR(bins, hB_ADC, qcd_mc, chi2_mask, unblinded_mask,lumi=0,   outName=None, suffix="", sameWidth_flag=True, corrected=False):
    sameWidth=np.ones(len(bins)-1) if sameWidth_flag == True else np.diff(bins)
    x = (bins[1:] + bins[:-1])/2

    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
    fig.align_ylabels([ax[0],ax[1]])
    ax[0].hist(bins[:-1], bins=bins, weights=qcd_mc.values()*unblinded_mask/sameWidth, label='QCD = Data - Resonant')
    ax[0].hist(bins[:-1], bins=bins, weights=(hB_ADC.values())/sameWidth, label='QCD = ABCD estimation', histtype='step', color='red', linewidth=2)

    observed = np.array(hB_ADC.values())[chi2_mask]
    expected = np.array(qcd_mc.values())[chi2_mask]
    sigma = np.sqrt(hB_ADC.variances()[chi2_mask] + qcd_mc.variances()[chi2_mask])  
    ndof = np.sum(chi2_mask)

    chi2_stat = np.sum(((observed-expected)/sigma)**2)
    # Compute pulls
    pulls = (observed - expected) / sigma
    pvalues = np.where( pulls > 0, 
                        1 - norm.cdf(pulls),  # For pulls > 0
                        norm.cdf(pulls)       # For pulls < 0
                        )

    # Print results
    print("Gaussian pulls in SR between QCD via ABCD and Data - MC")
    for pull, pval in zip(pulls, pvalues):
        print(f"Gaussian Pull: {pull:.2f}, p-value: {pval:.4f}")
    
    #shapiro_stat, shapiro_p = shapiro(pulls)
    #print(f"Shapiro-Wilk Test: Statistic = {shapiro_stat:.4f}, p-value = {shapiro_p:.4f}")
    #KSstat, p_value_ks = kstest(pulls, 'norm', args=(0, 1))
    #print(f"KS Test: Statistic = {KSstat:.4f}, p-value = {p_value_ks:.4f}")


    # Anderson-Darling Test
    
    #result = anderson(pulls, dist='norm')
    #print(f"Anderson-Darling Test Statistic: {result.statistic:.4f}")
    #print("Critical Values:")
    #for i, cv in enumerate(result.critical_values):
    #    print(f"  {result.significance_level[i]}%: {cv:.4f}")
    #if result.statistic > result.critical_values[2]:  # Compare with 5% critical value
    #    print("The data does not follow a normal distribution at 5% significance level.")
    #else:
    #    print("The data follows a normal distribution at 5% significance level.")

    chi2_pvalue = 1- chi2.cdf(chi2_stat, ndof)
    
    ax[0].text(x=0.95, y=0.8, s="$\chi^2$/ndof = %.1f/%d, p-value = %.3f"%(chi2_stat, ndof, chi2_pvalue), transform=ax[0].transAxes, ha='right')
    #ax[0].text(x=0.95, y=0.82, s="KS p-value = %.3f"%(p_value_ks), transform=ax[0].transAxes, ha='right')
    print(qcd_mc.values())
    ratios = hB_ADC.values()/qcd_mc.values()
    err_ratios = np.sqrt(hB_ADC.variances())/qcd_mc.values()
    ax[1].errorbar(x[unblinded_mask], ratios[unblinded_mask], yerr=err_ratios[unblinded_mask],linestyle='none', marker='o', color='red')
    #ylims = (0.995, 1.005) if corrected is not False else (0.9, 1.1)
    #ax[1].set_ylim(ylims)
    ax[1].set_xlim(bins[0], bins[-1])
    ax[1].hlines(y=1, xmin=bins[0], xmax=bins[-1], color='C0')
    ax[1].set_xlabel("Dijet Mass [GeV]")
    ax[1].set_ylabel("Ratio")
    ylabel = "Counts" if sameWidth_flag == True else "Counts / Bin Width"
    ax[0].set_ylabel(ylabel) 
    ax[0].legend()
    hep.cms.label(lumi=np.round(lumi, 2), ax=ax[0])
    if outName is None:
        assert False, "outName not None"
    fig.savefig(outName, bbox_inches='tight')

    return ratios, err_ratios


def QCDplusSM_SR(bins, regions, countsDict, hB_ADC, lumi, suffix, unblinded_mask, chi2_mask, sameWidth_flag=True, corrected=False):
    sameWidth=np.ones(len(bins)-1) if sameWidth_flag == True else np.diff(bins)
    x = (bins[1:] + bins[:-1])/2
    
    labels = ['Data', 'H', 'VV', 'ST', 'ttbar', 'WJets', 'QCD', 'ZJets']

    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
    fig.align_ylabels([ax[0],ax[1]])
    
    # Data in SR
    ax[0].errorbar(x, regions['B'].values()*unblinded_mask/sameWidth, yerr=np.sqrt(regions['B'].variances()*unblinded_mask)/sameWidth , marker='o', color='black', linestyle='none')

    # Prediction QCD in SR via ABCD
    cTot = np.zeros(len(bins)-1)
    cQCD = ax[0].hist(bins[:-1], bins=bins, weights=hB_ADC.values()*unblinded_mask/sameWidth, bottom=cTot, label='QCD')[0]
    cTot = cTot + cQCD
    # Prediction MC in SR
    for key in labels:
        if np.sum(countsDict[key].values())==0:
            continue
        print(key, countsDict[key].sum())
        ax[0].hist(bins[:-1], bins=bins, weights=countsDict[key].values()*unblinded_mask/sameWidth, bottom=cTot, label=key)
        cTot = cTot + countsDict[key].values()/sameWidth


    
    ax[0].set_yscale('log')
    ylabel = "Counts" if sameWidth_flag else "Counts / Bin Width"
    ax[0].set_ylabel(ylabel)
    
    ax[1].set_xlim(ax[1].get_xlim())    
    ax[1].hlines(y=1, xmin=ax[1].get_xlim()[0], xmax=ax[1].get_xlim()[1], color='black')
    mcPlusQCD = countsDict["mc"].copy()
    print("mcPlusQCD")
    print(mcPlusQCD.values())
    mcPlusQCD= mcPlusQCD + hB_ADC
    #ylims = (-3, 3) if corrections is not None else (-5, 5)
    #ax[1].set_ylim(ylims)
    #ax[1].set_yticks(ticks=[-3, -1, 1, 3])
    print(unblinded_mask)
    pulls = (regions["B"].values() - mcPlusQCD.values())[unblinded_mask] / (np.sqrt(regions["B"].variances() + mcPlusQCD.variances())[unblinded_mask])
    
    observed = np.array(regions["B"].values())[chi2_mask]
    expected = np.array(mcPlusQCD.values())[chi2_mask]
    sigma = np.sqrt(regions["B"].variances() + mcPlusQCD.variances())[chi2_mask]
    ndof = np.sum(chi2_mask)

    chi2_stat = np.sum(((observed-expected)/sigma)**2)
    p_value = 1 - chi2.cdf(chi2_stat, df=ndof)
    ax[0].text(x=0.95, y=0.9, s="$\chi^2$/ndof = %.1f/%d"%(chi2_stat, ndof), ha='right', transform=ax[0].transAxes)
    ax[0].text(x=0.95, y=0.82, s="p-value = %.3f"%p_value, ha='right', transform=ax[0].transAxes)
    ax[0].legend()

    err_ratio = regions["B"].values()*unblinded_mask/mcPlusQCD.values() * np.sqrt(  regions["B"].variances()/(regions["B"].values())**2 +  mcPlusQCD.variances()/mcPlusQCD.values()**2)
    ax[1].errorbar(x[unblinded_mask], (regions["B"].values()/mcPlusQCD.values())[unblinded_mask], yerr=err_ratio[unblinded_mask] , marker='o', color='black', linestyle='none')
    print("*"*10)
    print("Ratios in SR of QCD + MC")
    for val in pulls:
        print("%.4f"%(val))
    print("*"*10)
    ax[1].set_xlabel("Dijet Mass [GeV]")
    ax[1].set_ylabel("Ratio")
    hep.cms.label(lumi=np.round(lumi, 2), ax=ax[0])
    outName = "/t3home/gcelotto/ggHbb/abcd/new/plots/ZQCDplusSM/ZQCDplusSM_%s.png"%suffix
    #ax[0].set_ylim(10**3, ax[0].get_ylim()[1])
    fig.savefig(outName)
    print(outName)

    return pulls


def pullsVsDisco(dcor_values, pulls_QCDPlusSM_SR, err_QCDPlusSM_SR, lumi=0, outName=None):
    assert False
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


def pullsVsDisco_pearson(dcor_values, pulls_QCDPlusSM_SR, err_QCDPlusSM_SR, mask, xerr, mass_bin_center, lumi=0, outName=None):
    '''
    xerr is none -> then use a normal fit
    otherwise use a residual fit
    '''
    import matplotlib.pyplot as plt

    pulls_QCDPlusSM_SR_masked = pulls_QCDPlusSM_SR[mask]
    err_QCDPlusSM_SR_masked = err_QCDPlusSM_SR[mask]
    dcor_values_masked = dcor_values[mask]


    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10), constrained_layout=True)
    fig.align_ylabels([ax[0],ax[1]])


    if xerr is not None:
        # ODR Fit
        import scipy.odr as odr
        xerr = xerr[mask]

        def model_func(beta, x):
            return beta[0] * x + beta[1]

        linear_model = odr.Model(model_func)
        data = odr.RealData(dcor_values_masked, pulls_QCDPlusSM_SR_masked, sx=xerr, sy=err_QCDPlusSM_SR_masked)
        odr_fit = odr.ODR(data, linear_model, beta0=[1, 1])
        output = odr_fit.run()

        m_fit, q_fit = output.beta
        m_err, q_err = output.sd_beta
        cov_matrix = output.cov_beta * output.res_var
        # the covariance matrix is scaled with residuals
        # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.Output.html#scipy.odr.Output
        y_fit = model_func(output.beta, dcor_values)
        y_green = model_func((m_fit, q_fit), dcor_values[~mask])
        
        
        # Compute normalized orthogonal distance
        numerator = (pulls_QCDPlusSM_SR_masked - (m_fit * dcor_values_masked + q_fit))
        denominator = np.sqrt(m_fit**2 + 1)
        orthogonal_distance = numerator / denominator
        total_error = np.sqrt((xerr**2 * m_fit**2 + err_QCDPlusSM_SR_masked**2) / (m_fit**2 + 1))
        normalized_distance = orthogonal_distance / total_error

        ax[1].errorbar(dcor_values_masked, normalized_distance, xerr=0, yerr=1, linestyle='none', color='black', marker='o')
        ax[1].set_ylabel("Diagonal Distance")
        ax[1].yaxis.set_label_coords(-0.15, 1.2)
    else:
        from scipy.optimize import curve_fit
        # No x-error: use standard curve_fit
        xerr = np.zeros_like(dcor_values_masked)

        def model_func(x, a, b):
            return a * x + b

        popt, pcov = curve_fit(model_func, dcor_values_masked, pulls_QCDPlusSM_SR_masked, sigma=err_QCDPlusSM_SR_masked, absolute_sigma=True)
        m_fit, q_fit = popt
        m_err, q_err = np.sqrt(np.diag(pcov))
        cov_matrix = pcov
        y_fit = model_func(dcor_values, *popt)
        y_green = pulls_QCDPlusSM_SR[~mask]


        # Plot residuals
        deltaY = (pulls_QCDPlusSM_SR_masked - (m_fit * dcor_values_masked + q_fit))

        ax[1].errorbar(dcor_values_masked, deltaY/err_QCDPlusSM_SR_masked, xerr=0, yerr=1, linestyle='none', color='black', marker='o')
        ax[1].errorbar(dcor_values[~mask], (y_green-(m_fit * dcor_values[~mask] + q_fit))/err_QCDPlusSM_SR[~mask], xerr=0, yerr=1, linestyle='none', color='green', marker='o')
        ax[1].set_ylabel("Pulls")
    
    # Plotting
    ax[0].errorbar(dcor_values_masked, pulls_QCDPlusSM_SR_masked, xerr=xerr, yerr=err_QCDPlusSM_SR_masked,
                   linestyle='none', color='black', marker='o', label='CR')
    for idx, (x, y, m) in enumerate(zip(dcor_values_masked, pulls_QCDPlusSM_SR_masked, mass_bin_center[mask])):
        ax[0].text(x , y + (-1)**idx *  0.01, f'{m:.1f}', fontsize=8, ha='center')
    for idx, (x, y, m) in enumerate(zip(dcor_values[~mask], pulls_QCDPlusSM_SR[~mask], mass_bin_center[~mask])):
        ax[0].text(x , y + (-1)**idx *  0.01, f'{m:.1f}', fontsize=8, color='green', ha='center')
    ax[0].plot(dcor_values, y_fit, label=f'Fit: y = {m_fit:.2f}x + {q_fit:.2f}', color='red')
    ax[0].errorbar(dcor_values[~mask], y_green, label='VR and SR',
                   color='green', linestyle='none', marker='o')
    ax[0].set_ylabel("Ratio")
    ax[1].set_xlabel("Pearson R")
    ax[0].legend()

    
    ax[1].hlines(y=0, xmin=ax[1].get_xlim()[0], xmax=ax[1].get_xlim()[1], color='red')
    
    ax[0].yaxis.set_label_coords(-0.15, 1)
    

    # Chi2
    chi2_stat = np.sum((pulls_QCDPlusSM_SR_masked - (m_fit * dcor_values_masked + q_fit))**2 / (err_QCDPlusSM_SR_masked**2 + (m_fit * xerr)**2))
    ndof = len(pulls_QCDPlusSM_SR_masked) - 2
    chi2_pvalue = 1 - chi2.cdf(chi2_stat, ndof)

    ax[0].text(0.05, 0.12, f"$\\chi^2$/ndof = {chi2_stat:.1f}/{ndof}, p-value = {chi2_pvalue:.3f}", transform=ax[0].transAxes, ha='left')
    ax[0].text(0.05, 0.04, f"q = {q_fit:.3f} ± {q_err:.3f}", transform=ax[0].transAxes, ha='left')

    if outName is not None:
        fig.savefig(outName, bbox_inches='tight')

    return (m_fit, q_fit), (m_err, q_err), cov_matrix
    

