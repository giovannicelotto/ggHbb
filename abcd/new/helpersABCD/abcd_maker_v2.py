import sys
sys.path.append("/t3home/gcelotto/ggHbb/abcd/new/helpersABCD")
from plot_v2 import plot4ABCD, QCD_SR, QCDplusSM_SR, SM_SR, SM_CR
from createRootHists import createRootHists
from hist import Hist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def ABCD(       dfData: pd.DataFrame, dfMC: pd.DataFrame,
                x1: str,     x2: str,     xx: str, bins, t1: float,     t2: float,
                isMCList, dfProcessesMC, lumi: float, suffix: str, blindPar:tuple, chi2_mask,
                sameWidth_flag:bool,
                corrections=None, err_corrections=None, var=None, covs=None
                ):
    
    '''
    blindPar = parameters of blinding. Tuple (Bool, min x, max x)
    '''
    x=(bins[1:]+bins[:-1])/2
    unblinded_mask = (x < blindPar[1]) | (x > blindPar[2]) if blindPar[0]==True else np.full(len(bins)-1, True)
    for i, (start, end) in enumerate(zip(bins[:-1], bins[1:])):
        status = "unblinded" if unblinded_mask[i] else "blinded"
        print("%.1f - %.1f : %s"%(start, end, status))


    print("Creating histograms for SR and CR")
    hA = Hist.new.Var(bins, name="mjj").Weight()
    hB = Hist.new.Var(bins, name="mjj").Weight()
    hC = Hist.new.Var(bins, name="mjj").Weight()
    hD = Hist.new.Var(bins, name="mjj").Weight()
    #inclusive = Hist.new.Var(bins, name="mjj").Weight()
    regions = {
        'A' : hA,
        'B' : hB,
        'C' : hC,
        'D' : hD,
    }
    print("Empty Hist created")


    # Fill regions with data
    print("Filling regions with Data")

    mA      = (dfData[x1]<t1 ) & (dfData[x2]>t2 ) 
    mB      = (dfData[x1]>t1 ) & (dfData[x2]>t2 ) 
    mC      = (dfData[x1]<t1 ) & (dfData[x2]<t2 ) 
    mD      = (dfData[x1]>t1 ) & (dfData[x2]<t2 ) 
    regions['A'].fill(dfData[mA][xx], weight=1)
    regions['B'].fill(dfData[mB][xx], weight=1)
    regions['C'].fill(dfData[mC][xx], weight=1)
    regions['D'].fill(dfData[mD][xx], weight=1)
    #inclusive.fill(dfData[xx])

    print("\n\nData counts in ABCD regions after filling")
    print("Region A Sum ", regions["A"].sum())
    print("Region B Sum ", regions["B"].sum())
    print("Region C Sum ", regions["C"].sum())
    print("Region D Sum ", regions["D"].sum())

    originalVariances = {
        'A' : regions["A"].variances()[:].copy(),
        'B' : regions["B"].variances()[:].copy(),
        'C' : regions["C"].variances()[:].copy(),
        'D' : regions["D"].variances()[:].copy()}
    
    # remove MC from non QCD processes simulations from A, C, D
    print("\n\nRemoving MC from CRs A, C, D")
    #for idx, df in enumerate(dfsMC):
    #print(idx, dfProcessesMC.process[isMCList[idx]])
    mA      = (dfMC[x1]<t1 ) & (dfMC[x2]>=t2 ) 
    mB      = (dfMC[x1]>=t1 ) & (dfMC[x2]>=t2 ) 
    mC      = (dfMC[x1]<t1 ) & (dfMC[x2]<t2 ) 
    mD      = (dfMC[x1]>=t1 ) & (dfMC[x2]<t2 ) 
    
    # Subtract the events by filling with opposite weights (variances will be updated consequently)
    # Hist will sum the variances. Variances theorem
    regions['A'].fill(dfMC[mA][xx], weight=-dfMC[mA].weight)  
    regions['C'].fill(dfMC[mC][xx], weight=-dfMC[mC].weight)  
    regions['D'].fill(dfMC[mD][xx], weight=-dfMC[mD].weight)  
    # In B don't do it, we want to see the excess from Data - QCD = MCnonQCD
        
    print("Data counts in ABCD regions after MC subtraction")
    print("Region A Sum ", regions["A"].sum())
    print("Region B Sum ", regions["B"].sum())
    print("Region C Sum ", regions["C"].sum())
    print("Region D Sum ", regions["D"].sum())

    variancesAfterMCSubtraction = {
        'A' : regions["A"].variances()[:],
        'B' : regions["B"].variances()[:],
        'C' : regions["C"].variances()[:],
        'D' : regions["D"].variances()[:]}
    
    print("Error new after subtraction / Error before subtraction")
    for idx, region in enumerate(regions.keys()):
        print("Region ", region)
        errorRatio = np.sqrt(np.array(variancesAfterMCSubtraction[region])/np.array(originalVariances[region]))
        np.set_printoptions(precision=5)
        print(errorRatio)

# Fist Plot
    # Compute hB_ADC from ABCD. Note A, D, C are not poissonian after subtraction
    # Plot 4 regions of ABCD
    # The variances are propagated from A, D, C. They give uncertainties on QCD
    hB_ADC = plot4ABCD(regions=regions, bins=bins, x1=x1, x2=x2, t1=t1, t2=t2, suffix=suffix, unblinded_mask=unblinded_mask, sameWidth_flag=sameWidth_flag)
    
    if corrections is not None:
        #print("Corrections applied")
        #print(hB_ADC.variances()[:]*corrections**2)
        #print((hB_ADC.values()[:]*err_corrections)**2)
        #print(2*hB_ADC.values()[:]*corrections*covs)
        
        hB_ADC.variances()[:] = err_corrections**2
        hB_ADC.values()[:] = hB_ADC.values()[:]*corrections 

    
    fig, ax =plt.subplots(1, 1)
    ax.errorbar(x-np.diff(bins)/6, np.zeros(len(x)), yerr=np.sqrt(originalVariances['B']), label='original')
    ax.errorbar(x+0*np.diff(bins)/6, np.zeros(len(x)), yerr=np.sqrt(variancesAfterMCSubtraction['B']), label='After MC subtraction')
    ax.errorbar(x+np.diff(bins)/6, np.zeros(len(x)), yerr=np.sqrt(hB_ADC.variances()[:]), label='ABCD estimation')
    ax.legend()
    ax.set_ylabel("Uncertainty in SR")
    ax.set_xlabel("Dijet Mass [GeV]")
    fig.savefig("/t3home/gcelotto/ggHbb/abcd/new/plots/variances/var_%s.png"%suffix, bbox_inches='tight')


# Second Plot
    countsDict_SR = SM_SR(regions, hB_ADC, bins, dfData, dfMC, isMCList, dfProcessesMC, x1, t1, x2, t2, lumi, suffix=suffix, unblinded_mask=unblinded_mask,  chi2_mask=chi2_mask, sameWidth_flag=sameWidth_flag)

    #SM_CR('A', bins, dfMC, isMCList, dfProcessesMC, x1, t1, x2, t2, lumi, suffix, blindPar, sameWidth_flag=True)
    #SM_CR('C', bins, dfMC, isMCList, dfProcessesMC, x1, t1, x2, t2, lumi, suffix, blindPar, sameWidth_flag=True)
    #SM_CR('D', bins, dfMC, isMCList, dfProcessesMC, x1, t1, x2, t2, lumi, suffix, blindPar, sameWidth_flag=True)
    #SM_CR('B', bins, dfMC, isMCList, dfProcessesMC, x1, t1, x2, t2, lumi, suffix, blindPar, sameWidth_flag=True)
    


    # put negative values to countsDict_SR in order to subtract from SR
    print("Subtracting MC from SR")
    for key in countsDict_SR:
        countsDict_SR[key].values()[:] = -countsDict_SR[key].values()[:]
    t_ = regions['B'].copy()
    qcd__DataMinusMC = regions['B'] + countsDict_SR['H'] + countsDict_SR['ttbar'] + countsDict_SR['ST'] + countsDict_SR['VV'] + countsDict_SR['VV'] + countsDict_SR['WJets'] + countsDict_SR['ZJets']
    assert regions['B']==t_
    ## Restore positive histograms
    for key in countsDict_SR:
        countsDict_SR[key].values()[:] = -countsDict_SR[key].values()[:]
    
    #print("Variances with ABCD")
    #print(hB_ADC.variances()[:])
    pulls_QCD_SR, err_QCD_SR = QCD_SR(bins, hB_ADC, qcd__DataMinusMC, lumi=lumi, suffix=suffix, unblinded_mask=unblinded_mask,  sameWidth_flag=sameWidth_flag, outName="/t3home/gcelotto/ggHbb/abcd/new/plots/QCDclosure/QCD_closure_%s.png"%suffix, corrected=True if corrections is not None else False, chi2_mask=chi2_mask)
    plt.close('all')
    pulls_QCDPlusSM_SR = QCDplusSM_SR(bins, regions, countsDict_SR, hB_ADC, lumi=lumi, suffix=suffix, unblinded_mask=unblinded_mask, sameWidth_flag=sameWidth_flag, corrected=True if corrections is not None else False, chi2_mask=chi2_mask)
    plt.close('all')
#
    if corrections is not None:
        createRootHists(countsDict_SR, hB_ADC, regions, bins, suffix)
    return pulls_QCD_SR, err_QCD_SR