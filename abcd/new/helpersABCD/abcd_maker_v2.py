import sys
sys.path.append("/t3home/gcelotto/ggHbb/abcd/new/helpersABCD")
from plot_v2 import plot4ABCD, QCD_SR, QCDplusSM_SR, SM_SR
from createRootHists import createRootHists
from hist import Hist
import matplotlib.pyplot as plt

def ABCD(dfsData, dfsMC, x1, x2, xx, bins, t1, t2, isMCList, dfProcessesMC, lumi, suffix, blindPar):
    hA = Hist.new.Reg(len(bins)-1, bins[0], bins[-1], name="mjj").Weight()
    hB = Hist.new.Reg(len(bins)-1, bins[0], bins[-1], name="mjj").Weight()
    hC = Hist.new.Reg(len(bins)-1, bins[0], bins[-1], name="mjj").Weight()
    hD = Hist.new.Reg(len(bins)-1, bins[0], bins[-1], name="mjj").Weight()
    regions = {
        'A' : hA,
        'B' : hB,
        'C' : hC,
        'D' : hD,
    }


    # Fill regions with data
    for df in dfsData:
        mA      = (df[x1]<t1 ) & (df[x2]>t2 ) 
        mB      = (df[x1]>t1 ) & (df[x2]>t2 ) 
        mC      = (df[x1]<t1 ) & (df[x2]<t2 ) 
        mD      = (df[x1]>t1 ) & (df[x2]<t2 ) 
        regions['A'].fill(df[mA][xx])
        regions['B'].fill(df[mB][xx])
        regions['C'].fill(df[mC][xx])
        regions['D'].fill(df[mD][xx])

    print("Data counts in ABCD regions")
    print("Region A : ", regions["A"].sum())
    print("Region B : ", regions["B"].sum())
    print("Region C : ", regions["C"].sum())
    print("Region D : ", regions["D"].sum())
    # remove MC from non QCD processes simulations from A, C, D
    for idx, df in enumerate(dfsMC):
        print(idx, dfProcessesMC.process[isMCList[idx]])
        mA      = (df[x1]<t1 ) & (df[x2]>t2 ) 
        mB      = (df[x1]>t1 ) & (df[x2]>t2 ) 
        mC      = (df[x1]<t1 ) & (df[x2]<t2 ) 
        mD      = (df[x1]>t1 ) & (df[x2]<t2 ) 
        # Subtract the events by filling with opposite weights (variances will be updated consequently)
        # Hist will sum the variances. Variances theorem
        regions['A'].fill(df[mA][xx], weight=-df[mA].weight)  
        regions['C'].fill(df[mC][xx], weight=-df[mC].weight)  
        regions['D'].fill(df[mD][xx], weight=-df[mD].weight)  
        # In B don't do it, we want to see the excess from Data - QCD = MCnonQCD
        

# Fist Plot
    # Compute hB_ADC from ABCD. Note A, D, C are not poissonian after subtraction
    # Plot 4 regions of ABCD
    # The variances are propagated from A, D, C. They give uncertainties on QCD
    hB_ADC = plot4ABCD(regions=regions, bins=bins, x1=x1, x2=x2, t1=t1, t2=t2, suffix=suffix, blindPar=blindPar)
    
    plt.close('all')    
# Second Plot
    countsDict = SM_SR(regions, hB_ADC, bins, dfsData, dfsMC, isMCList, dfProcessesMC, x1, t1, x2, t2, lumi, suffix=suffix, blindPar=blindPar)
    plt.close('all')



    for letter in ['A', 'B', 'C', 'D']:
        print(regions[letter].sum())


    # put negative values to countsDict
    print("Put negative values")
    for key in countsDict:
        countsDict[key].values()[:] = -countsDict[key].values()[:]
    print("regions B values", regions["B"].values())
    qcd_mc = regions['B'] + countsDict['H'] + countsDict['ttbar'] + countsDict['ST'] + countsDict['VV'] + countsDict['VV'] + countsDict['WJets'] + countsDict['ZJets']
    ## Restore positive histograms
    for key in countsDict:
        countsDict[key].values()[:] = -countsDict[key].values()[:]
    print(print("regions B values unchagned", regions["B"].values()))
    QCD_SR(bins, hB_ADC, qcd_mc, lumi=lumi, suffix=suffix, blindPar=blindPar)
    plt.close('all')
    QCDplusSM_SR(bins, regions, countsDict, hB_ADC, lumi=lumi, suffix=suffix, blindPar=blindPar)
    plt.close('all')
#
    #createRootHists(countsDict, hB_ADC, regions, bins, suffix)