import sys
sys.path.append("/t3home/gcelotto/ggHbb/abcd/new/helpersABCD")
from plot import plotQCDhists_SR_CR, controlPlotABCD, plotQCDClosure, plotQCDPlusSM
from createRootHists import createRootHists
from hist import Hist

def ABCD(dfs, x1, x2, xx, bins, t1, t2, isMCList, dfProcesses, nReal, suffix):
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
    mA      = (dfs[0][x1]<t1 ) & (dfs[0][x2]>t2 ) 
    mB      = (dfs[0][x1]>t1 ) & (dfs[0][x2]>t2 ) 
    mC      = (dfs[0][x1]<t1 ) & (dfs[0][x2]<t2 ) 
    mD      = (dfs[0][x1]>t1 ) & (dfs[0][x2]<t2 ) 
    regions['A'].fill(dfs[0][mA][xx])
    regions['B'].fill(dfs[0][mB][xx])
    regions['C'].fill(dfs[0][mC][xx])
    regions['D'].fill(dfs[0][mD][xx])

    print("Data counts in ABCD regions")
    print("Region A : ", regions["A"].sum())
    print("Region B : ", regions["B"].sum())
    print("Region C : ", regions["C"].sum())
    print("Region D : ", regions["D"].sum())

    # remove MC from non QCD processes simulations from A, C, D
    for idx, df in enumerate(dfs[1:]):
        print(idx, df.dijet_mass.mean())
        mA      = (df[x1]<t1 ) & (df[x2]>t2 ) 
        mB      = (df[x1]>t1 ) & (df[x2]>t2 ) 
        mC      = (df[x1]<t1 ) & (df[x2]<t2 ) 
        mD      = (df[x1]>t1 ) & (df[x2]<t2 ) 
        # Subtract the events by filling with opposite weights
        regions['A'].fill(df[mA][xx], weight=-df[mA].weight)  
        regions['C'].fill(df[mC][xx], weight=-df[mC].weight)  
        regions['D'].fill(df[mD][xx], weight=-df[mD].weight)  
        # In B don't do it, we want to see the excess from Data - QCD = MCnonQCD
        #regions['B'] = regions['B'] - np.histogram(df[mB][xx], bins=bins, weights=df[mB].weight)[0]

    hB_ADC = plotQCDhists_SR_CR(regions=regions, bins=bins, x1=x1, x2=x2, t1=t1, t2=t2, suffix=suffix)

    countsDict = controlPlotABCD(regions, hB_ADC, bins, dfs, isMCList, dfProcesses, x1, t1, x2, t2, nReal, suffix=suffix)



    for letter in ['A', 'B', 'C', 'D']:
        print(regions[letter].sum())


    # put negative values to countsDict
    for key in countsDict:
        countsDict[key].values()[:] = -countsDict[key].values()[:]

    qcd_mc = regions['B'] + countsDict['H'] + countsDict['ttbar'] + countsDict['ST'] + countsDict['VV'] + countsDict['VV'] + countsDict['W+Jets'] + countsDict['Z+Jets']
    # Restore positive histograms
    for key in countsDict:
        countsDict[key].values()[:] = -countsDict[key].values()[:]

    plotQCDClosure(bins, hB_ADC, qcd_mc, suffix)
    plotQCDPlusSM(bins, regions, countsDict, hB_ADC, suffix)

    createRootHists(countsDict, hB_ADC, regions, bins, suffix)