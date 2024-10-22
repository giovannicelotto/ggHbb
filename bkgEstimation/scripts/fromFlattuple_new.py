import numpy as np
import pandas as pd
import glob, sys, re
import random
import matplotlib.pyplot as plt
import mplhep as hep
import awkward as ak
import subprocess
import json
sys.path.append("/t3home/gcelotto/ggHbb/PNN/helpers")
from preprocessMultiClass import preprocessMultiClass
from functions import loadMultiParquet, cut
hep.style.use("CMS")
epsilon=1e-7



def closure(nReal, nMC):
    # Define name of the process, folder for the files and xsections
    outFolder="/t3home/gcelotto/ggHbb/bkgEstimation/output"

    
    currentLumi = nReal * 0.774 / 1017
    featureDisplay='PNN'
    bins=np.linspace(0, 1, 21)




    predictionsPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/PNNpredictions_v3b"
    isMCList = [0, 1,
            2,
            3, 4, 5,
            6,7,8,9,10,11,
            12,13,14,
            15,16,17,18,19,
            20, 21, 22, 23, 36,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35 # QCD
            #39    # Data2A
    ]
    if isMCList[-1]==39:
        nReal = nReal *2
    dfProcesses = pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processes.csv")
    processes = dfProcesses.process[isMCList].values

    predictionsFileNames = []
    for p in processes:
        print(p)
        predictionsFileNames.append(glob.glob(predictionsPath+"/%s/*.parquet"%p))


    predictionsFileNumbers = []
    for isMC, p in zip(isMCList, processes):
        idx = isMCList.index(isMC)
        print("Process %s # %d"%(p, isMC))
        l = []
        for fileName in predictionsFileNames[idx]:
            print
            fn = re.search(r'fn(\d+)\.parquet', fileName).group(1)
            l.append(int(fn))

        predictionsFileNumbers.append(l)


    paths = list(dfProcesses.flatPath[isMCList])
    dfs= []
    print(predictionsFileNumbers)
    dfs, numEventsList, fileNumberList = loadMultiParquet(paths=paths, nReal=nReal, nMC=nMC,
                                                        columns=['sf', 'dijet_mass', 'dijet_pt', 'jet1_pt',
                                                                'jet2_pt','jet1_mass', 'jet2_mass', 'jet1_eta',
                                                                'jet2_eta', 'jet1_qgl', 'jet2_qgl', 'dijet_dR',
                                                                    'jet3_mass', 'jet3_qgl', 'Pileup_nTrueInt',
                                                                'jet2_btagDeepFlavB', 'dijet_cs','leptonClass',
                                                                'jet1_btagDeepFlavB',  'muon_pt'],
                                                                returnNumEventsTotal=True, selectFileNumberList=predictionsFileNumbers,
                                                                returnFileNumberList=True)
    
    preds = []
    predictionsFileNamesNew = []
    for isMC, p in zip(isMCList, processes):
        idx = isMCList.index(isMC)
        print("Process %s # %d"%(p, isMC))
        l =[]
        for fileName in predictionsFileNames[idx]:
            print(fileName)
            fn = int(re.search(r'fn(\d+)\.parquet', fileName).group(1))
            if fn in fileNumberList[idx]:
                l.append(fileName)
        predictionsFileNamesNew.append(l)

        print(len(predictionsFileNamesNew[idx]), " files for process")
        df = pd.read_parquet(predictionsFileNamesNew[idx])
        preds.append(df)


    dfs = preprocessMultiClass(dfs=dfs)
    dfs = cut(dfs, 'jet1_pt', 20, None)
    dfs = cut(dfs, 'jet2_pt', 20, None)
    #dfs = cut(dfs, 'leptonClass', 0, 2)
    dfs_precut = dfs.copy()
    print(len(preds[0]))
    print(len(dfs[0]))
    countsDic={}
    for idx, df in enumerate(dfs):
        isMC = isMCList[idx]
        process = dfProcesses.process[isMC]
        xsection = dfProcesses.xsection[isMC]
        print("isMC ", isMC)
        print("Process ", process)
        print("Xsection ", xsection)
        dfs[idx]['weight'] = df.PU_SF*df.sf*xsection * nReal * 1000 * 0.774 /1017/numEventsList[idx] if 'Data' not in process else 1
        dfs[idx]['PNN'] = np.array(preds[idx])
        


        #dfProcess=dfProcess[(dfProcess.jet1_pt>20) & (dfProcess.jet2_pt>20) & (dfProcess.muon_pt>7) & (abs(dfProcess.muon_eta)<1.5)]
        if process != 'Data':
            print("="*25)
            counts = np.histogram(df[featureDisplay], bins=bins, weights=df.weight )[0] #
            countsDic[process] = counts
            print("Flat Efficiency for ",process," : ", len(df)/numEventsList[idx])
        else :
            print("Data : ", process)
            counts = np.histogram(df[featureDisplay], bins=bins, weights=None)[0]
            dataCounts = counts
            dataErr = np.sqrt(dataCounts)

    fig,(ax1, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.1)
    ax1.errorbar((bins[1:]+bins[:-1])/2, dataCounts, xerr=np.diff(bins)/2, yerr=dataErr, marker='o', color='black', linestyle='none', label='Data')
    compactDict = {}
    compactDict['VV'] = countsDic['WW'] + countsDic['WZ'] + countsDic['ZZ']
    qcd_sum, wjets, zjets, singleTop=0, 0, 0, 0
    for key, value in countsDic.items():
        if 'QCD' in key:
            qcd_sum += value
        elif 'WJets' in key:
            wjets+=value
        elif 'ZJets' in key:
            zjets+=value
        elif 'ST_' in key:
            singleTop+=value
    compactDict['H']= countsDic['GluGluHToBB']
    compactDict['ST']= singleTop
    compactDict['tt'] = countsDic['TTTo2L2Nu'] +  countsDic['TTToHadronic'] + countsDic['TTToSemiLeptonic']
    compactDict['W+Jets']= wjets
    compactDict['Z+Jets']= zjets
    compactDict['QCD']= qcd_sum
    # plot the MC processes compacted in categories
    allCounts = np.zeros(len(bins)-1)
    print("="*50)
    for key in compactDict.keys():
        print(key, compactDict[key])
        ax1.bar((bins[1:]+bins[:-1])/2, compactDict[key] , align='center', width=np.diff(bins), label=key, alpha = 1, bottom=allCounts)
        allCounts = allCounts + compactDict[key]

    hep.cms.label(lumi=round(float(currentLumi), 4), ax=ax1)
    ax1.set_yscale('log')
    ax1.set_xlim(bins[0], bins[-1])
    ax2.set_xlim(bins[0], bins[-1])
    ax2.set_ylim(0, 2)
    ax1.set_ylim(10**0, ax1.get_ylim()[1])
    ax2.hlines(y=1, xmin=bins[0], xmax=bins[-1], color='black')
    ax2.set_xlabel(featureDisplay)    #"$\mathrm{Muon p_{T}}$ [GeV]"
    ax1.set_ylabel("Events")
    ax2.set_ylabel("Data/MC", fontsize=28)
    print(("="*30))
    ax2.errorbar((bins[1:]+bins[:-1])/2, (dataCounts/(allCounts+epsilon)), marker='o', color='black', linewidth=1, linestyle='', label='MC', alpha=1)
    ax1.legend(bbox_to_anchor=(1 ,1), ncols=1)
    outName = outFolder+"/%s_trigAndPU_jet20.png"%(featureDisplay)
    print("Savin in ", outName)
    fig.savefig(outName, bbox_inches='tight')


    print("Data/MC", dataCounts/(allCounts+epsilon))


       
    return 

if __name__ == "__main__":
    nReal, nMC = sys.argv[1:]
    nReal = int(nReal)
    nMC = int(nMC)
    if nReal==-1:
        nReal=1017

    closure(nReal, nMC)