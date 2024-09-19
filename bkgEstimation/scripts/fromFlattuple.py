import numpy as np
import pandas as pd
import glob, sys, re
import random
import matplotlib.pyplot as plt
import mplhep as hep
import awkward as ak
import subprocess
import json
hep.style.use("CMS")
epsilon=1e-7

def load_mapping_dict(file_path):
    with open(file_path, 'r') as file:
        mapping_dict = json.load(file)
        # Convert keys back to integers if needed (they should already be integers)
        mapping_dict = {int(k): v for k, v in mapping_dict.items()}

    return mapping_dict


def closure(nFilesData, nFilesMC):
    # Define name of the process, folder for the files and xsections
    outFolder="/t3home/gcelotto/ggHbb/bkgEstimation/output"

    df = pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processes.csv")
    currentLumi = nFilesData * 0.774 / 1017
    featureDisplay='muon_pt'
    bins=np.linspace(0, 50, 51)
    toSkip = [
        #'Data',                      
        #'WW',                           'WZ',
        #'ZZ',                          
        #'ST_s-channel-hadronic',        
        #'ST_s-channel-leptononic',
        #'ST_t-channel-antitop',         
        #'ST_t-channel-top',          
        #'ST_tW-antitop',
        #'ST_tW-top',                    
        #'TTTo2L2Nu',                    'TTToHadronic',
        #'TTToSemiLeptonic',
        #'WJetsToLNu',
        #'WJetsToQQ_200to400',
        #'WJetsToQQ_400to600',
        #'WJetsToQQ_600to800',
        #'WJetsToQQ_800toInf',
        #'ZJetsToQQ_200to400',
        #'ZJetsToQQ_400to600',
        #'ZJetsToQQ_600to800',
        #'ZJetsToQQ_800toInf',
        #'QCD_MuEnriched_Pt-1000',       'QCD_MuEnriched_Pt-800To1000',  'QCD_MuEnriched_Pt-600To800',
        #'QCD_MuEnriched_Pt-470To600',   'QCD_MuEnriched_Pt-300To470',   'QCD_MuEnriched_Pt-170To300',
        #'QCD_MuEnriched_Pt-120To170',   'QCD_MuEnriched_Pt-80To120',    'QCD_MuEnriched_Pt-50To80',
        #'QCD_MuEnriched_Pt-30To50',     'QCD_MuEnriched_Pt-20To30',     'QCD_MuEnriched_Pt-15To20',
    ]

    # open the dict

    countsDic={}

    print("="*50,"\nMiniDf found\n","="*50)
    miniDf = pd.read_csv("/t3home/gcelotto/ggHbb/outputs/counters/miniDf_June.csv")
    PU_map = load_mapping_dict('/t3home/gcelotto/ggHbb/PU_reweighting/profileFromData/PU_PVtoPUSF.json')

    for (process, path, xsection) in zip(df.process, df.flatPath, df.xsection):
        print("Starting process ", process)
        if process in list(toSkip):
            print("skipping process ....", process)
            continue
        nFiles=nFilesData if process=='Data' else nFilesMC
        fileNames = glob.glob(path+"/**/*.parquet", recursive=True)
        nFiles = nFiles if nFiles != -1 else len(fileNames)
        fileNames = fileNames[:nFiles]
        print(len(fileNames), " files found ")
        

        #print(miniDf[(miniDf.process==process) & (miniDf.fileNumber==int(fileNumber))])
        mini = 0
        if process!='Data':
            for fileName in fileNames:
                fileNumber = re.search(r'_(\d+)\.parquet', fileName).group(1)
                add = miniDf[(miniDf.process==process) & (miniDf.fileNumber==int(fileNumber)) ].numEventsPassed.sum()
                if add==0:
                    print(process, fileNumber, add)
                mini=mini+ add
        columns=['sf', 'jet1_pt','jet2_pt','muon_pt', 'muon_eta', 'muon_dxySig', 'Pileup_nTrueInt','PV_npvs']
        if process=='Data':
            columns.remove('Pileup_nTrueInt') 
        dfProcess=pd.read_parquet(fileNames, filters=[('muon_pt', '>=', 7)])


        dfProcess=dfProcess[(dfProcess.jet1_pt>20) & (dfProcess.jet2_pt>20) & (dfProcess.muon_pt>7) & (abs(dfProcess.muon_eta)<1.5)]
        if process!='Data':
            #print(dfProcess.loc[dfProcess.Pileup_nTrueInt.isna()])
            print("="*25)
            dfProcess['PU_SF'] = dfProcess['Pileup_nTrueInt'].apply(int).map(PU_map)
            dfProcess.loc[dfProcess['Pileup_nTrueInt'] > 98, 'PU_SF'] = 0
            #print(dfProcess.PU_SF)
        if process != 'Data':
            #mini = np.load(outFolder+"/mini_%s.npy"%process)
            print("="*25)
            print(mini)
            counts = np.histogram(dfProcess[featureDisplay], bins=bins, weights=dfProcess.sf*dfProcess.PU_SF )[0] #
            counts = counts * xsection*1000*currentLumi/(mini+epsilon)
            countsDic[process] = counts
            print("Flat Efficiency for ",process," : ", len(dfProcess)/mini)
        else :

            counts = np.histogram(dfProcess[featureDisplay], bins=bins, weights=dfProcess.sf)[0]
            dataCounts = counts
            dataErr = np.sqrt(dataCounts)
            



    fig,(ax1, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.1)
    ax1.errorbar((bins[1:]+bins[:-1])/2, dataCounts, xerr=np.diff(bins)/2, yerr=dataErr, marker='o', color='black', linestyle='none', label='Data')
    compactDict = {}
    compactDict['VV'] = countsDic['WW'] + countsDic['WZ'] + countsDic['ZZ']
    qcd_sum, singleBoson, singleTop=0, 0, 0
    for key, value in countsDic.items():
        if 'QCD' in key:
            qcd_sum += value
        elif 'WJets' in key or 'ZJets' in key:
            singleBoson+=value
        elif 'ST_' in key:
            singleTop+=value
    compactDict['H']= countsDic['GluGluHToBB']
    compactDict['ST']= singleTop
    compactDict['tt'] = countsDic['TTTo2L2Nu'] +  countsDic['TTToHadronic'] + countsDic['TTToSemiLeptonic']
    compactDict['W/Z+Jets']= singleBoson
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
    outName = outFolder+"/%s_trigAndPU.png"%(featureDisplay)
    print("Savin in ", outName)
    fig.savefig(outName, bbox_inches='tight')


    print("Data/MC", dataCounts/(allCounts+epsilon))


       
    return 

if __name__ == "__main__":
    nFilesData, nFilesMC = sys.argv[1:]
    nFilesData = int(nFilesData)
    nFilesMC = int(nFilesMC)
    if nFilesData==-1:
        nFilesData=1017

    closure(nFilesData, nFilesMC)