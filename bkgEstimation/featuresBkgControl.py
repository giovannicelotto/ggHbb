import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import re, sys
import mplhep as hep
hep.style.use("CMS")

def getProcessesDataFrame():
    processes = {
        'Data':                             ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A", -1],
'GluGluHToBB':                              ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB", 48.52*0.5801],
        'WW':                               ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/diboson/WW", 75.8],
        'WZ':                               ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/diboson/WZ",27.6],
        'ZZ':                               ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/diboson/ZZ",12.14	],
        'ST_s-channel-hadronic':            ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/singleTop/s-channel_hadronic", 11.24],
        'ST_s-channel-leptononic':          ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/singleTop/s-channel_leptonic",3.74],
        'ST_t-channel-antitop':             ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/singleTop/t-channel_antitop",69.09],
        'ST_t-channel-top':                 ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/singleTop/t-channel_top", 115.3],
        'ST_tW-antitop':                    ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/singleTop/tW-channel_antitop", 34.97],
        'ST_tW-top':                        ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/singleTop/tW-channel_top", 34.91	],
        'TTTo2L2Nu':                        ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ttbar/ttbar2L2Nu", 831*0.09],
        'TTToHadronic':                     ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ttbar/ttbarHadronic", 831*0.46],
        'TTToSemiLeptonic':                 ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ttbar/ttbarSemiLeptonic", 831*0.45],
        'WJetsToLNu':                       ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/WJets/WJetsToLNu", 62070.0],
        'WJetsToQQ_200to400'      :         ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/WJets/WJetsToQQ_HT-200to400",2549.0],
        'WJetsToQQ_400to600'      :         ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/WJets/WJetsToQQ_HT-400to600",276.5],
        'WJetsToQQ_600to800'      :         ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/WJets/WJetsToQQ_HT-600to800",59.25],
        'WJetsToQQ_800toInf'      :         ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/WJets/WJetsToQQ_HT-800toInf",28.75],
        
        'ZJetsToQQ_200to400'      :         ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-200to400*",1012.0],
        'ZJetsToQQ_400to600'      :         ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-400to600*",114.2],
        'ZJetsToQQ_600to800'      :         ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-600to800*",25.34],
        'ZJetsToQQ_800toInf'      :         ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/ZJets/ZJetsToQQ_HT-800toInf*",12.99],

        'QCD_MuEnriched_Pt-1000':           ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/QCD_Pt1000*", 1.085],
        'QCD_MuEnriched_Pt-800To1000':      ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/QCD_Pt800To1000*", 3.318],
        'QCD_MuEnriched_Pt-600To800':       ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/QCD_Pt600To800*", 18.12	],
        'QCD_MuEnriched_Pt-470To600':       ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/QCD_Pt470To600*", 58.9],
        'QCD_MuEnriched_Pt-300To470':       ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/QCD_Pt300To470*", 622.6],
        'QCD_MuEnriched_Pt-170To300':       ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/QCD_Pt170To300*", 7000.0],
        'QCD_MuEnriched_Pt-120To170':       ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/QCD_Pt120To170*", 21280.0],
        'QCD_MuEnriched_Pt-80To120':        ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/QCD_Pt80To120*", 87740.0],
        'QCD_MuEnriched_Pt-50To80':         ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/QCD_Pt50To80*", 381700.0],
        'QCD_MuEnriched_Pt-30To50':         ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/QCD_Pt30To50*", 1367000.0],
        'QCD_MuEnriched_Pt-20To30':         ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/QCD_Pt20To30*", 2527000.0],
        'QCD_MuEnriched_Pt-15To20':         ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/QCD_Pt15To20*", 2800000.0	],

    }
    df = pd.DataFrame(processes).T
    df.columns = ['path', 'xsection']
    return df
def main(nData, nMC):
    miniDf = pd.read_csv("/t3home/gcelotto/ggHbb/abcd/output/miniDf.csv")
    datasets = getProcessesDataFrame()



    fig,(ax1, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 12))
    bins= np.linspace(0, 300, 101)
    allCounts = np.zeros(len(bins)-1)

    countsDict = {
        'Data':np.zeros(len(bins)-1),
        'H':np.zeros(len(bins)-1),
        'VV':np.zeros(len(bins)-1),
        'ST':np.zeros(len(bins)-1),
        'ttbar':np.zeros(len(bins)-1),
        'Z+Jets':np.zeros(len(bins)-1),
        'W+Jets':np.zeros(len(bins)-1),
        'W+Jets':np.zeros(len(bins)-1),
        'QCD':np.zeros(len(bins)-1),
    }

    lumiPerEvent = np.load("/t3home/gcelotto/ggHbb/outputs/lumiPerEvent.npy")


    for (process, path, xsection) in zip(datasets.index, datasets.path, datasets.xsection):
        nFiles = nData if process=='Data' else nMC
        fileNames = glob.glob(path+"/**/*.parquet", recursive=True)[:nFiles]
        if len(fileNames)==0:
            sys.exit("No file found for process %s"%process)
        df = pd.read_parquet(fileNames, columns=['dijet_mass', 'sf', 'dijet_pt', 'jet2_btagDeepFlavB'])
        if process=='Data':
            currentLumi=lumiPerEvent*len(df)
            print(currentLumi)
        else:
            mini=0
            for fileName in fileNames:
                fileNumber = int(re.search(r'\D(\d{1,4})\.\w+$', fileName).group(1))    
                mini = mini + miniDf[(miniDf.process==process) & (miniDf.fileNumber == fileNumber)].numEventsTotal.values[0]

            xsection = datasets[datasets.index==process].xsection.values[0]*1000
        

        df=df[(df.dijet_pt>100) & (df.jet2_btagDeepFlavB>0.8)]
        counts = np.histogram(df.dijet_mass, bins=bins, weights=df.sf)[0]
        if process=='Data':
            pass
        else:
            counts=counts*xsection / mini*currentLumi
        if 'Data' in process:
            countsDict['Data'] = countsDict['Data'] +counts
        elif 'GluGluHToBB' in process:
            
            countsDict['H'] = countsDict['H'] +counts
        elif 'ST' in process:
            countsDict['ST'] = countsDict['ST'] +counts
        elif 'TTTo' in process:
            countsDict['ttbar'] = countsDict['ttbar'] +counts
        elif 'QCD' in process:
            countsDict['QCD'] = countsDict['QCD'] +counts
        elif 'ZJets' in process:
            countsDict['Z+Jets'] = countsDict['Z+Jets'] +counts
        elif 'WJets' in process:
            countsDict['W+Jets'] = countsDict['W+Jets'] +counts
        elif (('WW' in process) | ('ZZ' in process) | ('WZ' in process)):
            countsDict['VV'] = countsDict['VV'] +counts

    for key in countsDict.keys():
        if key=='Data':
            ax1.errorbar((bins[1:]+bins[:-1])/2, countsDict[key], xerr=np.diff(bins)/2, yerr=np.sqrt(countsDict[key]), color='black', linestyle='none')[0]
        elif key=='H':
            ax1.errorbar((bins[1:]+bins[:-1])/2, countsDict[key], xerr=np.diff(bins)/2, color='darkgreen', linestyle='none')
        else:
            print(key, np.sum(countsDict[key]))
            ax1.hist(bins[:-1], bins=bins, weights=countsDict[key], bottom=allCounts, label=key)
            allCounts = allCounts + countsDict[key]
        

    ax1.set_yscale('log')
    hep.cms.label(lumi=round(float(currentLumi), 4), ax=ax1)
    ax1.set_ylim(1, ax1.get_ylim()[1]*1.3)
    ax1.set_xlim(bins[0], bins[-1])
    ax2.set_xlim(bins[0], bins[-1])
    
    ax1.legend()


    
    ax2.errorbar((bins[:-1]+bins[1:])/2, countsDict['Data']/allCounts, xerr=np.diff(bins)/2, color='black', linestyle='none')
    ax2.set_ylim(0.5, 1.5)
    ax2.hlines(y=1, xmin=bins[0], xmax=bins[-1], color='black')
    fig.savefig("/t3home/gcelotto/ggHbb/bkgEstimation/output/dijet_mass.png", bbox_inches='tight')

if __name__=="__main__":
    nData, nMC = int(sys.argv[1]), int(sys.argv[2])
    main(nData, nMC)