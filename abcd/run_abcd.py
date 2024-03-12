import numpy as np
import pandas as pd
import glob, sys
import random
import matplotlib.pyplot as plt
import mplhep as hep
import awkward as ak
import subprocess
import time
hep.style.use("CMS")

def getProcessesDataFrame():
    processes = {
        'Data':                         ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A_2023Nov30", -1],
        'WW':                               ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/diboson2024Feb14/WW_TuneCP5_13TeV-pythia8", 75.8],
        'WZ':                               ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/diboson2024Feb14/WZ_TuneCP5_13TeV-pythia8",27.6],
        'ZZ':                               ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/diboson2024Feb14/ZZ_TuneCP5_13TeV-pythia8",12.14	],
        'ST_s-channel-hadronic':            ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/singleTop2024Feb14/ST_s-channel_4f_hadronicDecays_TuneCP5_13TeV-amcatnlo-pythia8", 11.24],
        'ST_s-channel-leptononic':          ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/singleTop2024Feb14/ST_s-channel_4f_leptonDecays_TuneCP5CR1_13TeV-amcatnlo-pythia8",3.74],
        'ST_t-channel-antitop':             ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/singleTop2024Feb14/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5CR1_13TeV-powheg-madspin-pythia8",69.09],
        'ST_t-channel-top':                 ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/singleTop2024Feb14/ST_t-channel_top_4f_InclusiveDecays_TuneCP5CR1_13TeV-powheg-madspin-pythia8", 115.3],
        'ST_tW-antitop':                    ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/singleTop2024Feb14/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8", 34.97],
        'ST_tW-top':                        ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/singleTop2024Feb14/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8", 34.91	],
        'TTTo2L2Nu':                        ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ttbar2024Feb14/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8", 831*0.09],
        'TTToHadronic':                     ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ttbar2024Feb14/TTToHadronic_TuneCP5_13TeV-powheg-pythia8", 831*0.46],
        'TTToSemiLeptonic':                 ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ttbar2024Feb14/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8", 831*0.45],
        'WJetsToLNu':                       ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/WJetsToLNu2024Feb20/*", 62070.0],
        'WJetsToQQ_200to400'      :         ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/WJetsToQQ2024Feb20/WJetsToQQ_HT-200to400*",2549.0],
        'WJetsToQQ_400to600'      :         ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/WJetsToQQ2024Feb20/WJetsToQQ_HT-400to600*",276.5],
        'WJetsToQQ_600to800'      :         ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/WJetsToQQ2024Feb20/WJetsToQQ_HT-600to800*",59.25],
        'WJetsToQQ_800toInf'      :         ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/WJetsToQQ2024Feb20/WJetsToQQ_HT-800toInf*",28.75],
        'ZJetsToQQ_200to400'      :         ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ZJetsToQQ2024Feb20/ZJetsToQQ_HT-200to400*",1012.0],
        'ZJetsToQQ_400to600'      :         ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ZJetsToQQ2024Feb20/ZJetsToQQ_HT-400to600*",114.2],
        'ZJetsToQQ_600to800'      :         ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ZJetsToQQ2024Feb20/ZJetsToQQ_HT-600to800*",25.34],
        'ZJetsToQQ_800toInf'      :         ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ZJetsToQQ2024Feb20/ZJetsToQQ_HT-800toInf*",12.99],
        'QCD_MuEnriched_Pt-1000':           ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-1000*", 1.085],
        'QCD_MuEnriched_Pt-800To1000':      ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-800To1000*", 3.318],
        'QCD_MuEnriched_Pt-600To800':       ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-600To800*", 18.12	],
        'QCD_MuEnriched_Pt-470To600':       ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-470To600*", 58.9],
        'QCD_MuEnriched_Pt-300To470':       ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-300To470*", 622.6],
        'QCD_MuEnriched_Pt-170To300':       ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-170To300*", 7000.0],
        'QCD_MuEnriched_Pt-120To170':       ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-120To170*", 21280.0],
        'QCD_MuEnriched_Pt-80To120':        ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-80To120*", 87740.0],
        'QCD_MuEnriched_Pt-50To80':         ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-50To80*", 381700.0],
        'QCD_MuEnriched_Pt-30To50':         ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-30To50*", 1367000.0],
        'QCD_MuEnriched_Pt-20To30':         ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-20To30*", 2527000.0],
        'QCD_MuEnriched_Pt-15To20':         ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-15To20*", 2800000.0	],

    }
    df = pd.DataFrame(processes).T
    df.columns = ['path', 'xsection']
    return df

def closure(nFilesData, nFilesMC):
    # Define name of the process, folder for the files and xsections
    df = getProcessesDataFrame()
    df.to_csv("/t3home/gcelotto/ggHbb/bkgEstimation/output/processes.csv")
    
    currentLumi = nFilesData * 0.774 / 1017
    np.save('/t3home/gcelotto/ggHbb/bkgEstimation/output/currentLumi.npy', currentLumi)
    # choose binning for HT
    bins=np.linspace(0, 800, 20)
    toSkip = df.index
    np.save('/t3home/gcelotto/ggHbb/bkgEstimation/output/binsForHT.npy', bins)
    for (process, path, xsection) in zip(df.index, df.path, df.xsection):
        print("Starting process ", process)
        if process in list(toSkip):
            print("skipping process ....", process)
            continue
        print(process, list(toSkip))
        time.sleep(50)
        nFiles=nFilesData if process=='Data' else nFilesMC
        
        fileNames = glob.glob(path+"/**/*.root", recursive=True)
        nFiles = nFiles if nFiles != -1 else len(fileNames)
        if nFiles > len(fileNames) :
            nFiles = len(fileNames)
        
        if process!='Data':
            np.save("/t3home/gcelotto/ggHbb/bkgEstimation/output/mini_%s.npy"%process, np.float32(0))
        counts = np.zeros(len(bins)-1) # counts of MC sample
        np.save("/t3home/gcelotto/ggHbb/bkgEstimation/output/counts_%s.npy"%process, counts)

        print("Process : ", process, "     %d files"%nFiles)
        for fileName in fileNames[:nFiles]:
            subprocess.run(['sbatch', '-J', process+"%d"%random.randint(1, 4), '/t3home/gcelotto/ggHbb/bkgEstimation/scripts/process_job.sh', fileName, process])
       
    return 

if __name__ == "__main__":
    nFilesData, nFilesMC = sys.argv[1:]
    nFilesData = int(nFilesData)
    nFilesMC = int(nFilesMC)

    closure(nFilesData, nFilesMC)