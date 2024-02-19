import numpy as np
import pandas as pd
import glob, sys
import uproot
import matplotlib.pyplot as plt
import mplhep as hep
import awkward as ak
import subprocess
hep.style.use("CMS")

def getProcessesDataFrame():
    processes = {
        'Data':                         ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A_2023Nov30", -1],
        'QCD_MuEnriched_Pt-15To20':   ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-15To20*", 2800000.0	],
        'QCD_MuEnriched_Pt-20To30':   ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-20To30*", 2527000.0],
        'QCD_MuEnriched_Pt-30To50':   ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-30To50*", 1367000.0],
        'QCD_MuEnriched_Pt-50To80':   ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-50To80*", 381700.0],
        'QCD_MuEnriched_Pt-80To120':   ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-80To120*", 87740.0],
        'QCD_MuEnriched_Pt-120To170':   ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-120To170*", 21280.0],
        'QCD_MuEnriched_Pt-170To300':   ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-170To300*", 7000.0],
        'QCD_MuEnriched_Pt-300To470':   ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-300To470*", 622.6],
        'QCD_MuEnriched_Pt-470To600':   ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-470To600*", 58.9],
        'QCD_MuEnriched_Pt-600To800':   ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-600To800*", 18.12	],
        'QCD_MuEnriched_Pt-800To1000':   ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-800To1000*", 3.318],
        'QCD_MuEnriched_Pt-1000':       ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-1000*", 1.085],
        #'GluGluHToBB':                  ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/GluGluHToBB2024Feb16", 30.52],
        'TTTo2L2Nu':                 ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ttbar2024Feb14/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8", 831*0.46],
        'TTToSemiLeptonic':             ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ttbar2024Feb14/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8", 831*0.45],
        'TTToHadronic':                    ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ttbar2024Feb14/TTToHadronic_TuneCP5_13TeV-powheg-pythia8", 831*0.09],
    }
    df = pd.DataFrame(processes).T
    df.columns = ['path', 'xsection']
    return df

def closure():
    # Define name of the process, folder for the files and xsections
    df = getProcessesDataFrame()
    df.to_csv("/t3home/gcelotto/ggHbb/bkgEstimation/CountsBins/processes.csv")
    
    # choose binning for HT
    bins=np.linspace(0, 800, 20)
    np.save('/t3home/gcelotto/ggHbb/bkgEstimation/CountsBins/binsForHT.npy', bins)
  
    for (process, path, xsection) in zip(df.index, df.path, df.xsection):
        nFiles=10 if process=='Data' else 1
        
        fileNames = glob.glob(path+"/**/*.root", recursive=True)
        nFiles = nFiles if nFiles != -1 else len(fileNames)
        if nFiles > len(fileNames) :
            nFiles = len(fileNames)
        if process=='Data': 
            currentLumi = nFiles * 0.774 / 1017
            np.save('/t3home/gcelotto/ggHbb/bkgEstimation/CountsBins/currentLumi.npy', currentLumi)
        
        np.save("/t3home/gcelotto/ggHbb/bkgEstimation/CountsBins/mini_%s.npy"%process, np.float32(0))
        counts = np.zeros(len(bins)-1) # counts of MC sample
        np.save("/t3home/gcelotto/ggHbb/bkgEstimation/CountsBins/counts_%s.npy"%process, counts)

        print("Starting process : ", process, "     %d files"%nFiles)
        for fileName in fileNames[:nFiles]:
            subprocess.run(['sbatch', '/t3home/gcelotto/ggHbb/bkgEstimation/scripts/process_job.sh', fileName, process])
       
    return 

if __name__ == "__main__":
    closure()