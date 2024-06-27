import numpy as np
import pandas as pd
import glob, sys
import random
import matplotlib.pyplot as plt
import mplhep as hep
import awkward as ak
import subprocess
hep.style.use("CMS")



def closure(nFilesData, nFilesMC):
    # Define name of the process, folder for the files and xsections
    outFolder="/t3home/gcelotto/ggHbb/bkgEstimation/output"


    df.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processes.csv")
    
    currentLumi = nFilesData * 0.774 / 1017
    np.save(outFolder+"/currentLumi.npy", currentLumi)
    # choose binning for HT
    bins=np.linspace(0, 800, 20)
    toSkip = [
        'Data',                         
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
    #toSkip = []
    np.save(outFolder+"/binsForHT.npy", bins)
    
    for (process, path, xsection) in zip(df.index, df.path, df.xsection):
        print("Starting process ", process)
        if process in list(toSkip):
            print("skipping process ....", process)
            continue
        

        nFiles=nFilesData if process=='Data' else nFilesMC
        
        fileNames = glob.glob(path+"/**/*.root", recursive=True)
        nFiles = nFiles if nFiles != -1 else len(fileNames)
        if nFiles > len(fileNames) :
            nFiles = len(fileNames)
        
        if process!='Data':
            np.save(outFolder+"/mini_%s.npy"%process, np.float32(0))
        counts = np.zeros(len(bins)-1) # counts of MC sample
        np.save(outFolder+"/counts_%s.npy"%process, counts)

        print("Process : ", process, "     %d files"%nFiles)
        for fileName in fileNames[:nFiles]:
            subprocess.run(['sbatch', '-J', process+"%d"%random.randint(1, 100), '/t3home/gcelotto/ggHbb/bkgEstimation/scripts/process_job.sh', fileName, process])
       
    return 

if __name__ == "__main__":
    nFilesData, nFilesMC = sys.argv[1:]
    nFilesData = int(nFilesData)
    nFilesMC = int(nFilesMC)
    if nFilesData==-1:
        nFilesData=1017

    closure(nFilesData, nFilesMC)