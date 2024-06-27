import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import awkward as ak
import random
import glob
import sys
import mplhep as hep
import subprocess
hep.style.use("CMS")
sys.path.append("/t3home/gcelotto/ggHbb/scripts/plotScripts")
from utilsForPlot import getXSectionBR

def main(nFilesData, nFilesMC):

    df.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processes.csv")

    # lumi for MC normalization
    nFilesData = 1017 if nFilesData==-1 else nFilesData
    currentLumi = nFilesData * 0.774 / 1017
    np.save("/t3home/gcelotto/ggHbb/PU_reweighting/output/currentLumi.npy", currentLumi)
    # bins for PU
    bins = np.arange(100)
    np.save("/t3home/gcelotto/ggHbb/PU_reweighting/output/bins.npy", bins)
    for (process, path, xsection) in zip(df.index, df.path, df.xsection):
        print(process, "...")
        nFiles = nFilesData if process=='Data' else nFilesMC
        fileNames = glob.glob(path+"/**/*.root", recursive=True)
        nFiles = nFiles if nFiles != -1 else len(fileNames)
        if nFiles > len(fileNames) :
            nFiles = len(fileNames)
    
        counts = np.zeros(len(bins)-1)
        np.save("/t3home/gcelotto/ggHbb/PU_reweighting/output/counts_%s.npy"%process, counts)
        
        # save the total numbers of miniAOD for MC only to correctly normalize
        if process!='Data':
            np.save("/t3home/gcelotto/ggHbb/PU_reweighting/output/numEventsTotal_%s.npy"%process,0)
    
        for fileName in fileNames[:nFiles]:
            subprocess.run(['sbatch', '-J', process+"%d"%random.randint(1, 10), '/t3home/gcelotto/ggHbb/PU_reweighting/process_PUjob.sh', fileName, process])

        # note only jobs with different names can run at the same time (see batch script). In this way 4 jobs of the same process can run at the same time
            

    return

if __name__ == "__main__":
    nFilesData, nFilesMC = sys.argv[1:]
    nFilesData, nFilesMC = int(nFilesData), int(nFilesMC)
    main(nFilesData, nFilesMC)