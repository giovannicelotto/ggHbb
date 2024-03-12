import numpy as np
import pandas as pd
import glob, sys
import uproot
import matplotlib.pyplot as plt
import mplhep as hep
import awkward as ak
hep.style.use("CMS")
def closure():
    processes = {
        'Data':                         ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data1A2024Mar05/ParkingBPH1", -1],
        #'QCD_MuEnriched_Pt-15To20':   ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-15To20*", 2800000.0	],
        #'QCD_MuEnriched_Pt-20To30':   ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-20To30*", 2527000.0],
        #'QCD_MuEnriched_Pt-30To50':   ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-30To50*", 1367000.0],
        #'QCD_MuEnriched_Pt-50To80':   ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-50To80*", 381700.0],
        #'QCD_MuEnriched_Pt-80To120':   ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-80To120*", 87740.0],
        #'QCD_MuEnriched_Pt-120To170':   ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-120To170*", 21280.0],
        #'QCD_MuEnriched_Pt-170To300':   ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-170To300*", 7000.0],
        #'QCD_MuEnriched_Pt-300To470':   ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-300To470*", 622.6],
        #'QCD_MuEnriched_Pt-470To600':   ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-470To600*", 58.9],
        #'QCD_MuEnriched_Pt-600To800':   ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-600To800*", 18.12	],
        #'QCD_MuEnriched_Pt-800To1000':   ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-800To1000*", 3.318],
        #'QCD_MuEnriched_Pt-1000':       ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/QCD_MuEnriched2024Feb14/QCD_Pt-1000*", 1.085],
        #'GluGluHToBB':                  ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/GluGluHToBB2024Feb16", 30.52],
        #'TTTo2L2Nu':                 ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ttbar2024Feb14/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8", 831*0.46],
        #'TTToSemiLeptonic':             ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ttbar2024Feb14/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8", 831*0.45],
        #'TTToHadronic':                    ["/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/ttbar2024Feb14/TTToHadronic_TuneCP5_13TeV-powheg-pythia8", 831*0.09],
    }
    df = pd.DataFrame(processes).T
    df.columns = ['path', 'xsection']
    
    fig, ax = plt.subplots(1, 1)
    
    bins=np.linspace(0, 800, 20)
    allCounts = np.zeros(len(bins)-1)
    for (process, path, xsection) in zip(df.index, df.path, df.xsection):
        nFiles=20 if process=='Data' else 1
        fileNames = glob.glob(path+"/**/*.root", recursive=True)
        ht_process = ak.Array([])
        counts_process = np.zeros(len(bins)-1)
        numEventsTotalProcess = 0
        print("Starting process : ", process, "     %d files"%nFiles)
        for fileName in fileNames[:nFiles]:
            print("%d / %d     ...\r"%(fileNames.index(fileName)+1, len(fileNames[:nFiles])) )
            #print("Opening file : ",fileName)
            f = uproot.open(fileName)
            tree = f['Events']
            branches = tree.arrays()
            Jet_pt = branches["Jet_pt"]
            Muon_fired_HLT_Mu9_IP6 = branches["Muon_fired_HLT_Mu9_IP6"]
            muon9_ip6_process = ak.sum(Muon_fired_HLT_Mu9_IP6, axis=-1)
            mask = muon9_ip6_process>0
            ht_process = ak.sum(Jet_pt, axis=-1)
            ht_process = ht_process[mask]

            
            counts = np.histogram(np.clip(ht_process, bins[0], bins[-1]), bins=bins)[0]
            counts_process = counts_process + counts
            #print("Length Jet_pt : ", len(ht_process))
            

            if process!= 'Data':
                lumiBlocks = f['LuminosityBlocks']
                numEventsTotal = np.sum(lumiBlocks.arrays()['GenFilter_numEventsTotal'])
                numEventsTotalProcess += numEventsTotal

        if process=='Data': 
            currentLumi = nFiles * 0.774 / 1017
            
            #print(counts)
            dataCounts = counts
            ax.errorbar((bins[1:]+bins[:-1])/2-0.5, counts_process, xerr=np.diff(bins)/2, marker='o', color='black', linestyle='none')
        else:
            
            #print(numEventsTotal, filterEfficiency)
            #print(lumiBlocks, filterEfficiency)
            
            print("Xsection : %.1f"%xsection)
            print("numEventsTotal : %.1f"%np.sum(numEventsTotalProcess))
            print("Counts : ",counts_process)
            counts_process = counts_process * xsection*1000*currentLumi/np.sum(numEventsTotalProcess)
            print(process, counts_process)
            ax.bar((bins[1:]+bins[:-1])/2-0.5, counts_process , align='center', width=np.diff(bins), label=process, alpha = 1, bottom=allCounts)
            allCounts=np.array(allCounts) + np.array(counts_process)
            
            
    ax.set_xlabel(r"$\mathrm{H_{T}}$")
    ax.set_ylabel("Events")
    ax.legend(bbox_to_anchor=(1,1))
    fig.savefig("/t3home/gcelotto/ggHbb/bkgEstimation/closure.png", bbox_inches='tight')
    print(allCounts)
    print(dataCounts)
    print("MC/Data", allCounts/dataCounts)


    
    return 

if __name__ == "__main__":
    closure()


