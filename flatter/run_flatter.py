import pandas as pd
import glob, sys, re, os
import random
import subprocess
import time


def getProcessesDataFrame():
    nanoPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH"
    flatPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB"
    processes = {
        'BParkingDataRun20181A':            [nanoPathCommon + "/Data1A2024Mar05",                                                                                       flatPathCommon + "/Data1A"    ,                   -1],         # 0              
        'GluGluHToBB':                      [nanoPathCommon + "/GluGluHToBB2024Mar05",                                                                                  flatPathCommon + "/GluGluHToBB"    ,             30.52],       # 1     
        'EWKZJets':                         [nanoPathCommon + "/EWKZJets2024Mar15",                                                                                     flatPathCommon + "/EWKZJets",                      9.8],       # 2     
        'WW':                               [nanoPathCommon + "/diboson2024Apr01/WW_TuneCP5_13TeV-pythia8",                                                             flatPathCommon + "/diboson/WW",                   75.8],       # 3     
        'WZ':                               [nanoPathCommon + "/diboson2024Apr01/WZ_TuneCP5_13TeV-pythia8",                                                             flatPathCommon + "/diboson/WZ",                   27.6],       # 4     
        'ZZ':                               [nanoPathCommon + "/diboson2024Apr01/ZZ_TuneCP5_13TeV-pythia8",                                                             flatPathCommon + "/diboson/ZZ",                   12.14	],     # 5         
        'ST_s-channel-hadronic':            [nanoPathCommon + "/singleTop2024Apr01/ST_s-channel_4f_hadronicDecays_TuneCP5_13TeV-amcatnlo-pythia8",                      flatPathCommon + "/singleTop/s-channel_hadronic", 11.24],      # 6     
        'ST_s-channel-leptononic':          [nanoPathCommon + "/singleTop2024Apr01/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8",                        flatPathCommon + "/singleTop/s-channel_leptonic", 3.74],       # 7 
        'ST_t-channel-antitop':             [nanoPathCommon + "/singleTop2024Apr01/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",       flatPathCommon + "/singleTop/t-channel_antitop" , 69.09],      # 8     
        'ST_t-channel-top':                 [nanoPathCommon + "/singleTop2024Apr01/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",           flatPathCommon + "/singleTop/t-channel_top"     , 115.3],      # 9     
        'ST_tW-antitop':                    [nanoPathCommon + "/singleTop2024Apr01/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",                      flatPathCommon + "/singleTop/tW-channel_antitop", 34.97],      # 10     
        'ST_tW-top':                        [nanoPathCommon + "/singleTop2024Apr01/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",                          flatPathCommon + "/singleTop/tW-channel_top"    , 34.91	],     # 11        
        'TTTo2L2Nu':                        [nanoPathCommon + "/ttbar2024Apr01/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",                                                 flatPathCommon + "/ttbar/ttbar2L2Nu"            , 831*0.09],   # 12        
        'TTToHadronic':                     [nanoPathCommon + "/ttbar2024Apr01/TTToHadronic_TuneCP5_13TeV-powheg-pythia8",                                              flatPathCommon + "/ttbar/ttbarHadronic"         , 6.871e+02],  # 13        
        'TTToSemiLeptonic':                 [nanoPathCommon + "/ttbar2024Apr01/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",                                          flatPathCommon + "/ttbar/ttbarSemiLeptonic"     , 831*0.45],   # 14        
        'WJetsToLNu':                       [nanoPathCommon + "/WJets2024Apr01/WJetsToLNu*",                                                                            flatPathCommon + "/WJets/WJetsToLNu"            , 62070.0],    # 15            
        'WJetsToQQ_200to400'      :         [nanoPathCommon + "/WJets2024Apr01/WJetsToQQ_HT-200to400*",                                                                 flatPathCommon + "/WJets/WJetsToQQ_HT-200to400" , 2549.0],     # 16    
        'WJetsToQQ_400to600'      :         [nanoPathCommon + "/WJets2024Apr01/WJetsToQQ_HT-400to600*",                                                                 flatPathCommon + "/WJets/WJetsToQQ_HT-400to600" , 276.5],      # 17
        'WJetsToQQ_600to800'      :         [nanoPathCommon + "/WJets2024Apr01/WJetsToQQ_HT-600to800*",                                                                 flatPathCommon + "/WJets/WJetsToQQ_HT-600to800" , 59.25],      # 18
        'WJetsToQQ_800toInf'      :         [nanoPathCommon + "/WJets2024Apr01/WJetsToQQ_HT-800toInf*",                                                                 flatPathCommon + "/WJets/WJetsToQQ_HT-800toInf" , 28.75],      # 19
        'ZJetsToQQ_200to400'      :         [nanoPathCommon + "/ZJets2024Apr01/ZJetsToQQ_HT-200to400*",                                                                 flatPathCommon + "/ZJets/ZJetsToQQ_HT-200to400" , 1012.0],     # 20    
        'ZJetsToQQ_400to600'      :         [nanoPathCommon + "/ZJets2024Apr01/ZJetsToQQ_HT-400to600*",                                                                 flatPathCommon + "/ZJets/ZJetsToQQ_HT-400to600" , 114.2],      # 21
        'ZJetsToQQ_600to800'      :         [nanoPathCommon + "/ZJets2024Apr01/ZJetsToQQ_HT-600to800*",                                                                 flatPathCommon + "/ZJets/ZJetsToQQ_HT-600to800" , 25.34],      # 22
        'ZJetsToQQ_800toInf'      :         [nanoPathCommon + "/ZJets2024Apr01/ZJetsToQQ_HT-800toInf*",                                                                 flatPathCommon + "/ZJets/ZJetsToQQ_HT-800toInf" , 12.99],      # 23
        'QCD_MuEnriched_Pt-1000':           [nanoPathCommon + "/QCD_MuEnriched2024Apr01/QCD_Pt-1000*",                                                                  flatPathCommon + "/QCD_Pt1000ToInf"             , 1.085],      # 24    
        'QCD_MuEnriched_Pt-800To1000':      [nanoPathCommon + "/QCD_MuEnriched2024Apr01/QCD_Pt-800To1000*",                                                             flatPathCommon + "/QCD_Pt800To1000"             , 3.318],      # 25    
        'QCD_MuEnriched_Pt-600To800':       [nanoPathCommon + "/QCD_MuEnriched2024Apr01/QCD_Pt-600To800*",                                                              flatPathCommon + "/QCD_Pt600To800"              , 18.12	],     # 26        
        'QCD_MuEnriched_Pt-470To600':       [nanoPathCommon + "/QCD_MuEnriched2024Apr01/QCD_Pt-470To600*",                                                              flatPathCommon + "/QCD_Pt470To600"              , 58.9],       # 27    
        'QCD_MuEnriched_Pt-300To470':       [nanoPathCommon + "/QCD_MuEnriched2024Apr01/QCD_Pt-300To470*",                                                              flatPathCommon + "/QCD_Pt300To470"              , 622.6],      # 28    
        'QCD_MuEnriched_Pt-170To300':       [nanoPathCommon + "/QCD_MuEnriched2024Apr01/QCD_Pt-170To300*",                                                              flatPathCommon + "/QCD_Pt170To300"              , 7000.0],     # 29        
        'QCD_MuEnriched_Pt-120To170':       [nanoPathCommon + "/QCD_MuEnriched2024Apr01/QCD_Pt-120To170*",                                                              flatPathCommon + "/QCD_Pt120To170"              , 21280.0],    # 30        
        'QCD_MuEnriched_Pt-80To120':        [nanoPathCommon + "/QCD_MuEnriched2024Apr01/QCD_Pt-80To120*",                                                               flatPathCommon + "/QCD_Pt80To120"               , 87740.0],    # 31        
        'QCD_MuEnriched_Pt-50To80':         [nanoPathCommon + "/QCD_MuEnriched2024Apr01/QCD_Pt-50To80*",                                                                flatPathCommon + "/QCD_Pt50To80"                , 381700.0],   # 32        
        'QCD_MuEnriched_Pt-30To50':         [nanoPathCommon + "/QCD_MuEnriched2024Apr01/QCD_Pt-30To50*",                                                                flatPathCommon + "/QCD_Pt30To50"                , 1367000.0],  # 33        
        'QCD_MuEnriched_Pt-20To30':         [nanoPathCommon + "/QCD_MuEnriched2024Apr01/QCD_Pt-20To30*",                                                                flatPathCommon + "/QCD_Pt20To30"                , 2527000.0],  # 34        
        'QCD_MuEnriched_Pt-15To20':         [nanoPathCommon + "/QCD_MuEnriched2024Apr01/QCD_Pt-15To20*",                                                                flatPathCommon + "/QCD_Pt15To20"                , 2800000.0	], # 35            

    }

    
    df = pd.DataFrame(processes).T
    df.columns = ['nanoPath', 'flatPath', 'xsection']
    return df

def main(isMC, nFiles, maxEntries, maxJet):
    # Define name of the process, folder for the files and xsections
    df = getProcessesDataFrame()
    df.to_csv("/t3home/gcelotto/ggHbb/flatter/output/processesPath.csv")
    nanoPath = list(df.nanoPath)[isMC]
    flatPath = list(df.flatPath)[isMC]
    if not os.path.exists(flatPath):
        os.makedirs(flatPath)
    process = list(df.index)[isMC]
    nanoFileNames = glob.glob(nanoPath+"/**/*.root", recursive=True)
    flatFileNames = glob.glob(flatPath+"/**/*.parquet", recursive=True)
    print(process, len(flatFileNames), "/", len(nanoFileNames))
    if len(flatFileNames)==len(nanoFileNames):
        return
        
    time.sleep(3)
    #sys.exit("exit")
    
    nFiles = nFiles if nFiles != -1 else len(nanoFileNames)
    if nFiles > len(nanoFileNames) :
        nFiles = len(nanoFileNames)
    #nFiles to be done
    doneFiles = 0
    for nanoFileName in nanoFileNames:
        if doneFiles==nFiles:
            break
        fileNumber = re.search(r'\D(\d{1,4})\.\w+$', nanoFileName).group(1)
        filePattern = flatPath+"/**/"+process+"_"+fileNumber+".parquet"
        matching_files = glob.glob(filePattern, recursive=True)
        #print("Checking for ", flatPath+"/**/"+process+"_"+fileNumber+".parquet")

        if matching_files:
            #print(process+"_"+fileNumber+".parquet present. Skipped")
            continue
        print(process, fileNumber)
        print(flatPath)
        subprocess.run(['sbatch', '-J', process+"%d"%random.randint(1, 20), '/t3home/gcelotto/ggHbb/flatter/job.sh', nanoFileName, process, fileNumber, flatPath])
        doneFiles = doneFiles+1
    return 

if __name__ == "__main__":
    isMC        = int(sys.argv[1]) if len(sys.argv) > 1 else 1 # 0 = Data , 1 = ggHbb, 2 = ZJets
    nFiles      = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    # optional
    maxEntries  = int(sys.argv[3]) if len(sys.argv) > 3 else -1
    maxJet      = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    main(isMC, nFiles, maxEntries, maxJet)