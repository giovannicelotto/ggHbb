import uproot
import numpy as np
import pandas as pd
import glob, re, sys
#def getProcessesDataFrame():
#    nanoPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH"
#    flatPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB"
#    processes = {
#        #'BParkingDataRun20181A':            [nanoPathCommon + "/Data1A2024Mar05",                                                                                flatPathCommon + "/Data1A"    ,                   -1],
#        'GluGluHToBB':                      [nanoPathCommon + "/GluGluHToBB2024Mar05",                                                                                  flatPathCommon + "/GluGluHToBB"    ,      30.52],
#        'WW':                               [nanoPathCommon + "/diboson2024Apr01/WW_TuneCP5_13TeV-pythia8",                                                           flatPathCommon + "/diboson/WW",                   75.8],
#        'WZ':                               [nanoPathCommon + "/diboson2024Apr01/WZ_TuneCP5_13TeV-pythia8",                                                           flatPathCommon + "/diboson/WZ",                   27.6],
#        'ZZ':                               [nanoPathCommon + "/diboson2024Apr01/ZZ_TuneCP5_13TeV-pythia8",                                                           flatPathCommon + "/diboson/ZZ",                   12.14	],
#        'ST_s-channel-hadronic':            [nanoPathCommon + "/singleTop2024Apr01/ST_s-channel_4f_hadronicDecays_TuneCP5_13TeV-amcatnlo-pythia8",                    flatPathCommon + "/singleTop/s-channel_hadronic", 11.24],
#        'ST_s-channel-leptononic':          [nanoPathCommon + "/singleTop2024Apr01/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8",                   flatPathCommon + "/singleTop/s-channel_leptonic", 3.74],
#        'ST_t-channel-antitop':             [nanoPathCommon + "/singleTop2024Apr01/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",  flatPathCommon + "/singleTop/t-channel_antitop" , 69.09],
#        'ST_t-channel-top':                 [nanoPathCommon + "/singleTop2024Apr01/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",      flatPathCommon + "/singleTop/t-channel_top"     , 115.3],
#        'ST_tW-antitop':                    [nanoPathCommon + "/singleTop2024Apr01/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",                    flatPathCommon + "/singleTop/tW-channel_antitop", 34.97],
#        'ST_tW-top':                        [nanoPathCommon + "/singleTop2024Apr01/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",                        flatPathCommon + "/singleTop/tW-channel_top"    , 34.91	],
#        'TTTo2L2Nu':                        [nanoPathCommon + "/ttbar2024Apr01/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",                                               flatPathCommon + "/ttbar/ttbar2L2Nu"            , 831*0.09],
#        'TTToHadronic':                     [nanoPathCommon + "/ttbar2024Apr01/TTToHadronic_TuneCP5_13TeV-powheg-pythia8",                                            flatPathCommon + "/ttbar/ttbarHadronic"         , 831*0.46],
#        'TTToSemiLeptonic':                 [nanoPathCommon + "/TTToSemiLeptonic2024Apr11/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/crab_TTToSemiLeptonic/240411_121925",                                        flatPathCommon + "/ttbar/ttbarSemiLeptonic"     , 831*0.45],
#        'WJetsToLNu':                       [nanoPathCommon + "/WJets2024Apr01/WJetsToLNu*/*",                                                                               flatPathCommon + "/WJets/WJetsToLNu"            , 62070.0],
#        'WJetsToQQ_200to400'      :         [nanoPathCommon + "/WJets2024Apr01/WJetsToQQ_HT-200to400*",                                                           flatPathCommon + "/WJets/WJetsToQQ_HT-200to400" , 2549.0],
#        'WJetsToQQ_400to600'      :         [nanoPathCommon + "/WJets2024Apr01/WJetsToQQ_HT-400to600*",                                                           flatPathCommon + "/WJets/WJetsToQQ_HT-400to600" , 276.5],
#        'WJetsToQQ_600to800'      :         [nanoPathCommon + "/WJets2024Apr01/WJetsToQQ_HT-600to800*",                                                           flatPathCommon + "/WJets/WJetsToQQ_HT-600to800" , 59.25],
#        'WJetsToQQ_800toInf'      :         [nanoPathCommon + "/WJets2024Apr01/WJetsToQQ_HT-800toInf*",                                                           flatPathCommon + "/WJets/WJetsToQQ_HT-800toInf" , 28.75],
#        'ZJetsToQQ_100to200'      :         [nanoPathCommon + "/ZJets2024Apr01/ZJetsToQQ_HT-100to200*",                                                           flatPathCommon + "/ZJets/ZJetsToQQ_HT-100to200" , 5.261e+03],
#        'ZJetsToQQ_200to400'      :         [nanoPathCommon + "/ZJets2024Apr01/ZJetsToQQ_HT-200to400*",                                                           flatPathCommon + "/ZJets/ZJetsToQQ_HT-200to400" , 1012.0],
#        'ZJetsToQQ_400to600'      :         [nanoPathCommon + "/ZJets2024Apr01/ZJetsToQQ_HT-400to600*",                                                           flatPathCommon + "/ZJets/ZJetsToQQ_HT-400to600" , 114.2],
#        'ZJetsToQQ_600to800'      :         [nanoPathCommon + "/ZJets2024Apr01/ZJetsToQQ_HT-600to800*",                                                           flatPathCommon + "/ZJets/ZJetsToQQ_HT-600to800" , 25.34],
#        'ZJetsToQQ_800toInf'      :         [nanoPathCommon + "/ZJets2024Apr01/ZJetsToQQ_HT-800toInf*",                                                           flatPathCommon + "/ZJets/ZJetsToQQ_HT-800toInf" , 12.99],
#        'EWKZJets'                :         [nanoPathCommon + "/EWKZJets2024Mar15",                                                                                   flatPathCommon + "/EWKZJets" , 9.8],
#        'QCD_MuEnriched_Pt-1000':           [nanoPathCommon + "/QCD_MuEnriched2024Apr01/QCD_Pt-1000*",                                                                flatPathCommon + "/QCD_Pt1000ToInf"             , 1.085],
#        'QCD_MuEnriched_Pt-800To1000':      [nanoPathCommon + "/QCD_MuEnriched2024Apr01/QCD_Pt-800To1000*",                                                           flatPathCommon + "/QCD_Pt800To1000"             , 3.318],
#        'QCD_MuEnriched_Pt-600To800':       [nanoPathCommon + "/QCD_MuEnriched2024Apr01/QCD_Pt-600To800*",                                                            flatPathCommon + "/QCD_Pt600To800"              , 18.12	],
#        'QCD_MuEnriched_Pt-470To600':       [nanoPathCommon + "/QCD_MuEnriched2024Apr01/QCD_Pt-470To600*",                                                            flatPathCommon + "/QCD_Pt470To600"              , 58.9],
#        'QCD_MuEnriched_Pt-300To470':       [nanoPathCommon + "/QCD_MuEnriched2024Apr01/QCD_Pt-300To470*",                                                            flatPathCommon + "/QCD_Pt300To470"              , 622.6],
#        'QCD_MuEnriched_Pt-170To300':       [nanoPathCommon + "/QCD_MuEnriched2024Apr01/QCD_Pt-170To300*",                                                            flatPathCommon + "/QCD_Pt170To300"              , 7000.0],
#        'QCD_MuEnriched_Pt-120To170':       [nanoPathCommon + "/QCD_MuEnriched2024Apr01/QCD_Pt-120To170*",                                                            flatPathCommon + "/QCD_Pt120To170"              , 21280.0],
#        'QCD_MuEnriched_Pt-80To120':        [nanoPathCommon + "/QCD_MuEnriched2024Apr01/QCD_Pt-80To120*",                                                             flatPathCommon + "/QCD_Pt80To120"               , 87740.0],
#        'QCD_MuEnriched_Pt-50To80':         [nanoPathCommon + "/QCD_MuEnriched2024Apr01/QCD_Pt-50To80*",                                                              flatPathCommon + "/QCD_Pt50To80"                , 381700.0],
#        'QCD_MuEnriched_Pt-30To50':         [nanoPathCommon + "/QCD_MuEnriched2024Apr01/QCD_Pt-30To50*",                                                              flatPathCommon + "/QCD_Pt30To50"                , 1367000.0],
#        'QCD_MuEnriched_Pt-20To30':         [nanoPathCommon + "/QCD_MuEnriched2024Apr01/QCD_Pt-20To30*",                                                              flatPathCommon + "/QCD_Pt20To30"                , 2527000.0],
#        'QCD_MuEnriched_Pt-15To20':         [nanoPathCommon + "/QCD_MuEnriched2024Apr01/QCD_Pt-15To20*",                                                              flatPathCommon + "/QCD_Pt15To20"                , 2800000.0],
#        'VBFHToBB'      :                   [nanoPathCommon + "/VBFHToBB2024Aug05/",                                                                                    flatPathCommon + "/VBFHToBB" , 3.766*0.58],     # 37    
#        'MINLOGluGluHToBB'      :                   [nanoPathCommon + "/MINLOGluGluHToBB/",                                                                                    flatPathCommon + "/MINLOGluGluHToBB" , 48.61*0.58],     # 37    
#
#    }
#
#    df = pd.DataFrame(processes).T
#    df.columns = ['nanoPath', 'flatPath', 'xsection']
#    return df
def computeMini():
    df = pd.read_csv("/t3home/gcelotto/ggHbb/commonScripts/processes.csv")

    miniDf = {'process' :   [],
              'fileNumber': [],
              'numEventsPassed':       []}
    for (process, nanoPath, xsection) in zip(df.index, df.nanoPath, df.xsection):
        nanoFileNames = glob.glob(nanoPath+"/**/*.root", recursive=True)
        print("Searching for", nanoPath+"/**/*.root ... %d files found"%len(nanoFileNames))

        for nanoFileName in nanoFileNames:
            try:
                fileNumber = re.search(r'\D(\d{1,4})\.\w+$', nanoFileName).group(1)
            except:
                sys.exit(1)
                #print(fileName)
                #print("filenumber not found in ", nanoFileName)
                #try:
                    #
                    #print("This is ZJets100To200")
                #except:
                #sys.exit()
            f = uproot.open(nanoFileName)
            lumiBlocks = f['LuminosityBlocks']
            numEventsPassed = np.sum(lumiBlocks.arrays()['GenFilter_numEventsPassed'])
            miniDf['process'].append(process)
            miniDf['fileNumber'].append(fileNumber)
            miniDf['numEventsPassed'].append(numEventsPassed)
        miniPandasDf = pd.DataFrame(miniDf)
        miniPandasDf.to_csv("/t3home/gcelotto/ggHbb/outputs/counters/miniDf_Sep.csv")
    miniDf = pd.DataFrame(miniDf)
    miniDf.to_csv("/t3home/gcelotto/ggHbb/outputs/counters/miniDf_Sep.csv")




    return

if __name__ == "__main__":
    computeMini()