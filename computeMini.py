import uproot
import numpy as np
import pandas as pd
import glob, re
def getProcessesDataFrame():
    nanoPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH"
    flatPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB"
    processes = {
        #'BParkingDataRun20181A':            [nanoPathCommon + "/Data20181A_2023Nov30",                                                                                flatPathCommon + "/Data1A"    ,                   -1],
        'GluGluHToBB':                      [nanoPathCommon + "/GluGluHToBB_20UL18",                                                                                  flatPathCommon + "/GluGluHToBB"    ,      30.52],
        'WW':                               [nanoPathCommon + "/diboson2024Feb14/WW_TuneCP5_13TeV-pythia8",                                                           flatPathCommon + "/diboson/WW",                   75.8],
        'WZ':                               [nanoPathCommon + "/diboson2024Feb14/WZ_TuneCP5_13TeV-pythia8",                                                           flatPathCommon + "/diboson/WZ",                   27.6],
        'ZZ':                               [nanoPathCommon + "/diboson2024Feb14/ZZ_TuneCP5_13TeV-pythia8",                                                           flatPathCommon + "/diboson/ZZ",                   12.14	],
        'ST_s-channel-hadronic':            [nanoPathCommon + "/singleTop2024Feb14/ST_s-channel_4f_hadronicDecays_TuneCP5_13TeV-amcatnlo-pythia8",                    flatPathCommon + "/singleTop/s-channel_hadronic", 11.24],
        'ST_s-channel-leptononic':          [nanoPathCommon + "/singleTop2024Feb14/ST_s-channel_4f_leptonDecays_TuneCP5CR1_13TeV-amcatnlo-pythia8",                   flatPathCommon + "/singleTop/s-channel_leptonic", 3.74],
        'ST_t-channel-antitop':             [nanoPathCommon + "/singleTop2024Feb14/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5CR1_13TeV-powheg-madspin-pythia8",  flatPathCommon + "/singleTop/t-channel_antitop" , 69.09],
        'ST_t-channel-top':                 [nanoPathCommon + "/singleTop2024Feb14/ST_t-channel_top_4f_InclusiveDecays_TuneCP5CR1_13TeV-powheg-madspin-pythia8",      flatPathCommon + "/singleTop/t-channel_top"     , 115.3],
        'ST_tW-antitop':                    [nanoPathCommon + "/singleTop2024Feb14/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",                    flatPathCommon + "/singleTop/tW-channel_antitop", 34.97],
        'ST_tW-top':                        [nanoPathCommon + "/singleTop2024Feb14/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",                        flatPathCommon + "/singleTop/tW-channel_top"    , 34.91	],
        'TTTo2L2Nu':                        [nanoPathCommon + "/ttbar2024Feb14/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",                                               flatPathCommon + "/ttbar/ttbar2L2Nu"            , 831*0.09],
        'TTToHadronic':                     [nanoPathCommon + "/ttbar2024Feb14/TTToHadronic_TuneCP5_13TeV-powheg-pythia8",                                            flatPathCommon + "/ttbar/ttbarHadronic"         , 831*0.46],
        'TTToSemiLeptonic':                 [nanoPathCommon + "/ttbar2024Feb14/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",                                        flatPathCommon + "/ttbar/ttbarSemiLeptonic"     , 831*0.45],
        'WJetsToLNu':                       [nanoPathCommon + "/WJetsToLNu2024Feb20/*",                                                                               flatPathCommon + "/WJets/WJetsToLNu"            , 62070.0],
        'WJetsToQQ_200to400'      :         [nanoPathCommon + "/WJetsToQQ2024Feb20/WJetsToQQ_HT-200to400*",                                                           flatPathCommon + "/WJets/WJetsToQQ_HT-200to400" , 2549.0],
        'WJetsToQQ_400to600'      :         [nanoPathCommon + "/WJetsToQQ2024Feb20/WJetsToQQ_HT-400to600*",                                                           flatPathCommon + "/WJets/WJetsToQQ_HT-400to600" , 276.5],
        'WJetsToQQ_600to800'      :         [nanoPathCommon + "/WJetsToQQ2024Feb20/WJetsToQQ_HT-600to800*",                                                           flatPathCommon + "/WJets/WJetsToQQ_HT-600to800" , 59.25],
        'WJetsToQQ_800toInf'      :         [nanoPathCommon + "/WJetsToQQ2024Feb20/WJetsToQQ_HT-800toInf*",                                                           flatPathCommon + "/WJets/WJetsToQQ_HT-800toInf" , 28.75],
        'ZJetsToQQ_200to400'      :         [nanoPathCommon + "/ZJetsToQQ2024Feb20/ZJetsToQQ_HT-200to400*",                                                           flatPathCommon + "/ZJets/ZJetsToQQ_HT-200to400" , 1012.0],
        'ZJetsToQQ_400to600'      :         [nanoPathCommon + "/ZJetsToQQ2024Feb20/ZJetsToQQ_HT-400to600*",                                                           flatPathCommon + "/ZJets/ZJetsToQQ_HT-400to600" , 114.2],
        'ZJetsToQQ_600to800'      :         [nanoPathCommon + "/ZJetsToQQ2024Feb20/ZJetsToQQ_HT-600to800*",                                                           flatPathCommon + "/ZJets/ZJetsToQQ_HT-600to800" , 25.34],
        'ZJetsToQQ_800toInf'      :         [nanoPathCommon + "/ZJetsToQQ2024Feb20/ZJetsToQQ_HT-800toInf*",                                                           flatPathCommon + "/ZJets/ZJetsToQQ_HT-800toInf" , 12.99],
        'QCD_MuEnriched_Pt-1000':           [nanoPathCommon + "/QCD_MuEnriched2024Feb14/QCD_Pt-1000*",                                                                flatPathCommon + "/QCD_Pt1000ToInf"             , 1.085],
        'QCD_MuEnriched_Pt-800To1000':      [nanoPathCommon + "/QCD_MuEnriched2024Feb14/QCD_Pt-800To1000*",                                                           flatPathCommon + "/QCD_Pt800To1000"             , 3.318],
        'QCD_MuEnriched_Pt-600To800':       [nanoPathCommon + "/QCD_MuEnriched2024Feb14/QCD_Pt-600To800*",                                                            flatPathCommon + "/QCD_Pt600To800"              , 18.12	],
        'QCD_MuEnriched_Pt-470To600':       [nanoPathCommon + "/QCD_MuEnriched2024Feb14/QCD_Pt-470To600*",                                                            flatPathCommon + "/QCD_Pt470To600"              , 58.9],
        'QCD_MuEnriched_Pt-300To470':       [nanoPathCommon + "/QCD_MuEnriched2024Feb14/QCD_Pt-300To470*",                                                            flatPathCommon + "/QCD_Pt300To470"              , 622.6],
        'QCD_MuEnriched_Pt-170To300':       [nanoPathCommon + "/QCD_MuEnriched2024Feb14/QCD_Pt-170To300*",                                                            flatPathCommon + "/QCD_Pt170To300"              , 7000.0],
        'QCD_MuEnriched_Pt-120To170':       [nanoPathCommon + "/QCD_MuEnriched2024Feb14/QCD_Pt-120To170*",                                                            flatPathCommon + "/QCD_Pt120To170"              , 21280.0],
        'QCD_MuEnriched_Pt-80To120':        [nanoPathCommon + "/QCD_MuEnriched2024Feb14/QCD_Pt-80To120*",                                                             flatPathCommon + "/QCD_Pt80To120"               , 87740.0],
        'QCD_MuEnriched_Pt-50To80':         [nanoPathCommon + "/QCD_MuEnriched2024Feb14/QCD_Pt-50To80*",                                                              flatPathCommon + "/QCD_Pt50To80"                , 381700.0],
        'QCD_MuEnriched_Pt-30To50':         [nanoPathCommon + "/QCD_MuEnriched2024Feb14/QCD_Pt-30To50*",                                                              flatPathCommon + "/QCD_Pt30To50"                , 1367000.0],
        'QCD_MuEnriched_Pt-20To30':         [nanoPathCommon + "/QCD_MuEnriched2024Feb14/QCD_Pt-20To30*",                                                              flatPathCommon + "/QCD_Pt20To30"                , 2527000.0],
        'QCD_MuEnriched_Pt-15To20':         [nanoPathCommon + "/QCD_MuEnriched2024Feb14/QCD_Pt-15To20*",                                                              flatPathCommon + "/QCD_Pt15To20"                , 2800000.0	],

    }

    df = pd.DataFrame(processes).T
    df.columns = ['nanoPath', 'flatPath', 'xsection']
    return df
def computeMini():
    df = getProcessesDataFrame()

    miniDf = {'process' :   [],
              'fileNumber': [],
              'numEventsPassed':       []}
    for (process, nanoPath, xsection) in zip(df.index, df.nanoPath, df.xsection):
        nanoFileNames = glob.glob(nanoPath+"/**/*.root", recursive=True)
        print("Searching for", nanoPath+"/**/*.root")

        for nanoFileName in nanoFileNames:
            fileNumber = re.search(r'\D(\d{1,4})\.\w+$', nanoFileName).group(1)
            f = uproot.open(nanoFileName)
            lumiBlocks = f['LuminosityBlocks']
            numEventsPassed = np.sum(lumiBlocks.arrays()['GenFilter_numEventsPassed'])
            miniDf['process'].append(process)
            miniDf['fileNumber'].append(fileNumber)
            miniDf['numEventsPassed'].append(numEventsPassed)
    miniDf = pd.DataFrame(miniDf)
    miniDf.to_csv("/t3home/gcelotto/ggHbb/abcd/output/miniDf.csv")




    return

if __name__ == "__main__":
    computeMini()