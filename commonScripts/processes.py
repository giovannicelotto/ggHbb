import pandas as pd
def getProcessesDataFrame():
    nanoPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH"
    flatPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB"
    processes = {
        'Data':                             [nanoPathCommon + "/Data1A2024Mar05",                                                                                       flatPathCommon + "/Data1A"    ,                   -1],         # 0              
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
        # ttbar cross section: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO
        # w boson decay modes : https://pdg.lbl.gov/2023/listings/rpp2023-list-w-boson.pdf
        'TTTo2L2Nu':                        [nanoPathCommon + "/ttbar2024Apr01/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",                                                 flatPathCommon + "/ttbar/ttbar2L2Nu"            , 833.9*(0.1086*3)**2],   # 12        
        'TTToHadronic':                     [nanoPathCommon + "/ttbar2024Apr01/TTToHadronic_TuneCP5_13TeV-powheg-pythia8",                                              flatPathCommon + "/ttbar/ttbarHadronic"         , 833.9*(0.6741)**2],  # 13        
        'TTToSemiLeptonic':                 [nanoPathCommon + "/TTToSemiLeptonic2024Apr11/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/crab_TTToSemiLeptonic/240411_121925",                                          flatPathCommon + "/ttbar/ttbarSemiLeptonic"     , 2*833.9*0.6741*0.1086*3],   # 14        
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
        'ZJetsToQQ_100to200'      :         [nanoPathCommon + "/ZJets2024Apr01/ZJetsToQQ_HT-100to200",                                                                  flatPathCommon + "/ZJets/ZJetsToQQ_HT-100to200" , 5.261e+03],     # 36    

    }

    
    df = pd.DataFrame(processes).T

    df.columns = ['nanoPath', 'flatPath', 'xsection']
    df = df.reset_index()
    df = df.rename(columns={'index': 'process'})
    print(df)
    df.to_csv("/t3home/gcelotto/ggHbb/commonScripts/processes.csv")
    return df
getProcessesDataFrame()