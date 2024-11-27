import pandas as pd
def getProcessesDataFrame():
    nanoPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH"
    flatPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB"
    processes = {
        'Data':                             [nanoPathCommon + "/Data1A2024Oct18",                                                                                       flatPathCommon + "/Data1A"    ,                   -1],         # 0              
        'GluGluHToBB':                      [nanoPathCommon + "/GluGluHToBB2024Oct21",                                                                                  flatPathCommon + "/GluGluHToBB"    ,             48.61*0.58],       # 1     
        'EWKZJets':                       [nanoPathCommon + "/EWKZJets2024Oct21",                                                                                     flatPathCommon + "/EWKZJets",                      9.8],       # 2     
        'WW':                               [nanoPathCommon + "/diboson2024Oct18/WW_TuneCP5_13TeV-pythia8",                                                             flatPathCommon + "/diboson/WW",                   75.8],       # 3     
        'WZ':                               [nanoPathCommon + "/diboson2024Oct18/WZ_TuneCP5_13TeV-pythia8",                                                             flatPathCommon + "/diboson/WZ",                   27.6],       # 4     
        'ZZ':                               [nanoPathCommon + "/diboson2024Oct18/ZZ_TuneCP5_13TeV-pythia8",                                                             flatPathCommon + "/diboson/ZZ",                   12.14	],     # 5         
        'ST_s-channel-hadronic':            [nanoPathCommon + "/ST2024Oct18/ST_s-channel_4f_hadronicDecays_TuneCP5_13TeV-amcatnlo-pythia8",                      flatPathCommon + "/singleTop/s-channel_hadronic", 11.24],      # 6     
        'ST_s-channel-leptononic':          [nanoPathCommon + "/ST2024Oct18/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8",                        flatPathCommon + "/singleTop/s-channel_leptonic", 3.74],       # 7 
        'ST_t-channel-antitop':             [nanoPathCommon + "/ST2024Oct18/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",       flatPathCommon + "/singleTop/t-channel_antitop" , 69.09],      # 8     
        'ST_t-channel-top':                 [nanoPathCommon + "/ST2024Oct18/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",           flatPathCommon + "/singleTop/t-channel_top"     , 115.3],      # 9     
        'ST_tW-antitop':                    [nanoPathCommon + "/ST2024Oct18/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",                      flatPathCommon + "/singleTop/tW-channel_antitop", 34.97],      # 10     
        'ST_tW-top':                        [nanoPathCommon + "/ST2024Oct18/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",                          flatPathCommon + "/singleTop/tW-channel_top"    , 34.91	],     # 11        
         #ttbar cross section: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO
         #w boson decay modes : https://pdg.lbl.gov/2023/listings/rpp2023-list-w-boson.pdf
        'TTTo2L2Nu':                        [nanoPathCommon + "/ttbar2024Oct18/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",                                                 flatPathCommon + "/ttbar/ttbar2L2Nu"            , 833.9*(0.1086*3)**2],   # 12        
        'TTToHadronic':                     [nanoPathCommon + "/ttbar2024Oct18/TTToHadronic_TuneCP5_13TeV-powheg-pythia8",                                              flatPathCommon + "/ttbar/ttbarHadronic"         , 833.9*(0.6741)**2],  # 13        
        'TTToSemiLeptonic':                 [nanoPathCommon + "/ttbar2024Oct18/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",                                          flatPathCommon + "/ttbar/ttbarSemiLeptonic"     , 2*833.9*0.6741*0.1086*3],   # 14        
        'WJetsToLNu':                       [nanoPathCommon + "/WJets2024Oct18/WJetsToLNu*",                                                                            flatPathCommon + "/WJets/WJetsToLNu"            , 62070.0],    # 15            
        'WJetsToQQ_200to400'      :         [nanoPathCommon + "/WJets2024Oct18/WJetsToQQ_HT-200to400*",                                                                 flatPathCommon + "/WJets/WJetsToQQ_HT-200to400" , 2549.0],     # 16    
        'WJetsToQQ_400to600'      :         [nanoPathCommon + "/WJets2024Oct18/WJetsToQQ_HT-400to600*",                                                                 flatPathCommon + "/WJets/WJetsToQQ_HT-400to600" , 276.5],      # 17
        'WJetsToQQ_600to800'      :         [nanoPathCommon + "/WJets2024Oct18/WJetsToQQ_HT-600to800*",                                                                 flatPathCommon + "/WJets/WJetsToQQ_HT-600to800" , 59.25],      # 18
        'WJetsToQQ_800toInf'      :         [nanoPathCommon + "/WJets2024Oct18/WJetsToQQ_HT-800toInf*",                                                                 flatPathCommon + "/WJets/WJetsToQQ_HT-800toInf" , 28.75],      # 19
        'ZJetsToQQ_200to400'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-200to400*",                                                                 flatPathCommon + "/ZJets/ZJetsToQQ_HT-200to400" , 1012.0],     # 20    
        'ZJetsToQQ_400to600'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-400to600*",                                                                 flatPathCommon + "/ZJets/ZJetsToQQ_HT-400to600" , 114.2],      # 21
        'ZJetsToQQ_600to800'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-600to800*",                                                                 flatPathCommon + "/ZJets/ZJetsToQQ_HT-600to800" , 25.34],      # 22
        'ZJetsToQQ_800toInf'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-800toInf*",                                                                 flatPathCommon + "/ZJets/ZJetsToQQ_HT-800toInf" , 12.99],      # 23
        'QCD_MuEnriched_Pt-1000':           [nanoPathCommon + "/QCDMuEnriched2024Oct18/QCD_Pt-1000*",                                                                  flatPathCommon + "/QCD_Pt1000ToInf"             , 1.085],      # 24    
        'QCD_MuEnriched_Pt-800To1000':      [nanoPathCommon + "/QCDMuEnriched2024Oct18/QCD_Pt-800To1000*",                                                             flatPathCommon + "/QCD_Pt800To1000"             , 3.318],      # 25    
        'QCD_MuEnriched_Pt-600To800':       [nanoPathCommon + "/QCDMuEnriched2024Oct18/QCD_Pt-600To800*",                                                              flatPathCommon + "/QCD_Pt600To800"              , 18.12	],     # 26        
        'QCD_MuEnriched_Pt-470To600':       [nanoPathCommon + "/QCDMuEnriched2024Oct18/QCD_Pt-470To600*",                                                              flatPathCommon + "/QCD_Pt470To600"              , 58.9],       # 27    
        'QCD_MuEnriched_Pt-300To470':       [nanoPathCommon + "/QCDMuEnriched2024Oct18/QCD_Pt-300To470*",                                                              flatPathCommon + "/QCD_Pt300To470"              , 622.6],      # 28    
        'QCD_MuEnriched_Pt-170To300':       [nanoPathCommon + "/QCDMuEnriched2024Oct18/QCD_Pt-170To300*",                                                              flatPathCommon + "/QCD_Pt170To300"              , 7000.0],     # 29        
        'QCD_MuEnriched_Pt-120To170':       [nanoPathCommon + "/QCDMuEnriched2024Oct18/QCD_Pt-120To170*",                                                              flatPathCommon + "/QCD_Pt120To170"              , 21280.0],    # 30        
        'QCD_MuEnriched_Pt-80To120':        [nanoPathCommon + "/QCDMuEnriched2024Oct18/QCD_Pt-80To120*",                                                               flatPathCommon + "/QCD_Pt80To120"               , 87740.0],    # 31        
        'QCD_MuEnriched_Pt-50To80':         [nanoPathCommon + "/QCDMuEnriched2024Oct18/QCD_Pt-50To80*",                                                                flatPathCommon + "/QCD_Pt50To80"                , 381700.0],   # 32        
        'QCD_MuEnriched_Pt-30To50':         [nanoPathCommon + "/QCDMuEnriched2024Oct18/QCD_Pt-30To50*",                                                                flatPathCommon + "/QCD_Pt30To50"                , 1367000.0],  # 33        
        'QCD_MuEnriched_Pt-20To30':         [nanoPathCommon + "/QCDMuEnriched2024Oct18/QCD_Pt-20To30*",                                                                flatPathCommon + "/QCD_Pt20To30"                , 2527000.0],  # 34        
        'QCD_MuEnriched_Pt-15To20':         [nanoPathCommon + "/QCDMuEnriched2024Oct18/QCD_Pt-15To20*",                                                                flatPathCommon + "/QCD_Pt15To20"                , 2800000.0	], # 35            
        'ZJetsToQQ_100to200'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-100to200",                                                                  flatPathCommon + "/ZJets/ZJetsToQQ_HT-100to200" , 5.261e+03],     # 36    
        'VBFHToBB'      :                   [nanoPathCommon + "/VBFHToBB2024Aug05/",                                                                                    flatPathCommon + "/VBFHToBB" , 3.766*0.58],     # 37    
        'GluGluHToBBMINLO'          :       [nanoPathCommon + "/MINLOGluGluHToBB/",                                                                                     flatPathCommon + "/MINLOGluGluHToBB", 48.61*0.58], # 38
        'Data_2A':                          [nanoPathCommon + "/Data2A2024Oct23",                                                                                       flatPathCommon + "/Data2A"    ,                   -1],         # 39              
        'GluGluH_M50_ToBB':                 [nanoPathCommon  + "/GluGluSpin0_M50",                                                                                      flatPathCommon + "/GluGluH_M50_ToBB", 48.61*0.58],           # 40
        'GluGluH_M70_ToBB':                 [nanoPathCommon  + "/GluGluSpin0_M70",                                                                                      flatPathCommon + "/GluGluH_M70_ToBB", 48.61*0.58],           # 41
        'GluGluH_M100_ToBB':                [nanoPathCommon  + "/GluGluSpin0_M100",                                                                                     flatPathCommon + "/GluGluH_M100_ToBB", 48.61*0.58],        # 42
        'GluGluH_M200_ToBB':                [nanoPathCommon  + "/GluGluSpin0_M200",                                                                                     flatPathCommon + "/GluGluH_M200_ToBB", 48.61*0.58],        # 43
        'GluGluH_M300_ToBB':                [nanoPathCommon  + "/GluGluSpin0_M300",                                                                                     flatPathCommon + "/GluGluH_M300_ToBB", 48.61*0.58],        # 44


        'EWKZJetsBB':                         [nanoPathCommon + "/EWKZJets2024Oct21",                                                                                   flatPathCommon + "/EWKZJetsBB",                      9.8],     # 45     
        'ZJetsToBB_100to200'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-100to200",                                                                  flatPathCommon + "/ZJets/ZJetsToBB_HT-100to200" , 5.261e+03],  # 46    
        'ZJetsToBB_200to400'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-200to400*",                                                                 flatPathCommon + "/ZJets/ZJetsToBB_HT-200to400" , 1012.0],     # 47    
        'ZJetsToBB_400to600'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-400to600*",                                                                 flatPathCommon + "/ZJets/ZJetsToBB_HT-400to600" , 114.2],      # 48
        'ZJetsToBB_600to800'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-600to800*",                                                                 flatPathCommon + "/ZJets/ZJetsToBB_HT-600to800" , 25.34],      # 49
        'ZJetsToBB_800toInf'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-800toInf*",                                                                 flatPathCommon + "/ZJets/ZJetsToBB_HT-800toInf" , 12.99],      # 50
        'EWKZJetsqq':                       [nanoPathCommon + "/EWKZJets2024Oct21",                                                                                     flatPathCommon + "/EWKZJetsqq",                      9.8],       # 51     
        'ZJetsToqq_100to200'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-100to200",                                                                  flatPathCommon + "/ZJets/ZJetsToqq_HT-100to200" , 5.261e+03],  # 52    
        'ZJetsToqq_200to400'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-200to400*",                                                                 flatPathCommon + "/ZJets/ZJetsToqq_HT-200to400" , 1012.0],     # 53    
        'ZJetsToqq_400to600'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-400to600*",                                                                 flatPathCommon + "/ZJets/ZJetsToqq_HT-400to600" , 114.2],      # 54
        'ZJetsToqq_600to800'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-600to800*",                                                                 flatPathCommon + "/ZJets/ZJetsToqq_HT-600to800" , 25.34],      # 55
        'ZJetsToqq_800toInf'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-800toInf*",                                                                 flatPathCommon + "/ZJets/ZJetsToqq_HT-800toInf" , 12.99],      # 56
        

    }

    
    df = pd.DataFrame(processes).T

    df.columns = ['nanoPath', 'flatPath', 'xsection']
    df = df.reset_index()
    df = df.rename(columns={'index': 'process'})
    print(df)
    df.to_csv("/t3home/gcelotto/ggHbb/commonScripts/processes.csv")
    return df
getProcessesDataFrame()