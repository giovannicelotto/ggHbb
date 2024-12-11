import pandas as pd
def getProcessesDataFrame():
    nanoPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH"
    flatPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB"
    processesMC = {
        'GluGluHToBB':                      [nanoPathCommon + "/GluGluHToBB2024Oct21",                                                                                  flatPathCommon + "/GluGluHToBB"    ,                48.61*0.58],                # 0     
        'EWKZJets':                         [nanoPathCommon + "/EWKZJets2024Oct21",                                                                                     flatPathCommon + "/EWKZJets",                       9.8],                       # 1     
        'WW':                               [nanoPathCommon + "/diboson2024Oct18/WW_TuneCP5_13TeV-pythia8",                                                             flatPathCommon + "/diboson/WW",                     75.8],                      # 2     
        'WZ':                               [nanoPathCommon + "/diboson2024Oct18/WZ_TuneCP5_13TeV-pythia8",                                                             flatPathCommon + "/diboson/WZ",                     27.6],                      # 3     
        'ZZ':                               [nanoPathCommon + "/diboson2024Oct18/ZZ_TuneCP5_13TeV-pythia8",                                                             flatPathCommon + "/diboson/ZZ",                     12.14	],                  # 4         
        'ST_s-channel-hadronic':            [nanoPathCommon + "/ST2024Oct18/ST_s-channel_4f_hadronicDecays_TuneCP5_13TeV-amcatnlo-pythia8",                      flatPathCommon + "/singleTop/s-channel_hadronic",          11.24],                     # 5     
        'ST_s-channel-leptononic':          [nanoPathCommon + "/ST2024Oct18/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8",                        flatPathCommon + "/singleTop/s-channel_leptonic",          3.74],                      # 6 
        'ST_t-channel-antitop':             [nanoPathCommon + "/ST2024Oct18/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",       flatPathCommon + "/singleTop/t-channel_antitop" ,          69.09],                     # 7     
        'ST_t-channel-top':                 [nanoPathCommon + "/ST2024Oct18/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",           flatPathCommon + "/singleTop/t-channel_top"     ,          115.3],                     # 8     
        'ST_tW-antitop':                    [nanoPathCommon + "/ST2024Oct18/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",                      flatPathCommon + "/singleTop/tW-channel_antitop",          34.97],                     # 9     
        'ST_tW-top':                        [nanoPathCommon + "/ST2024Oct18/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",                          flatPathCommon + "/singleTop/tW-channel_top"    ,          34.91	],                  # 10        
         #ttbar cross section: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO
         #w boson decay modes : https://pdg.lbl.gov/2023/listings/rpp2023-list-w-boson.pdf
        'TTTo2L2Nu':                        [nanoPathCommon + "/ttbar2024Oct18/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",                                                 flatPathCommon + "/ttbar/ttbar2L2Nu"            ,   833.9*(0.1086*3)**2],       # 11        
        'TTToHadronic':                     [nanoPathCommon + "/ttbar2024Oct18/TTToHadronic_TuneCP5_13TeV-powheg-pythia8",                                              flatPathCommon + "/ttbar/ttbarHadronic"         ,   833.9*(0.6741)**2],         # 12        
        'TTToSemiLeptonic':                 [nanoPathCommon + "/ttbar2024Oct18/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",                                          flatPathCommon + "/ttbar/ttbarSemiLeptonic"     ,   2*833.9*0.6741*0.1086*3],   # 13        
        'WJetsToLNu':                       [nanoPathCommon + "/WJets2024Oct18/WJetsToLNu*",                                                                            flatPathCommon + "/WJets/WJetsToLNu"            ,   62070.0],                   # 14            
        'WJetsToQQ_200to400'      :         [nanoPathCommon + "/WJets2024Oct18/WJetsToQQ_HT-200to400*",                                                                 flatPathCommon + "/WJets/WJetsToQQ_HT-200to400" ,   2549.0],                    # 15    
        'WJetsToQQ_400to600'      :         [nanoPathCommon + "/WJets2024Oct18/WJetsToQQ_HT-400to600*",                                                                 flatPathCommon + "/WJets/WJetsToQQ_HT-400to600" ,   276.5],                     # 16
        'WJetsToQQ_600to800'      :         [nanoPathCommon + "/WJets2024Oct18/WJetsToQQ_HT-600to800*",                                                                 flatPathCommon + "/WJets/WJetsToQQ_HT-600to800" ,   59.25],                     # 17
        'WJetsToQQ_800toInf'      :         [nanoPathCommon + "/WJets2024Oct18/WJetsToQQ_HT-800toInf*",                                                                 flatPathCommon + "/WJets/WJetsToQQ_HT-800toInf" ,   28.75],                     # 18
        'ZJetsToQQ_200to400'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-200to400*",                                                                 flatPathCommon + "/ZJets/ZJetsToQQ_HT-200to400" ,   1012.0],                    # 19    
        'ZJetsToQQ_400to600'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-400to600*",                                                                 flatPathCommon + "/ZJets/ZJetsToQQ_HT-400to600" ,   114.2],                     # 20
        'ZJetsToQQ_600to800'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-600to800*",                                                                 flatPathCommon + "/ZJets/ZJetsToQQ_HT-600to800" ,   25.34],                     # 21
        'ZJetsToQQ_800toInf'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-800toInf*",                                                                 flatPathCommon + "/ZJets/ZJetsToQQ_HT-800toInf" ,   12.99],                     # 22
        'QCD_MuEnriched_Pt-1000':           [nanoPathCommon + "/QCDMuEnriched2024Oct18/QCD_Pt-1000*",                                                                  flatPathCommon + "/QCD_Pt1000ToInf"             ,    1.085],                     # 23    
        'QCD_MuEnriched_Pt-800To1000':      [nanoPathCommon + "/QCDMuEnriched2024Oct18/QCD_Pt-800To1000*",                                                             flatPathCommon + "/QCD_Pt800To1000"             ,    3.318],                     # 24    
        'QCD_MuEnriched_Pt-600To800':       [nanoPathCommon + "/QCDMuEnriched2024Oct18/QCD_Pt-600To800*",                                                              flatPathCommon + "/QCD_Pt600To800"              ,    18.12	],                  # 25        
        'QCD_MuEnriched_Pt-470To600':       [nanoPathCommon + "/QCDMuEnriched2024Oct18/QCD_Pt-470To600*",                                                              flatPathCommon + "/QCD_Pt470To600"              ,    58.9],                      # 26    
        'QCD_MuEnriched_Pt-300To470':       [nanoPathCommon + "/QCDMuEnriched2024Oct18/QCD_Pt-300To470*",                                                              flatPathCommon + "/QCD_Pt300To470"              ,    622.6],                     # 27    
        'QCD_MuEnriched_Pt-170To300':       [nanoPathCommon + "/QCDMuEnriched2024Oct18/QCD_Pt-170To300*",                                                              flatPathCommon + "/QCD_Pt170To300"              ,    7000.0],                    # 28        
        'QCD_MuEnriched_Pt-120To170':       [nanoPathCommon + "/QCDMuEnriched2024Oct18/QCD_Pt-120To170*",                                                              flatPathCommon + "/QCD_Pt120To170"              ,    21280.0],                   # 29        
        'QCD_MuEnriched_Pt-80To120':        [nanoPathCommon + "/QCDMuEnriched2024Oct18/QCD_Pt-80To120*",                                                               flatPathCommon + "/QCD_Pt80To120"               ,    87740.0],                   # 30        
        'QCD_MuEnriched_Pt-50To80':         [nanoPathCommon + "/QCDMuEnriched2024Oct18/QCD_Pt-50To80*",                                                                flatPathCommon + "/QCD_Pt50To80"                ,    381700.0],                  # 31        
        'QCD_MuEnriched_Pt-30To50':         [nanoPathCommon + "/QCDMuEnriched2024Oct18/QCD_Pt-30To50*",                                                                flatPathCommon + "/QCD_Pt30To50"                ,    1367000.0],                 # 32        
        'QCD_MuEnriched_Pt-20To30':         [nanoPathCommon + "/QCDMuEnriched2024Oct18/QCD_Pt-20To30*",                                                                flatPathCommon + "/QCD_Pt20To30"                ,    2527000.0],                 # 33        
        'QCD_MuEnriched_Pt-15To20':         [nanoPathCommon + "/QCDMuEnriched2024Oct18/QCD_Pt-15To20*",                                                                flatPathCommon + "/QCD_Pt15To20"                ,    2800000.0	],              # 34            
        'ZJetsToQQ_100to200'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-100to200",                                                                  flatPathCommon + "/ZJets/ZJetsToQQ_HT-100to200" ,   5.261e+03],                 # 35    
        'VBFHToBB'      :                   [nanoPathCommon + "/VBFHToBB2024Aug05/",                                                                                    flatPathCommon + "/VBFHToBB" ,                      3.766*0.58],                # 36    
        'GluGluHToBBMINLO'          :       [nanoPathCommon + "/MINLOGluGluHToBB/",                                                                                     flatPathCommon + "/MINLOGluGluHToBB",               48.61*0.58],                # 37
        'GluGluH_M50_ToBB':                 [nanoPathCommon  + "/GluGluSpin0_M50",                                                                                      flatPathCommon + "/GluGluH_M50_ToBB",               48.61*0.58],                # 38
        'GluGluH_M70_ToBB':                 [nanoPathCommon  + "/GluGluSpin0_M70",                                                                                      flatPathCommon + "/GluGluH_M70_ToBB",               48.61*0.58],                # 40
        'GluGluH_M100_ToBB':                [nanoPathCommon  + "/GluGluSpin0_M100",                                                                                     flatPathCommon + "/GluGluH_M100_ToBB",              48.61*0.58],                # 41
        'GluGluH_M200_ToBB':                [nanoPathCommon  + "/GluGluSpin0_M200",                                                                                     flatPathCommon + "/GluGluH_M200_ToBB",              48.61*0.58],                # 42
        'GluGluH_M300_ToBB':                [nanoPathCommon  + "/GluGluSpin0_M300",                                                                                     flatPathCommon + "/GluGluH_M300_ToBB",              48.61*0.58],                # 43


        'EWKZJetsBB':                         [nanoPathCommon + "/EWKZJets2024Oct21",                                                                                   flatPathCommon + "/EWKZJetsBB",                     9.8],                       # 44     
        'ZJetsToBB_100to200'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-100to200",                                                                  flatPathCommon + "/ZJets/ZJetsToBB_HT-100to200" ,   5.261e+03],                 # 45    
        'ZJetsToBB_200to400'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-200to400*",                                                                 flatPathCommon + "/ZJets/ZJetsToBB_HT-200to400" ,   1012.0],                    # 46    
        'ZJetsToBB_400to600'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-400to600*",                                                                 flatPathCommon + "/ZJets/ZJetsToBB_HT-400to600" ,   114.2],                     # 47
        'ZJetsToBB_600to800'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-600to800*",                                                                 flatPathCommon + "/ZJets/ZJetsToBB_HT-600to800" ,   25.34],                     # 48
        'ZJetsToBB_800toInf'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-800toInf*",                                                                 flatPathCommon + "/ZJets/ZJetsToBB_HT-800toInf" ,   12.99],                     # 49
        'EWKZJetsqq':                       [nanoPathCommon + "/EWKZJets2024Oct21",                                                                                     flatPathCommon + "/EWKZJetsqq",                     9.8],                       # 50     
        'ZJetsToqq_100to200'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-100to200",                                                                  flatPathCommon + "/ZJets/ZJetsToqq_HT-100to200" ,   5.261e+03],                 # 51    
        'ZJetsToqq_200to400'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-200to400*",                                                                 flatPathCommon + "/ZJets/ZJetsToqq_HT-200to400" ,   1012.0],                    # 52    
        'ZJetsToqq_400to600'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-400to600*",                                                                 flatPathCommon + "/ZJets/ZJetsToqq_HT-400to600" ,   114.2],                     # 53
        'ZJetsToqq_600to800'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-600to800*",                                                                 flatPathCommon + "/ZJets/ZJetsToqq_HT-600to800" ,   25.34],                     # 54
        'ZJetsToqq_800toInf'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-800toInf*",                                                                 flatPathCommon + "/ZJets/ZJetsToqq_HT-800toInf" ,   12.99],                     # 55
      

    }
    dfMC = pd.DataFrame(processesMC).T
    dfMC.columns = ['nanoPath', 'flatPath', 'xsection']
    dfMC = dfMC.reset_index()
    dfMC = dfMC.rename(columns={'index': 'process'})
    print(dfMC)
    dfMC.to_csv("/t3home/gcelotto/ggHbb/commonScripts/processesMC.csv")
    processesData ={
    'Data1A':                             [nanoPathCommon + "/Data1A2024Oct18",                                                                                       flatPathCommon + "/Data1A"    ,                     0.774,     1017],                  # 0              
    'Data2A':                          [nanoPathCommon + "/Data2A2024Oct23",                                                                                       flatPathCommon + "/Data2A"    ,                     0.774,     1017],                     # 1              
    'Data1D'      :                    [nanoPathCommon + "/Data1D2024Dec04",                                                                                       flatPathCommon + "/Data1D" ,                        5.302,     5509],                     # 2
    }
    dfData = pd.DataFrame(processesData).T
    dfData.columns = ['nanoPath', 'flatPath', 'lumi', 'nFiles']
    dfData = dfData.reset_index()
    dfData = dfData.rename(columns={'index': 'process'})
    
    print(dfData)
    dfData.to_csv("/t3home/gcelotto/ggHbb/commonScripts/processesData.csv")
    return dfMC, dfData
getProcessesDataFrame()