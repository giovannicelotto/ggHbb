import pandas as pd
def getProcessesDataFrame():
    nanoPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH"
    flatPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB"
    processesMC = {
        'GluGluHToBB':                      [nanoPathCommon + "/MCfiducial_corrections2025Mar10/GluGluHToBB_M-125_TuneCP5_13TeV-powheg-pythia8/*/*/*/others",                   flatPathCommon + "/GluGluHToBB/others"    ,                48.61*0.58],                # 0     
        'EWKZJets':                         [nanoPathCommon + "/MCfiducial_corrections2025Mar10/EWKZ2Jets*",                                                                          flatPathCommon + "/EWKZJets",                       9.8],                       # 1     
        'WW':                               [nanoPathCommon + "/MCfiducial_corrections2025Mar10/WW_TuneCP5_13TeV-pythia8",                                                            flatPathCommon + "/diboson/WW",                     75.8],                      # 2     
        'WZ':                               [nanoPathCommon + "/MCfiducial_corrections2025Mar10/WZ_TuneCP5_13TeV-pythia8",                                                            flatPathCommon + "/diboson/WZ",                     27.6],                      # 3     
        'ZZ':                               [nanoPathCommon + "/MCfiducial_corrections2025Mar10/ZZ_TuneCP5_13TeV-pythia8",                                                            flatPathCommon + "/diboson/ZZ",                     12.14	],                  # 4         
        'ST_s-channel-hadronic':            [nanoPathCommon + "/MCfiducial_corrections2025Mar10/ST_s-channel_4f_hadronicDecays_TuneCP5_13TeV-amcatnlo-pythia8",                       flatPathCommon + "/singleTop/s-channel_hadronic",          11.24],                     # 5     
        'ST_s-channel-leptononic':          [nanoPathCommon + "/MCfiducial_corrections2025Mar10/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8",                         flatPathCommon + "/singleTop/s-channel_leptonic",          3.74],                      # 6 
        'ST_t-channel-antitop':             [nanoPathCommon + "/MCfiducial_corrections2025Mar10/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",        flatPathCommon + "/singleTop/t-channel_antitop" ,          69.09],                     # 7     
        'ST_t-channel-top':                 [nanoPathCommon + "/MCfiducial_corrections2025Mar10/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",            flatPathCommon + "/singleTop/t-channel_top"     ,          115.3],                     # 8     
        'ST_tW-antitop':                    [nanoPathCommon + "/MCfiducial_corrections2025Mar10/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",                       flatPathCommon + "/singleTop/tW-channel_antitop",          34.97],                     # 9     
        'ST_tW-top':                        [nanoPathCommon + "/MCfiducial_corrections2025Mar10/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",                           flatPathCommon + "/singleTop/tW-channel_top"    ,          34.91	],                  # 10        
         #ttbar cross section: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO
         #w boson decay modes : https://pdg.lbl.gov/2023/listings/rpp2023-list-w-boson.pdf
        'TTTo2L2Nu':                        [nanoPathCommon + "/MCfiducial_corrections2025Mar10/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",                                              flatPathCommon + "/ttbar/ttbar2L2Nu"            ,   833.9*(0.1086*3)**2],       # 11        
        'TTToHadronic':                     [nanoPathCommon + "/MCfiducial_corrections2025Mar10/TTToHadronic_TuneCP5_13TeV-powheg-pythia8",                                           flatPathCommon + "/ttbar/ttbarHadronic"         ,   833.9*(0.6741)**2],         # 12        
        'TTToSemiLeptonic':                 [nanoPathCommon + "/MCfiducial_corrections2025Mar10/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",                                       flatPathCommon + "/ttbar/ttbarSemiLeptonic"     ,   2*833.9*0.6741*0.1086*3],   # 13        
        'WJetsToLNu':                       [nanoPathCommon + "/MCfiducial_corrections2025Mar10/WJetsToLNu*",                                                                         flatPathCommon + "/WJets/WJetsToLNu"            ,   62070.0],                   # 14            
        'WJetsToQQ_200to400'      :         [nanoPathCommon + "/MCfiducial_corrections2025Mar10/WJetsToQQ_HT-200to400*",                                                              flatPathCommon + "/WJets/WJetsToQQ_HT-200to400" ,   2549.0],                    # 15    
        'WJetsToQQ_400to600'      :         [nanoPathCommon + "/MCfiducial_corrections2025Mar10/WJetsToQQ_HT-400to600*",                                                              flatPathCommon + "/WJets/WJetsToQQ_HT-400to600" ,   276.5],                     # 16
        'WJetsToQQ_600to800'      :         [nanoPathCommon + "/MCfiducial_corrections2025Mar10/WJetsToQQ_HT-600to800*",                                                              flatPathCommon + "/WJets/WJetsToQQ_HT-600to800" ,   59.25],                     # 17
        'WJetsToQQ_800toInf'      :         [nanoPathCommon + "/MCfiducial_corrections2025Mar10/WJetsToQQ_HT-800toInf*",                                                              flatPathCommon + "/WJets/WJetsToQQ_HT-800toInf" ,   28.75],                     # 18
        'ZJetsToQQ_200to400'      :         [nanoPathCommon + "/MCfiducial_corrections2025Mar10/ZJetsToQQ_HT-200to400*",                                                              flatPathCommon + "/ZJets/ZJetsToQQ_HT-200to400" ,   1012.0],                    # 19    
        'ZJetsToQQ_400to600'      :         [nanoPathCommon + "/MCfiducial_corrections2025Mar10/ZJetsToQQ_HT-400to600*",                                                              flatPathCommon + "/ZJets/ZJetsToQQ_HT-400to600" ,   114.2],                     # 20
        'ZJetsToQQ_600to800'      :         [nanoPathCommon + "/MCfiducial_corrections2025Mar10/ZJetsToQQ_HT-600to800*",                                                              flatPathCommon + "/ZJets/ZJetsToQQ_HT-600to800" ,   25.34],                     # 21
        'ZJetsToQQ_800toInf'      :         [nanoPathCommon + "/MCfiducial_corrections2025Mar10/ZJetsToQQ_HT-800toInf*",                                                              flatPathCommon + "/ZJets/ZJetsToQQ_HT-800toInf" ,   12.99],                     # 22
        'QCD_MuEnriched_Pt-1000':           [nanoPathCommon + "/QCDMuEnriched*/QCD_Pt-1000*",                                                                               flatPathCommon + "/QCD_Pt1000ToInf"             ,    1.085],                     # 23    
        'QCD_MuEnriched_Pt-800To1000':      [nanoPathCommon + "/QCDMuEnriched*/QCD_Pt-800To1000*",                                                                          flatPathCommon + "/QCD_Pt800To1000"             ,    3.318],                     # 24    
        'QCD_MuEnriched_Pt-600To800':       [nanoPathCommon + "/QCDMuEnriched*/QCD_Pt-600To800*",                                                                           flatPathCommon + "/QCD_Pt600To800"              ,    18.12	],                  # 25        
        'QCD_MuEnriched_Pt-470To600':       [nanoPathCommon + "/QCDMuEnriched*/QCD_Pt-470To600*",                                                                           flatPathCommon + "/QCD_Pt470To600"              ,    58.9],                      # 26    
        'QCD_MuEnriched_Pt-300To470':       [nanoPathCommon + "/QCDMuEnriched*/QCD_Pt-300To470*",                                                                           flatPathCommon + "/QCD_Pt300To470"              ,    622.6],                     # 27    
        'QCD_MuEnriched_Pt-170To300':       [nanoPathCommon + "/QCDMuEnriched*/QCD_Pt-170To300*",                                                                           flatPathCommon + "/QCD_Pt170To300"              ,    7000.0],                    # 28        
        'QCD_MuEnriched_Pt-120To170':       [nanoPathCommon + "/QCDMuEnriched*/QCD_Pt-120To170*",                                                                           flatPathCommon + "/QCD_Pt120To170"              ,    21280.0],                   # 29        
        'QCD_MuEnriched_Pt-80To120':        [nanoPathCommon + "/QCDMuEnriched*/QCD_Pt-80To120*",                                                                            flatPathCommon + "/QCD_Pt80To120"               ,    87740.0],                   # 30        
        'QCD_MuEnriched_Pt-50To80':         [nanoPathCommon + "/QCDMuEnriched*/QCD_Pt-50To80*",                                                                             flatPathCommon + "/QCD_Pt50To80"                ,    381700.0],                  # 31        
        'QCD_MuEnriched_Pt-30To50':         [nanoPathCommon + "/QCDMuEnriched*/QCD_Pt-30To50*",                                                                             flatPathCommon + "/QCD_Pt30To50"                ,    1367000.0],                 # 32        
        'QCD_MuEnriched_Pt-20To30':         [nanoPathCommon + "/QCDMuEnriched*/QCD_Pt-20To30*",                                                                             flatPathCommon + "/QCD_Pt20To30"                ,    2527000.0],                 # 33        
        'QCD_MuEnriched_Pt-15To20':         [nanoPathCommon + "/QCDMuEnriched*/QCD_Pt-15To20*",                                                                             flatPathCommon + "/QCD_Pt15To20"                ,    2800000.0	],              # 34            
        'ZJetsToQQ_100to200'      :         [nanoPathCommon + "/MCfiducial_corrections2025Mar10/ZJetsToQQ_HT-100to200_TuneCP5_13TeV-madgraphMLM-pythia8",                             flatPathCommon + "/ZJets/ZJetsToQQ_HT-100to200" ,   5.261e+03],                 # 35    
        'VBFHToBB'      :                   [nanoPathCommon + "/MCfiducial_corrections2025Mar10/VBFHToBB*/",                                                                          flatPathCommon + "/VBFHToBB" ,                      3.766*0.58],                # 36    
        'GluGluHToBBMINLO'          :       [nanoPathCommon + "/MCfiducial_corrections2025Mar10/GluGluHToBB_M-125_TuneCP5_MINLO_NNLOPS_13TeV-powheg-pythia*/",                        flatPathCommon + "/MINLOGluGluHToBB",               48.61*0.58],                # 37
        'GluGluH_M50_ToBB':                 [nanoPathCommon + "/MCfiducial_corrections2025Mar10/GluGluSpin0ToBBbar_W_1p0_M_50_MuEnriched_TuneCP5_13TeV_pythia8",                           flatPathCommon + "/GluGluH_M50_ToBB",               48.61*0.58],                # 38
        'GluGluH_M70_ToBB':                 [nanoPathCommon + "/MCfiducial_corrections2025Mar10/GluGluSpin0ToBBbar_W_1p0_M_70_MuEnriched_TuneCP5_13TeV_pythia8",                           flatPathCommon + "/GluGluH_M70_ToBB",               48.61*0.58],                # 39
        'GluGluH_M100_ToBB':                [nanoPathCommon + "/MCfiducial_corrections2025Mar10/GluGluSpin0ToBBbar_W_1p0_M_100_MuEnriched_TuneCP5_13TeV_pythia8",                          flatPathCommon + "/GluGluH_M100_ToBB",              48.61*0.58],                # 40
        'GluGluH_M200_ToBB':                [nanoPathCommon + "/MCfiducial_corrections2025Mar10/GluGluSpin0ToBBbar_W_1p0_M_200_MuEnriched_TuneCP5_13TeV_pythia8",                          flatPathCommon + "/GluGluH_M200_ToBB",              48.61*0.58],                # 41
        'GluGluH_M300_ToBB':                [nanoPathCommon + "/MCfiducial_corrections2025Mar10/GluGluSpin0ToBBbar_W_1p0_M_300_MuEnriched_TuneCP5_13TeV_pythia8",                          flatPathCommon + "/GluGluH_M300_ToBB",              48.61*0.58],                # 42
        'GluGluHToBB_tr':                   [nanoPathCommon + "/MCfiducial_corrections2025Mar10/GluGluHToBB_M-125_TuneCP5_13TeV-powheg-pythia8/*/*/*/training",                   flatPathCommon + "/GluGluHToBB/training"    ,                48.61*0.58],                # 0     


        #'EWKZJetsBB':                         [nanoPathCommon + "/EWKZJets2024Oct21",                                                                                   flatPathCommon + "/EWKZJetsBB",                     9.8],                       # 44     
        #'ZJetsToBB_100to200'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-100to200",                                                                  flatPathCommon + "/ZJets/ZJetsToBB_HT-100to200" ,   5.261e+03],                 # 45    
        #'ZJetsToBB_200to400'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-200to400*",                                                                 flatPathCommon + "/ZJets/ZJetsToBB_HT-200to400" ,   1012.0],                    # 46    
        #'ZJetsToBB_400to600'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-400to600*",                                                                 flatPathCommon + "/ZJets/ZJetsToBB_HT-400to600" ,   114.2],                     # 47
        #'ZJetsToBB_600to800'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-600to800*",                                                                 flatPathCommon + "/ZJets/ZJetsToBB_HT-600to800" ,   25.34],                     # 48
        #'ZJetsToBB_800toInf'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-800toInf*",                                                                 flatPathCommon + "/ZJets/ZJetsToBB_HT-800toInf" ,   12.99],                     # 49
        #'EWKZJetsqq':                       [nanoPathCommon + "/EWKZJets2024Oct21",                                                                                     flatPathCommon + "/EWKZJetsqq",                     9.8],                       # 50     
        #'ZJetsToqq_100to200'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-100to200",                                                                  flatPathCommon + "/ZJets/ZJetsToqq_HT-100to200" ,   5.261e+03],                 # 51    
        #'ZJetsToqq_200to400'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-200to400*",                                                                 flatPathCommon + "/ZJets/ZJetsToqq_HT-200to400" ,   1012.0],                    # 52    
        #'ZJetsToqq_400to600'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-400to600*",                                                                 flatPathCommon + "/ZJets/ZJetsToqq_HT-400to600" ,   114.2],                     # 53
        #'ZJetsToqq_600to800'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-600to800*",                                                                 flatPathCommon + "/ZJets/ZJetsToqq_HT-600to800" ,   25.34],                     # 54
        #'ZJetsToqq_800toInf'      :         [nanoPathCommon + "/ZJets2024Oct18/ZJetsToQQ_HT-800toInf*",                                                                 flatPathCommon + "/ZJets/ZJetsToqq_HT-800toInf" ,   12.99],                     # 55
      

    }
    dfMC = pd.DataFrame(processesMC).T
    dfMC.columns = ['nanoPath', 'flatPath', 'xsection']
    dfMC = dfMC.reset_index()
    dfMC = dfMC.rename(columns={'index': 'process'})
    print(dfMC)
    dfMC.to_csv("/t3home/gcelotto/ggHbb/commonScripts/processesMC.csv")
    processesData ={
    'Data1A':                          [nanoPathCommon + "/DataFiducial_corrections2025Mar12/ParkingBPH1/crab_data_Run2018A_part1",                         flatPathCommon + "/Data1A"    ,                     0.774,     1017],                  # 0              
    'Data2A':                          [nanoPathCommon + "/DataFiducial_corrections2025Mar12/ParkingBPH2/crab_data_Run2018A_part2",                         flatPathCommon + "/Data2A"    ,                     0.774,     1017],                     # 1              
    'Data1D':                          [nanoPathCommon + "/DataFiducial_corrections2025Mar12/ParkingBPH1/crab_data_Run2018D_part1",                         flatPathCommon + "/Data1D" ,                        5.302,     5508],                     # 2
    'Data3A':                          [nanoPathCommon + "/DataFiducial_corrections2025Mar12/ParkingBPH3/crab_data_Run2018A_part3",                         flatPathCommon + "/Data3A"    ,                     0.774,     1018],                     # 3              
    'Data4A':                          [nanoPathCommon + "/DataFiducial_corrections2025Mar12/ParkingBPH4/crab_data_Run2018A_part4",                         flatPathCommon + "/Data4A"    ,                     0.774,     1016],                     # 4              
    'Data5A':                          [nanoPathCommon + "/DataFiducial_corrections2025Mar12/ParkingBPH5/crab_data_Run2018A_part5",                         flatPathCommon + "/Data5A"    ,                     0.774,     1016],                     # 5              
    'Data6A':                          [nanoPathCommon + "/DataFiducial_corrections2025Mar12/ParkingBPH6/crab_data_Run2018A_part6",                         flatPathCommon + "/Data6A"    ,                     0.774,     1017],                     # 6              
    'Data2D':                          [nanoPathCommon + "/DataFiducial_corrections2025Mar12/ParkingBPH2/crab_data_Run2018D_part2",                         flatPathCommon + "/Data2D"    ,                     5.302,     5509],                     # 7              
    }
    dfData = pd.DataFrame(processesData).T
    dfData.columns = ['nanoPath', 'flatPath', 'lumi', 'nFiles']
    dfData = dfData.reset_index()
    dfData = dfData.rename(columns={'index': 'process'})
    
    print(dfData)
    dfData.to_csv("/t3home/gcelotto/ggHbb/commonScripts/processesData.csv")
    return dfMC, dfData
getProcessesDataFrame()