import pandas as pd
def getProcessesDataFrame():
    nanoPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH"
    flatPathCommon = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB"
    nanoFolder = "MC_fiducial_JESsmearedNominal2025Apr09"
    nanoFolder_smearedDown = "MC_fiducial_JESsmearedDown2025Apr08"
    nanoFolder_smearedUp = "MC_fiducial_JESsmearedUp2025Apr08"
    processesMC = {
    'GluGluHToBB':                      [nanoPathCommon + "/"+nanoFolder+"/GluGluHToBB_M-125_TuneCP5_13TeV-powheg-pythia8/*/*/*/others",                            flatPathCommon+"/MC" + "/GluGluHToBB/others"    ,                48.61*0.58],                # 0     
    'EWKZJets':                         [nanoPathCommon + "/"+nanoFolder+"/EWKZ2Jets*",                                                                          flatPathCommon+"/MC" + "/EWKZJets",                       9.8],                       # 1     
    'WW':                               [nanoPathCommon + "/"+nanoFolder+"/WW_TuneCP5_13TeV-pythia8",                                                            flatPathCommon+"/nonResonant" + "/diboson/WW",                     75.8],                      # 2     
    'WZ':                               [nanoPathCommon + "/"+nanoFolder+"/WZ_TuneCP5_13TeV-pythia8",                                                            flatPathCommon+"/MC" + "/diboson/WZ",                     27.6],                      # 3     
    'ZZ':                               [nanoPathCommon + "/"+nanoFolder+"/ZZ_TuneCP5_13TeV-pythia8",                                                            flatPathCommon+"/MC" + "/diboson/ZZ",                     12.14	],                  # 4         
    'ST_s-channel-hadronic':            [nanoPathCommon + "/"+nanoFolder+"/ST_s-channel_4f_hadronicDecays_TuneCP5_13TeV-amcatnlo-pythia8",                       flatPathCommon+"/nonResonant" + "/singleTop/s-channel_hadronic",          11.24],                     # 5     
    'ST_s-channel-leptononic':          [nanoPathCommon + "/"+nanoFolder+"/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8",                         flatPathCommon+"/nonResonant" + "/singleTop/s-channel_leptonic",          3.74],                      # 6 
    'ST_t-channel-antitop':             [nanoPathCommon + "/"+nanoFolder+"/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",        flatPathCommon+"/nonResonant" + "/singleTop/t-channel_antitop" ,          69.09],                     # 7     
    'ST_t-channel-top':                 [nanoPathCommon + "/"+nanoFolder+"/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",            flatPathCommon+"/nonResonant" + "/singleTop/t-channel_top"     ,          115.3],                     # 8     
    'ST_tW-antitop':                    [nanoPathCommon + "/"+nanoFolder+"/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",                       flatPathCommon+"/nonResonant" + "/singleTop/tW-channel_antitop",          34.97],                     # 9     
    'ST_tW-top':                        [nanoPathCommon + "/"+nanoFolder+"/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",                           flatPathCommon+"/nonResonant" + "/singleTop/tW-channel_top"    ,          34.91	],                  # 10        
        #ttbar cross section: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO
        #w boson decay modes : https://pdg.lbl.gov/2023/listings/rpp2023-list-w-boson.pdf
    'TTTo2L2Nu':                        [nanoPathCommon + "/"+nanoFolder+"/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",                                              flatPathCommon+"/nonResonant" + "/ttbar/ttbar2L2Nu"            ,   833.9*(0.1086*3)**2],       # 11        
    'TTToHadronic':                     [nanoPathCommon + "/"+nanoFolder+"/TTToHadronic_TuneCP5_13TeV-powheg-pythia8",                                           flatPathCommon+"/nonResonant" + "/ttbar/ttbarHadronic"         ,   833.9*(0.6741)**2],         # 12        
    'TTToSemiLeptonic':                 [nanoPathCommon + "/"+nanoFolder+"/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",                                       flatPathCommon+"/nonResonant" + "/ttbar/ttbarSemiLeptonic"     ,   2*833.9*0.6741*0.1086*3],   # 13        
    'WJetsToLNu':                       [nanoPathCommon + "/"+nanoFolder+"/WJetsToLNu*",                                                                         flatPathCommon+"/nonResonant" + "/WJets/WJetsToLNu"            ,   62070.0],                   # 14            
    'WJetsToQQ_200to400'      :         [nanoPathCommon + "/"+nanoFolder+"/WJetsToQQ_HT-200to400*",                                                              flatPathCommon+"/nonResonant" + "/WJets/WJetsToQQ_HT-200to400" ,   2549.0],                    # 15    
    'WJetsToQQ_400to600'      :         [nanoPathCommon + "/"+nanoFolder+"/WJetsToQQ_HT-400to600*",                                                              flatPathCommon+"/nonResonant" + "/WJets/WJetsToQQ_HT-400to600" ,   276.5],                     # 16
    'WJetsToQQ_600to800'      :         [nanoPathCommon + "/"+nanoFolder+"/WJetsToQQ_HT-600to800*",                                                              flatPathCommon+"/nonResonant" + "/WJets/WJetsToQQ_HT-600to800" ,   59.25],                     # 17
    'WJetsToQQ_800toInf'      :         [nanoPathCommon + "/"+nanoFolder+"/WJetsToQQ_HT-800toInf*",                                                              flatPathCommon+"/nonResonant" + "/WJets/WJetsToQQ_HT-800toInf" ,   28.75],                     # 18
    'ZJetsToQQ_200to400'      :         [nanoPathCommon + "/"+nanoFolder+"/ZJetsToQQ_HT-200to400*",                                                              flatPathCommon+"/MC" + "/ZJets/ZJetsToQQ_HT-200to400" ,   1012.0],                    # 19    
    'ZJetsToQQ_400to600'      :         [nanoPathCommon + "/"+nanoFolder+"/ZJetsToQQ_HT-400to600*",                                                              flatPathCommon+"/MC" + "/ZJets/ZJetsToQQ_HT-400to600" ,   114.2],                     # 20
    'ZJetsToQQ_600to800'      :         [nanoPathCommon + "/"+nanoFolder+"/ZJetsToQQ_HT-600to800*",                                                              flatPathCommon+"/MC" + "/ZJets/ZJetsToQQ_HT-600to800" ,   25.34],                     # 21
    'ZJetsToQQ_800toInf'      :         [nanoPathCommon + "/"+nanoFolder+"/ZJetsToQQ_HT-800toInf*",                                                              flatPathCommon+"/MC" + "/ZJets/ZJetsToQQ_HT-800toInf" ,   12.99],                     # 22
    'QCD_MuEnriched_Pt-1000':           [nanoPathCommon + "/"+nanoFolder+"/QCD_Pt-1000*",                                                                               flatPathCommon+"/nonResonant" + "/QCD_Pt1000ToInf"             ,    1.085],                     # 23    
    'QCD_MuEnriched_Pt-800To1000':      [nanoPathCommon + "/"+nanoFolder+"/QCD_Pt-800To1000*",                                                                          flatPathCommon+"/nonResonant" + "/QCD_Pt800To1000"             ,    3.318],                     # 24    
    'QCD_MuEnriched_Pt-600To800':       [nanoPathCommon + "/"+nanoFolder+"/QCD_Pt-600To800*",                                                                           flatPathCommon+"/nonResonant" + "/QCD_Pt600To800"              ,    18.12	],                  # 25        
    'QCD_MuEnriched_Pt-470To600':       [nanoPathCommon + "/"+nanoFolder+"/QCD_Pt-470To600*",                                                                           flatPathCommon+"/nonResonant" + "/QCD_Pt470To600"              ,    58.9],                      # 26    
    'QCD_MuEnriched_Pt-300To470':       [nanoPathCommon + "/"+nanoFolder+"/QCD_Pt-300To470*",                                                                           flatPathCommon+"/nonResonant" + "/QCD_Pt300To470"              ,    622.6],                     # 27    
    'QCD_MuEnriched_Pt-170To300':       [nanoPathCommon + "/"+nanoFolder+"/QCD_Pt-170To300*",                                                                           flatPathCommon+"/nonResonant" + "/QCD_Pt170To300"              ,    7000.0],                    # 28        
    'QCD_MuEnriched_Pt-120To170':       [nanoPathCommon + "/"+nanoFolder+"/QCD_Pt-120To170*",                                                                           flatPathCommon+"/nonResonant" + "/QCD_Pt120To170"              ,    21280.0],                   # 29        
    'QCD_MuEnriched_Pt-80To120':        [nanoPathCommon + "/"+nanoFolder+"/QCD_Pt-80To120*",                                                                            flatPathCommon+"/nonResonant" + "/QCD_Pt80To120"               ,    87740.0],                   # 30        
    'QCD_MuEnriched_Pt-50To80':         [nanoPathCommon + "/"+nanoFolder+"/QCD_Pt-50To80*",                                                                             flatPathCommon+"/nonResonant" + "/QCD_Pt50To80"                ,    381700.0],                  # 31        
    'QCD_MuEnriched_Pt-30To50':         [nanoPathCommon + "/"+nanoFolder+"/QCD_Pt-30To50*",                                                                             flatPathCommon+"/nonResonant" + "/QCD_Pt30To50"                ,    1367000.0],                 # 32        
    'QCD_MuEnriched_Pt-20To30':         [nanoPathCommon + "/"+nanoFolder+"/QCD_Pt-20To30*",                                                                             flatPathCommon+"/nonResonant" + "/QCD_Pt20To30"                ,    2527000.0],                 # 33        
    'QCD_MuEnriched_Pt-15To20':         [nanoPathCommon + "/"+nanoFolder+"/QCD_Pt-15To20*",                                                                             flatPathCommon+"/nonResonant" + "/QCD_Pt15To20"                ,    2800000.0	],              # 34            
    'ZJetsToQQ_100to200'      :         [nanoPathCommon + "/"+nanoFolder+"/ZJetsToQQ_HT-100to200_TuneCP5_13TeV-madgraphMLM-pythia8",                             flatPathCommon+"/MC" + "/ZJets/ZJetsToQQ_HT-100to200" ,   5.261e+03],                 # 35    
    'VBFHToBB'      :                   [nanoPathCommon + "/"+nanoFolder+"/VBFHToBB*/",                                                                          flatPathCommon+"/MC" + "/VBFHToBB" ,                      3.766*0.58],                # 36    
    'GluGluHToBBMINLO'          :       [nanoPathCommon + "/"+nanoFolder+"/GluGluHToBB_M-125_TuneCP5_MINLO_NNLOPS_13TeV-powheg-pythia*/*/*/*/others",                        flatPathCommon+"/MC" + "/MINLOGluGluHToBB/others",               48.61*0.58],                # 37
    'GluGluH_M50_ToBB':                 [nanoPathCommon + "/"+nanoFolder+"/GluGluSpin0ToBBbar_W_1p0_M_50_MuEnriched_TuneCP5_13TeV_pythia8",                           flatPathCommon+"/MC" + "/GluGluH_M50_ToBB",               48.61*0.58],                # 38
    'GluGluH_M70_ToBB':                 [nanoPathCommon + "/"+nanoFolder+"/GluGluSpin0ToBBbar_W_1p0_M_70_MuEnriched_TuneCP5_13TeV_pythia8",                           flatPathCommon+"/MC" + "/GluGluH_M70_ToBB",               48.61*0.58],                # 39
    'GluGluH_M100_ToBB':                [nanoPathCommon + "/"+nanoFolder+"/GluGluSpin0ToBBbar_W_1p0_M_100_MuEnriched_TuneCP5_13TeV_pythia8",                          flatPathCommon+"/MC" + "/GluGluH_M100_ToBB",              48.61*0.58],                # 40
    'GluGluH_M200_ToBB':                [nanoPathCommon + "/"+nanoFolder+"/GluGluSpin0ToBBbar_W_1p0_M_200_MuEnriched_TuneCP5_13TeV_pythia8",                          flatPathCommon+"/MC" + "/GluGluH_M200_ToBB",              48.61*0.58],                # 41
    'GluGluH_M300_ToBB':                [nanoPathCommon + "/"+nanoFolder+"/GluGluSpin0ToBBbar_W_1p0_M_300_MuEnriched_TuneCP5_13TeV_pythia8",                          flatPathCommon+"/MC" + "/GluGluH_M300_ToBB",              48.61*0.58],                # 42
    'GluGluHToBB_tr':                   [nanoPathCommon + "/"+nanoFolder+"/GluGluHToBB_M-125_TuneCP5_13TeV-powheg-pythia8/*/*/*/training",                   flatPathCommon+"/MC" + "/GluGluHToBB/training"    ,                48.61*0.58],                # 0 
    'GluGluHToBBMINLO_tr'          :       [nanoPathCommon + "/"+nanoFolder+"/GluGluHToBB_M-125_TuneCP5_MINLO_NNLOPS_13TeV-powheg-pythia*/*/*/*/training",                        flatPathCommon+"/MC" + "/MINLOGluGluHToBB/training",               48.61*0.58],                # 37
    'GluGluHToBB_amcatnlo'          :       [nanoPathCommon + "/"+"/GluGluHToBB_amcatnlo2025May22/GluGluHToBB_M-125_TuneCP5_13TeV-amcatnloFXFX-pythia8",                        flatPathCommon+"/MC" + "/GluGluHToBB_amcatnlo",               48.61*0.58],                # 37
        
    'GluGluHToBB_smearedDown':                      [nanoPathCommon + "/"+nanoFolder_smearedDown+"/GluGluHToBB_M-125_TuneCP5_13TeV-powheg-pythia8/*/*/*/others",                   flatPathCommon+"/JER" + "/GluGluHToBB_sD/others"    ,                48.61*0.58],                # 0     
    'EWKZJets_smearedDown':                         [nanoPathCommon + "/"+nanoFolder_smearedDown+"/EWKZ2Jets*",                                                                          flatPathCommon+"/JER" + "/EWKZJets_sD",                       9.8],                       # 1     
    'WZ_smearedDown':                               [nanoPathCommon + "/"+nanoFolder_smearedDown+"/WZ_TuneCP5_13TeV-pythia8",                                                            flatPathCommon+"/JER" + "/diboson/WZ_sD",                     27.6],                      # 3     
    'ZZ_smearedDown':                               [nanoPathCommon + "/"+nanoFolder_smearedDown+"/ZZ_TuneCP5_13TeV-pythia8",                                                            flatPathCommon+"/JER" + "/diboson/ZZ_sD",                     12.14	],                  # 4         
    'ZJetsToQQ_200to400_smearedDown'      :         [nanoPathCommon + "/"+nanoFolder_smearedDown+"/ZJetsToQQ_HT-200to400*",                                                              flatPathCommon+"/JER" + "/ZJets/ZJetsToQQ_HT-200to400_sD" ,   1012.0],                    # 19    
    'ZJetsToQQ_400to600_smearedDown'      :         [nanoPathCommon + "/"+nanoFolder_smearedDown+"/ZJetsToQQ_HT-400to600*",                                                              flatPathCommon+"/JER" + "/ZJets/ZJetsToQQ_HT-400to600_sD" ,   114.2],                     # 20
    'ZJetsToQQ_600to800_smearedDown'      :         [nanoPathCommon + "/"+nanoFolder_smearedDown+"/ZJetsToQQ_HT-600to800*",                                                              flatPathCommon+"/JER" + "/ZJets/ZJetsToQQ_HT-600to800_sD" ,   25.34],                     # 21
    'ZJetsToQQ_800toInf_smearedDown'      :         [nanoPathCommon + "/"+nanoFolder_smearedDown+"/ZJetsToQQ_HT-800toInf*",                                                              flatPathCommon+"/JER" + "/ZJets/ZJetsToQQ_HT-800toInf_sD" ,   12.99],                     # 22
    'ZJetsToQQ_100to200_smearedDown'      :         [nanoPathCommon + "/"+nanoFolder_smearedDown+"/ZJetsToQQ_HT-100to200_TuneCP5_13TeV-madgraphMLM-pythia8",                             flatPathCommon+"/JER" + "/ZJets/ZJetsToQQ_HT-100to200_sD" ,   5.261e+03],                 # 35    
    'VBFHToBB_smearedDown'      :                   [nanoPathCommon + "/"+nanoFolder_smearedDown+"/VBFHToBB*/",                                                                          flatPathCommon+"/JER" + "/VBFHToBB_sD",                      3.766*0.58],                # 36    
    'GluGluHToBBMINLO_smearedDown'          :       [nanoPathCommon + "/"+nanoFolder_smearedDown+"/GluGluHToBB_M-125_TuneCP5_MINLO_NNLOPS_13TeV-powheg-pythia*/*/*/*/others",                        flatPathCommon+"/JER" + "/MINLOGluGluHToBB_sD" ,               48.61*0.58],                # 37
    #'GluGluHToBB_tr_smearedDown':                      [nanoPathCommon + "/"+nanoFolder_smearedDown+"/GluGluHToBB_M-125_TuneCP5_13TeV-powheg-pythia8/*/*/*/training",                   flatPathCommon+"/JER" + "/GluGluHToBB_sD/training"    ,                48.61*0.58],                # 0     
    #'GluGluHToBBMINLO_tr_smearedDown'          :       [nanoPathCommon + "/"+nanoFolder+"/GluGluHToBB_M-125_TuneCP5_MINLO_NNLOPS_13TeV-powheg-pythia*/*/*/*/training",                        flatPathCommon+"/JER" + "/MINLOGluGluHToBB_tr_sD",               48.61*0.58],                # 37

    'GluGluHToBB_smearedUp':                      [nanoPathCommon + "/"+nanoFolder_smearedUp+"/GluGluHToBB_M-125_TuneCP5_13TeV-powheg-pythia8/*/*/*/others",                   flatPathCommon+"/JER" + "/GluGluHToBB_sU/others"    ,                48.61*0.58],                # 0     
    'EWKZJets_smearedUp':                         [nanoPathCommon + "/"+nanoFolder_smearedUp+"/EWKZ2Jets*",                                                                          flatPathCommon+"/JER" + "/EWKZJets_sU",                       9.8],                       # 1     
    'WZ_smearedUp':                               [nanoPathCommon + "/"+nanoFolder_smearedUp+"/WZ_TuneCP5_13TeV-pythia8",                                                            flatPathCommon+"/JER" + "/diboson/WZ_sU",                     27.6],                      # 3     
    'ZZ_smearedUp':                               [nanoPathCommon + "/"+nanoFolder_smearedUp+"/ZZ_TuneCP5_13TeV-pythia8",                                                            flatPathCommon+"/JER" + "/diboson/ZZ_sU",                     12.14	],                  # 4         
    'ZJetsToQQ_200to400_smearedUp'      :         [nanoPathCommon + "/"+nanoFolder_smearedUp+"/ZJetsToQQ_HT-200to400*",                                                              flatPathCommon+"/JER" + "/ZJets/ZJetsToQQ_HT-200to400_sU" ,   1012.0],                    # 19    
    'ZJetsToQQ_400to600_smearedUp'      :         [nanoPathCommon + "/"+nanoFolder_smearedUp+"/ZJetsToQQ_HT-400to600*",                                                              flatPathCommon+"/JER" + "/ZJets/ZJetsToQQ_HT-400to600_sU" ,   114.2],                     # 20
    'ZJetsToQQ_600to800_smearedUp'      :         [nanoPathCommon + "/"+nanoFolder_smearedUp+"/ZJetsToQQ_HT-600to800*",                                                              flatPathCommon+"/JER" + "/ZJets/ZJetsToQQ_HT-600to800_sU" ,   25.34],                     # 21
    'ZJetsToQQ_800toInf_smearedUp'      :         [nanoPathCommon + "/"+nanoFolder_smearedUp+"/ZJetsToQQ_HT-800toInf*",                                                              flatPathCommon+"/JER" + "/ZJets/ZJetsToQQ_HT-800toInf_sU" ,   12.99],                     # 22       
    'ZJetsToQQ_100to200_smearedUp'      :         [nanoPathCommon + "/"+nanoFolder_smearedUp+"/ZJetsToQQ_HT-100to200_TuneCP5_13TeV-madgraphMLM-pythia8",                             flatPathCommon+"/JER" + "/ZJets/ZJetsToQQ_HT-100to200_sU" ,   5.261e+03],                 # 35    
    'VBFHToBB_smearedUp'      :                   [nanoPathCommon + "/"+nanoFolder_smearedUp+"/VBFHToBB*/",                                                                          flatPathCommon+"/JER" + "/VBFHToBB_sU",                      3.766*0.58],                # 36    
    'GluGluHToBBMINLO_smearedUp'          :       [nanoPathCommon + "/"+nanoFolder_smearedUp+"/GluGluHToBB_M-125_TuneCP5_MINLO_NNLOPS_13TeV-powheg-pythia*/*/*/*/others",                        flatPathCommon+"/JER" + "/MINLOGluGluHToBB_sU" ,               48.61*0.58],                # 37
    #'GluGluHToBB_tr_smearedUp':                      [nanoPathCommon + "/"+nanoFolder_smearedUp+"/GluGluHToBB_M-125_TuneCP5_13TeV-powheg-pythia8/*/*/*/training",                   flatPathCommon+"/JER" + "/GluGluHToBB_sU/training"    ,                48.61*0.58],                # 0     
    #'GluGluHToBBMINLO_tr_smearedUp'          :       [nanoPathCommon + "/"+nanoFolder+"/GluGluHToBB_M-125_TuneCP5_MINLO_NNLOPS_13TeV-powheg-pythia*/*/*/*/training",                        flatPathCommon+"/JER" + "/MINLOGluGluHToBB_tr_sU",               48.61*0.58],                # 37
    }
    dfMC = pd.DataFrame(processesMC).T
    dfMC.columns = ['nanoPath', 'flatPath', 'xsection']
    dfMC = dfMC.reset_index()
    dfMC = dfMC.rename(columns={'index': 'process'})
    print(dfMC)
    dfMC.to_csv("/t3home/gcelotto/ggHbb/commonScripts/processesMC.csv")



# Now create a dataframe for JEC
    list_of_JEC_syst = ["JECAbsoluteMPFBias_Down", "JECAbsoluteMPFBias_Up", "JECAbsoluteScale_Down", "JECAbsoluteScale_Up", "JECAbsoluteStat_Down", "JECAbsoluteStat_Up", "JECFlavorQCD_Down", "JECFlavorQCD_Up", "JECFragmentation_Down", "JECFragmentation_Up", "JECPileUpDataMC_Down", "JECPileUpDataMC_Up", "JECPileUpPtBB_Down", "JECPileUpPtBB_Up", "JECPileUpPtEC1_Down", "JECPileUpPtEC1_Up", "JECPileUpPtEC2_Down", "JECPileUpPtEC2_Up", "JECPileUpPtHF_Down", "JECPileUpPtHF_Up", "JECPileUpPtRef_Down", "JECPileUpPtRef_Up", "JECRelativeBal_Down", "JECRelativeBal_Up", "JECRelativeFSR_Down", "JECRelativeFSR_Up", "JECRelativeJEREC1_Down", "JECRelativeJEREC1_Up", "JECRelativeJEREC2_Down", "JECRelativeJEREC2_Up", "JECRelativeJERHF_Down", "JECRelativeJERHF_Up", "JECRelativePtBB_Down", "JECRelativePtBB_Up", "JECRelativePtEC1_Down", "JECRelativePtEC1_Up", "JECRelativePtEC2_Down", "JECRelativePtEC2_Up", "JECRelativePtHF_Down", "JECRelativePtHF_Up", "JECRelativeSample_Down", "JECRelativeSample_Up", "JECRelativeStatEC_Down", "JECRelativeStatEC_Up", "JECRelativeStatFSR_Down", "JECRelativeStatFSR_Up", "JECRelativeStatHF_Down", "JECRelativeStatHF_Up", "JECSinglePionECAL_Down", "JECSinglePionECAL_Up", "JECSinglePionHCAL_Down", "JECSinglePionHCAL_Up", "JECTimePtEta_Down", "JECTimePtEta_Up",]
    processesMC_ref = {
    'GluGluHToBB':                      [nanoPathCommon + "/"+nanoFolder+"/GluGluHToBB_M-125_TuneCP5_13TeV-powheg-pythia8/*/*/*/others",                         flatPathCommon+"/JEC" + "/GluGluHToBB"    ,                48.61*0.58],                # 0     
    'EWKZJets':                         [nanoPathCommon + "/"+nanoFolder+"/EWKZ2Jets*",                                                                          flatPathCommon+"/JEC" + "/EWKZJets",                       9.8],                       # 1     
    'WZ':                               [nanoPathCommon + "/"+nanoFolder+"/WZ_TuneCP5_13TeV-pythia8",                                                            flatPathCommon+"/JEC" + "/diboson/WZ",                     27.6],                      # 3     
    'ZZ':                               [nanoPathCommon + "/"+nanoFolder+"/ZZ_TuneCP5_13TeV-pythia8",                                                            flatPathCommon+"/JEC" + "/diboson/ZZ",                     12.14	],                  # 4         
    'ZJetsToQQ_200to400'      :         [nanoPathCommon + "/"+nanoFolder+"/ZJetsToQQ_HT-200to400*",                                                              flatPathCommon+"/JEC" + "/ZJets/ZJetsToQQ_HT-200to400" ,   1012.0],                    # 19    
    'ZJetsToQQ_400to600'      :         [nanoPathCommon + "/"+nanoFolder+"/ZJetsToQQ_HT-400to600*",                                                              flatPathCommon+"/JEC" + "/ZJets/ZJetsToQQ_HT-400to600" ,   114.2],                     # 20
    'ZJetsToQQ_600to800'      :         [nanoPathCommon + "/"+nanoFolder+"/ZJetsToQQ_HT-600to800*",                                                              flatPathCommon+"/JEC" + "/ZJets/ZJetsToQQ_HT-600to800" ,   25.34],                     # 21
    'ZJetsToQQ_800toInf'      :         [nanoPathCommon + "/"+nanoFolder+"/ZJetsToQQ_HT-800toInf*",                                                              flatPathCommon+"/JEC" + "/ZJets/ZJetsToQQ_HT-800toInf" ,   12.99],                     # 22
    'VBFHToBB'      :                   [nanoPathCommon + "/"+nanoFolder+"/VBFHToBB*/",                                                                          flatPathCommon+"/JEC" + "/VBFHToBB" ,                      3.766*0.58],                # 36    
    'GluGluHToBBMINLO'          :       [nanoPathCommon + "/"+nanoFolder+"/GluGluHToBB_M-125_TuneCP5_MINLO_NNLOPS_13TeV-powheg-pythia*/*/*/*/others",            flatPathCommon+"/JEC" + "/MINLOGluGluHToBB",               48.61*0.58],                # 37
    #'ZJetsToQQ_100to200'      :         [nanoPathCommon + "/"+nanoFolder+"/ZJetsToQQ_HT-100to200_TuneCP5_13TeV-madgraphMLM-pythia8",                             flatPathCommon+"/JEC" + "/ZJets/ZJetsToQQ_HT-100to200" ,   5.261e+03],                 # 35    
    #'GluGluHToBB_tr':                   [nanoPathCommon + "/"+nanoFolder+"/GluGluHToBB_M-125_TuneCP5_13TeV-powheg-pythia8/*/*/*/training",                      flatPathCommon+"/JEC" + "/GluGluHToBB_tr"    ,                48.61*0.58],                # 0 
    #'GluGluHToBBMINLO_tr'          :    [nanoPathCommon + "/"+nanoFolder+"/GluGluHToBB_M-125_TuneCP5_MINLO_NNLOPS_13TeV-powheg-pythia*/*/*/*/training",         flatPathCommon+"/JEC" + "/MINLOGluGluHToBB/training",               48.61*0.58],                # 37
    }


    processesMC_JEC = {}

    for jec_var in list_of_JEC_syst:
        for proc_name, (nano_path, flat_path, xsec) in processesMC_ref.items():
            # Create a new key with variation appended
            processJEC_name = f"{proc_name}_{jec_var}"
            # Modify the flat path accordingly
            new_flat_path = f"{flat_path}_{jec_var}"
            # Copy nano path and xsec as-is
            processesMC_JEC[processJEC_name] = [nano_path, new_flat_path, xsec]

    dfMC_JEC = pd.DataFrame(processesMC_JEC).T
    dfMC_JEC.columns = ['nanoPath', 'flatPath', 'xsection']
    dfMC_JEC = dfMC_JEC.reset_index()
    dfMC_JEC = dfMC_JEC.rename(columns={'index': 'process'})
    print(dfMC_JEC)
    dfMC_JEC.to_csv("/t3home/gcelotto/ggHbb/commonScripts/processesMC_JEC.csv")



    processesData ={
    'Data1A':                          [nanoPathCommon + "/DataFiducial_corrections2025Mar12/ParkingBPH1/crab_data_Run2018A_part1",                         flatPathCommon + "/Data1A"    ,                     0.774,     1017],                  # 0              
    'Data2A':                          [nanoPathCommon + "/DataFiducial_corrections2025Mar12/ParkingBPH2/crab_data_Run2018A_part2",                         flatPathCommon + "/Data2A"    ,                     0.774,     1017],                     # 1              
    'Data3A':                          [nanoPathCommon + "/DataFiducial_corrections2025Mar12/ParkingBPH3/crab_data_Run2018A_part3",                         flatPathCommon + "/Data3A"    ,                     0.774,     1018],                     # 3              
    'Data4A':                          [nanoPathCommon + "/DataFiducial_corrections2025Mar12/ParkingBPH4/crab_data_Run2018A_part4",                         flatPathCommon + "/Data4A"    ,                     0.774,     1016],                     # 4              
    'Data5A':                          [nanoPathCommon + "/DataFiducial_corrections2025Mar12/ParkingBPH5/crab_data_Run2018A_part5",                         flatPathCommon + "/Data5A"    ,                     0.774,     1016],                     # 5              
    'Data6A':                          [nanoPathCommon + "/DataFiducial_corrections2025Mar12/ParkingBPH6/crab_data_Run2018A_part6",                         flatPathCommon + "/Data6A"    ,                     0.774,     1017],                     # 6              
    
    
    'Data1B':                          [nanoPathCommon + "/DataBC_NonSmeared_2025Apr14/ParkingBPH1/crab_data_Run2018B_part1",                         flatPathCommon + "/Data1B"    ,                     0.911,     1024],                  # 8              
    'Data2B':                          [nanoPathCommon + "/DataBC_NonSmeared_2025Apr14/ParkingBPH2/crab_data_Run2018B_part2",                         flatPathCommon + "/Data2B"    ,                     0.911,     1021],                     # 9              
    'Data3B':                          [nanoPathCommon + "/DataBC_NonSmeared_2025Apr14/ParkingBPH3/crab_data_Run2018B_part3",                         flatPathCommon + "/Data3B" ,                        0.911,     1020],                     # 10
    'Data4B':                          [nanoPathCommon + "/DataBC_NonSmeared_2025Apr14/ParkingBPH4/crab_data_Run2018B_part4",                         flatPathCommon + "/Data4B"    ,                     0.911,     1020],                     # 11
    'Data5B':                          [nanoPathCommon + "/DataBC_NonSmeared_2025Apr14/ParkingBPH5/crab_data_Run2018B_part5",                         flatPathCommon + "/Data5B"    ,                     0.911,     1018],                     # 12
    'Data6B':                          [nanoPathCommon + "/DataBC_NonSmeared_2025Apr14/ParkingBPH6/crab_data_Run2018B_part6",                         flatPathCommon + "/Data6B"    ,                     0.377,     494],                     # 13

    'Data1C':                          [nanoPathCommon + "/DataBC_NonSmeared_2025Apr14/ParkingBPH1/crab_data_Run2018C_part1",                         flatPathCommon + "/Data1C"    ,                     1.103,     1105],                  # 14              
    'Data2C':                          [nanoPathCommon + "/DataBC_NonSmeared_2025Apr14/ParkingBPH2/crab_data_Run2018C_part2",                         flatPathCommon + "/Data2C"    ,                     1.103,     1108],                     # 15
    'Data3C':                          [nanoPathCommon + "/DataBC_NonSmeared_2025Apr14/ParkingBPH3/crab_data_Run2018C_part3",                         flatPathCommon + "/Data3C" ,                        1.103,     1102],                     # 16
    'Data4C':                          [nanoPathCommon + "/DataBC_NonSmeared_2025Apr14/ParkingBPH4/crab_data_Run2018C_part4",                         flatPathCommon + "/Data4C"    ,                     1.103,     1109],                     # 17
    'Data5C':                          [nanoPathCommon + "/DataBC_NonSmeared_2025Apr14/ParkingBPH5/crab_data_Run2018C_part5",                         flatPathCommon + "/Data5C"    ,                     1.103,     1105],                     # 18

    'Data1D':                          [nanoPathCommon + "/DataFiducial_corrections2025Mar12/ParkingBPH1/crab_data_Run2018D_part1",                         flatPathCommon + "/Data1D" ,                        5.302,     5508],                     # 2
    'Data2D':                          [nanoPathCommon + "/DataFiducial_corrections2025Mar12/ParkingBPH2/crab_data_Run2018D_part2",                         flatPathCommon + "/Data2D"    ,                     5.302,     5509],                     # 7              
    'Data3D':                          [nanoPathCommon + "/Data_D2025May21/ParkingBPH3/crab_data_Run2018D_part4",                         flatPathCommon + "/Data3D"    ,                     5.302,     5528],                     # 7              
    'Data4D':                          [nanoPathCommon + "/Data_D2025May21/ParkingBPH4/crab_data_Run2018D_part5",                         flatPathCommon + "/Data4D"    ,                     5.302,     5521],                     # 7              
    'Data5D':                          [nanoPathCommon + "/Data_D2025May21/ParkingBPH5/crab_data_Run2018D_part6",                         flatPathCommon + "/Data5D"    ,                     5.302,     5518],                     # 7              


    }
    dfData = pd.DataFrame(processesData).T
    dfData.columns = ['nanoPath', 'flatPath', 'lumi', 'nFiles']
    dfData = dfData.reset_index()
    dfData = dfData.rename(columns={'index': 'process'})
    
    print(dfData)
    dfData.to_csv("/t3home/gcelotto/ggHbb/commonScripts/processesData.csv")
    return dfMC, dfData
getProcessesDataFrame()