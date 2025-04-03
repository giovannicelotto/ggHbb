def getFlatFeatureNames(mc=True):
    features_btag = [
                    'jet1_btag_central',            'jet1_btag_down',           'jet1_btag_up',
                    # see here if needed to divide into 4 categories:
                    # https://btv-wiki.docs.cern.ch/PerformanceCalibration/SFUncertaintiesAndCorrelations/
                    #'jet1_btag_down_jes',           'jet1_btag_down_pileup',    'jet1_btag_down_statistic', 'jet1_btag_down_type3',
                    #'jet1_btag_up_jes',             'jet1_btag_up_pileup',      'jet1_btag_up_statistic',   'jet1_btag_up_type3',
                    #'jet1_btag_down_correlated',    'jet1_btag_down_uncorrelated',
                    #'jet1_btag_up_correlated',      'jet1_btag_up_uncorrelated',
                    
                    'jet2_btag_central',            'jet2_btag_up',           'jet2_btag_down']
    features_JEC = [
    'jet1_sys_JECAbsoluteMPFBias_Down',     'jet1_sys_JECAbsoluteMPFBias_Up',     'jet1_sys_JECAbsoluteScale_Down',     'jet1_sys_JECAbsoluteScale_Up',     'jet1_sys_JECAbsoluteStat_Down',     'jet1_sys_JECAbsoluteStat_Up',     'jet1_sys_JECFlavorQCD_Down',     'jet1_sys_JECFlavorQCD_Up',     'jet1_sys_JECFragmentation_Down',     'jet1_sys_JECFragmentation_Up',     'jet1_sys_JECPileUpDataMC_Down',     'jet1_sys_JECPileUpDataMC_Up',     'jet1_sys_JECPileUpPtBB_Down',     'jet1_sys_JECPileUpPtBB_Up',     'jet1_sys_JECPileUpPtEC1_Down',     'jet1_sys_JECPileUpPtEC1_Up',     'jet1_sys_JECPileUpPtEC2_Down',     'jet1_sys_JECPileUpPtEC2_Up',     'jet1_sys_JECPileUpPtHF_Down',     'jet1_sys_JECPileUpPtHF_Up',     'jet1_sys_JECPileUpPtRef_Down',     'jet1_sys_JECPileUpPtRef_Up',     'jet1_sys_JECRelativeBal_Down',     'jet1_sys_JECRelativeBal_Up',     'jet1_sys_JECRelativeFSR_Down',     'jet1_sys_JECRelativeFSR_Up',     'jet1_sys_JECRelativeJEREC1_Down',     'jet1_sys_JECRelativeJEREC1_Up',     'jet1_sys_JECRelativeJEREC2_Down',     'jet1_sys_JECRelativeJEREC2_Up',     'jet1_sys_JECRelativeJERHF_Down',     'jet1_sys_JECRelativeJERHF_Up',     'jet1_sys_JECRelativePtBB_Down',     'jet1_sys_JECRelativePtBB_Up',     'jet1_sys_JECRelativePtEC1_Down',     'jet1_sys_JECRelativePtEC1_Up',     'jet1_sys_JECRelativePtEC2_Down',     'jet1_sys_JECRelativePtEC2_Up',     'jet1_sys_JECRelativePtHF_Down',     'jet1_sys_JECRelativePtHF_Up',     'jet1_sys_JECRelativeSample_Down',     'jet1_sys_JECRelativeSample_Up',     'jet1_sys_JECRelativeStatEC_Down',     'jet1_sys_JECRelativeStatEC_Up',     'jet1_sys_JECRelativeStatFSR_Down',     'jet1_sys_JECRelativeStatFSR_Up',     'jet1_sys_JECRelativeStatHF_Down',     'jet1_sys_JECRelativeStatHF_Up',     'jet1_sys_JECSinglePionECAL_Down',     'jet1_sys_JECSinglePionECAL_Up',     'jet1_sys_JECSinglePionHCAL_Down',     'jet1_sys_JECSinglePionHCAL_Up',     'jet1_sys_JECTimePtEta_Down',     'jet1_sys_JECTimePtEta_Up',
    'jet2_sys_JECAbsoluteMPFBias_Down',     'jet2_sys_JECAbsoluteMPFBias_Up',     'jet2_sys_JECAbsoluteScale_Down',     'jet2_sys_JECAbsoluteScale_Up',     'jet2_sys_JECAbsoluteStat_Down',     'jet2_sys_JECAbsoluteStat_Up',     'jet2_sys_JECFlavorQCD_Down',     'jet2_sys_JECFlavorQCD_Up',     'jet2_sys_JECFragmentation_Down',     'jet2_sys_JECFragmentation_Up',     'jet2_sys_JECPileUpDataMC_Down',     'jet2_sys_JECPileUpDataMC_Up',     'jet2_sys_JECPileUpPtBB_Down',     'jet2_sys_JECPileUpPtBB_Up',     'jet2_sys_JECPileUpPtEC1_Down',     'jet2_sys_JECPileUpPtEC1_Up',     'jet2_sys_JECPileUpPtEC2_Down',     'jet2_sys_JECPileUpPtEC2_Up',     'jet2_sys_JECPileUpPtHF_Down',     'jet2_sys_JECPileUpPtHF_Up',     'jet2_sys_JECPileUpPtRef_Down',     'jet2_sys_JECPileUpPtRef_Up',     'jet2_sys_JECRelativeBal_Down',     'jet2_sys_JECRelativeBal_Up',     'jet2_sys_JECRelativeFSR_Down',     'jet2_sys_JECRelativeFSR_Up',     'jet2_sys_JECRelativeJEREC1_Down',     'jet2_sys_JECRelativeJEREC1_Up',     'jet2_sys_JECRelativeJEREC2_Down',     'jet2_sys_JECRelativeJEREC2_Up',     'jet2_sys_JECRelativeJERHF_Down',     'jet2_sys_JECRelativeJERHF_Up',     'jet2_sys_JECRelativePtBB_Down',     'jet2_sys_JECRelativePtBB_Up',     'jet2_sys_JECRelativePtEC1_Down',     'jet2_sys_JECRelativePtEC1_Up',     'jet2_sys_JECRelativePtEC2_Down',     'jet2_sys_JECRelativePtEC2_Up',     'jet2_sys_JECRelativePtHF_Down',     'jet2_sys_JECRelativePtHF_Up',     'jet2_sys_JECRelativeSample_Down',     'jet2_sys_JECRelativeSample_Up',     'jet2_sys_JECRelativeStatEC_Down',     'jet2_sys_JECRelativeStatEC_Up',     'jet2_sys_JECRelativeStatFSR_Down',     'jet2_sys_JECRelativeStatFSR_Up',     'jet2_sys_JECRelativeStatHF_Down',     'jet2_sys_JECRelativeStatHF_Up',     'jet2_sys_JECSinglePionECAL_Down',     'jet2_sys_JECSinglePionECAL_Up',     'jet2_sys_JECSinglePionHCAL_Down',     'jet2_sys_JECSinglePionHCAL_Up',     'jet2_sys_JECTimePtEta_Down',     'jet2_sys_JECTimePtEta_Up'
    ]

    featureNames = [
                # Jet 1
                    'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass',
                    'jet1_nMuons','jet1_nConstituents','jet1_nTightMuons','jet1_nElectrons', 'jet1_btagDeepFlavB', 'jet1_btagTight',#'jet1_btagPNetB', 
                    'jet1_idx',
                    'jet1_rawFactor', 'jet1_bReg2018', #'jet1_PNetRegPtRawCorr', 'jet1_PNetRegPtRawCorrNeutrino',
                    'jet1_id', 'jet1_puId',
                    'jet1_sv_pt', 'jet1_sv_mass','jet1_sv_Ntrk', 'jet1_sv_3dSig',
                    'jet1_btag_central',            'jet1_btag_up',           'jet1_btag_down',
                    'jet1_sys_JECAbsoluteMPFBias_Down',     'jet1_sys_JECAbsoluteMPFBias_Up',     'jet1_sys_JECAbsoluteScale_Down',     'jet1_sys_JECAbsoluteScale_Up',     'jet1_sys_JECAbsoluteStat_Down',     'jet1_sys_JECAbsoluteStat_Up',     'jet1_sys_JECFlavorQCD_Down',     'jet1_sys_JECFlavorQCD_Up',     'jet1_sys_JECFragmentation_Down',     'jet1_sys_JECFragmentation_Up',     'jet1_sys_JECPileUpDataMC_Down',     'jet1_sys_JECPileUpDataMC_Up',     'jet1_sys_JECPileUpPtBB_Down',     'jet1_sys_JECPileUpPtBB_Up',     'jet1_sys_JECPileUpPtEC1_Down',     'jet1_sys_JECPileUpPtEC1_Up',     'jet1_sys_JECPileUpPtEC2_Down',     'jet1_sys_JECPileUpPtEC2_Up',     'jet1_sys_JECPileUpPtHF_Down',     'jet1_sys_JECPileUpPtHF_Up',     'jet1_sys_JECPileUpPtRef_Down',     'jet1_sys_JECPileUpPtRef_Up',     'jet1_sys_JECRelativeBal_Down',     'jet1_sys_JECRelativeBal_Up',     'jet1_sys_JECRelativeFSR_Down',     'jet1_sys_JECRelativeFSR_Up',     'jet1_sys_JECRelativeJEREC1_Down',     'jet1_sys_JECRelativeJEREC1_Up',     'jet1_sys_JECRelativeJEREC2_Down',     'jet1_sys_JECRelativeJEREC2_Up',     'jet1_sys_JECRelativeJERHF_Down',     'jet1_sys_JECRelativeJERHF_Up',     'jet1_sys_JECRelativePtBB_Down',     'jet1_sys_JECRelativePtBB_Up',     'jet1_sys_JECRelativePtEC1_Down',     'jet1_sys_JECRelativePtEC1_Up',     'jet1_sys_JECRelativePtEC2_Down',     'jet1_sys_JECRelativePtEC2_Up',     'jet1_sys_JECRelativePtHF_Down',     'jet1_sys_JECRelativePtHF_Up',     'jet1_sys_JECRelativeSample_Down',     'jet1_sys_JECRelativeSample_Up',     'jet1_sys_JECRelativeStatEC_Down',     'jet1_sys_JECRelativeStatEC_Up',     'jet1_sys_JECRelativeStatFSR_Down',     'jet1_sys_JECRelativeStatFSR_Up',     'jet1_sys_JECRelativeStatHF_Down',     'jet1_sys_JECRelativeStatHF_Up',     'jet1_sys_JECSinglePionECAL_Down',     'jet1_sys_JECSinglePionECAL_Up',     'jet1_sys_JECSinglePionHCAL_Down',     'jet1_sys_JECSinglePionHCAL_Up',     'jet1_sys_JECTimePtEta_Down',     'jet1_sys_JECTimePtEta_Up',
                # Jet 2
                    'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass',
                    'jet2_nMuons', 'jet2_nConstituents','jet2_nTightMuons', 'jet2_nElectrons', 'jet2_btagDeepFlavB','jet2_btagTight',#'jet2_btagPNetB',
                    'jet2_idx',
                    'jet2_rawFactor', 'jet2_bReg2018',#'jet2_PNetRegPtRawCorr', 'jet2_PNetRegPtRawCorrNeutrino',
                    'jet2_id', 'jet2_puId',
                    'jet2_sv_pt','jet2_sv_mass', 'jet2_sv_Ntrk','jet2_sv_3dSig',
                    'jet2_btag_central',            'jet2_btag_up',           'jet2_btag_down',
                    'jet2_sys_JECAbsoluteMPFBias_Down',     'jet2_sys_JECAbsoluteMPFBias_Up',     'jet2_sys_JECAbsoluteScale_Down',     'jet2_sys_JECAbsoluteScale_Up',     'jet2_sys_JECAbsoluteStat_Down',     'jet2_sys_JECAbsoluteStat_Up',     'jet2_sys_JECFlavorQCD_Down',     'jet2_sys_JECFlavorQCD_Up',     'jet2_sys_JECFragmentation_Down',     'jet2_sys_JECFragmentation_Up',     'jet2_sys_JECPileUpDataMC_Down',     'jet2_sys_JECPileUpDataMC_Up',     'jet2_sys_JECPileUpPtBB_Down',     'jet2_sys_JECPileUpPtBB_Up',     'jet2_sys_JECPileUpPtEC1_Down',     'jet2_sys_JECPileUpPtEC1_Up',     'jet2_sys_JECPileUpPtEC2_Down',     'jet2_sys_JECPileUpPtEC2_Up',     'jet2_sys_JECPileUpPtHF_Down',     'jet2_sys_JECPileUpPtHF_Up',     'jet2_sys_JECPileUpPtRef_Down',     'jet2_sys_JECPileUpPtRef_Up',     'jet2_sys_JECRelativeBal_Down',     'jet2_sys_JECRelativeBal_Up',     'jet2_sys_JECRelativeFSR_Down',     'jet2_sys_JECRelativeFSR_Up',     'jet2_sys_JECRelativeJEREC1_Down',     'jet2_sys_JECRelativeJEREC1_Up',     'jet2_sys_JECRelativeJEREC2_Down',     'jet2_sys_JECRelativeJEREC2_Up',     'jet2_sys_JECRelativeJERHF_Down',     'jet2_sys_JECRelativeJERHF_Up',     'jet2_sys_JECRelativePtBB_Down',     'jet2_sys_JECRelativePtBB_Up',     'jet2_sys_JECRelativePtEC1_Down',     'jet2_sys_JECRelativePtEC1_Up',     'jet2_sys_JECRelativePtEC2_Down',     'jet2_sys_JECRelativePtEC2_Up',     'jet2_sys_JECRelativePtHF_Down',     'jet2_sys_JECRelativePtHF_Up',     'jet2_sys_JECRelativeSample_Down',     'jet2_sys_JECRelativeSample_Up',     'jet2_sys_JECRelativeStatEC_Down',     'jet2_sys_JECRelativeStatEC_Up',     'jet2_sys_JECRelativeStatFSR_Down',     'jet2_sys_JECRelativeStatFSR_Up',     'jet2_sys_JECRelativeStatHF_Down',     'jet2_sys_JECRelativeStatHF_Up',     'jet2_sys_JECSinglePionECAL_Down',     'jet2_sys_JECSinglePionECAL_Up',     'jet2_sys_JECSinglePionHCAL_Down',     'jet2_sys_JECSinglePionHCAL_Up',     'jet2_sys_JECTimePtEta_Down',     'jet2_sys_JECTimePtEta_Up',
                # Jet 3
                    'jet3_pt', 'jet3_eta', 'jet3_phi', 'jet3_mass', 'jet3_nTightMuons',
                    #'jet3_btagPNetB',
                    'jet3_btagDeepFlavB', 'dR_jet3_dijet',
                # Dijet
                    'dijet_pt', 'dijet_eta', 'dijet_phi', 'dijet_mass', 'dijet_dR', 'dijet_dEta', 'dijet_dPhi', 
                    'dijet_twist', 'dijet_cs', 'normalized_dijet_pt', 'cos_theta_star',
                    'dijet_pTAsymmetry', 'centrality', #'sphericity_T',
                    'sphericity',#'planarity',
                    'lambda1','lambda2','lambda3',
                    'T_max', 'phiT_max',
                # uncorrected
                    'jet1_pt_uncor', 'jet2_pt_uncor',
                # PNet variables
                    #'jet1_pt_pnet', 'jet2_pt_pnet', 
                    #'dijet_pt_pnet', 'dijet_eta_pnet', 'dijet_phi_pnet', 'dijet_mass_pnet',
                # Event Variables
                    'nJets', 'nJets_20',  'nJets_30', 'nJets_50', 'ht', 
                    'nMuons', 'nIsoMuons', 'nElectrons', #'nProbeTracks', 
                    #'nJet20_L', 'nJet20_M', 'nJet20_T',
                    'nSV',  # Error
                # Trig Muon
                    'muon_pt', 'muon_eta',  'muon_ptRel', 'muon_dxySig', 'muon_dzSig', 'muon_IP3d', 'muon_sIP3d', 'muon_tightId',
                    'muon_pfRelIso03_all', 'muon_pfRelIso04_all', 'muon_tkIsoId', 'muon_charge',
                # Trig Muon 2
                'leptonClass',
                    'muon2_pt', 'muon2_eta',  'muon2_dxySig', 'muon2_dzSig', 'muon2_IP3d', 'muon2_sIP3d', 'muon2_tightId',
                    'muon2_pfRelIso03_all', 'muon2_pfRelIso04_all', 'muon2_tkIsoId', 'muon2_charge',
                    'dimuon_mass',
                # Trigger Paths
                    'Muon_fired_HLT_Mu12_IP6', 'Muon_fired_HLT_Mu10p5_IP3p5','Muon_fired_HLT_Mu8p5_IP3p5', 'Muon_fired_HLT_Mu7_IP4',   'Muon_fired_HLT_Mu8_IP3',  'Muon_fired_HLT_Mu8_IP5',    'Muon_fired_HLT_Mu8_IP6',
                    'Muon_fired_HLT_Mu9_IP4',   'Muon_fired_HLT_Mu9_IP5',   'Muon_fired_HLT_Mu9_IP6',
                    'PV_npvs', 'Pileup_nTrueInt', 'sf', 'genWeight'
                    ]
    
    if mc == False:
        # We are in data remove scale factors
        filteredFeatureNames = [feature for feature in featureNames if feature not in features_btag]
        filteredFeatureNames = [feature for feature in filteredFeatureNames if feature not in features_JEC]
        return filteredFeatureNames
    else:
        return featureNames
    





