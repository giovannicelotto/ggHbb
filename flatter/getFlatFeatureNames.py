def getFlatFeatureNames(mc=True):
    featureToRemove = [
                    'jet1_btag_central',            'jet1_btag_down',           'jet1_btag_up',
                    'jet1_btag_down_jes',           'jet1_btag_down_pileup',    'jet1_btag_down_statistic', 'jet1_btag_down_type3',
                    'jet1_btag_up_jes',             'jet1_btag_up_pileup',      'jet1_btag_up_statistic',   'jet1_btag_up_type3',
                    'jet1_btag_down_correlated',    'jet1_btag_down_uncorrelated',
                    'jet1_btag_up_correlated',      'jet1_btag_up_uncorrelated',
                    'jet2_btag_central',   
    "down_cferr1", "down_cferr2", "down_hf", "down_hfstats1", "down_hfstats2",
    "down_jes", "down_jesAbsolute", "down_jesAbsoluteMPFBias", "down_jesAbsoluteScale",
    "down_jesAbsoluteStat", "down_jesAbsolute_2018", "down_jesBBEC1", "down_jesBBEC1_2018",
    "down_jesEC2", "down_jesEC2_2018", "down_jesFlavorQCD", "down_jesFragmentation",
    "down_jesHF", "down_jesHF_2018", "down_jesPileUpDataMC", "down_jesPileUpPtBB",
    "down_jesPileUpPtEC1", "down_jesPileUpPtEC2", "down_jesPileUpPtHF", "down_jesPileUpPtRef",
    "down_jesRelativeBal", "down_jesRelativeFSR", "down_jesRelativeJEREC1", "down_jesRelativeJEREC2",
    "down_jesRelativeJERHF", "down_jesRelativePtBB", "down_jesRelativePtEC1", "down_jesRelativePtEC2",
    "down_jesRelativePtHF", "down_jesRelativeSample", "down_jesRelativeSample_2018",
    "down_jesRelativeStatEC", "down_jesRelativeStatFSR", "down_jesRelativeStatHF",
    "down_jesSinglePionECAL", "down_jesSinglePionHCAL", "down_jesTimePtEta",
    "down_lf", "down_lfstats1", "down_lfstats2",
    "up_cferr1", "up_cferr2", "up_hf", "up_hfstats1", "up_hfstats2",
    "up_jes", "up_jesAbsolute", "up_jesAbsoluteMPFBias", "up_jesAbsoluteScale",
    "up_jesAbsoluteStat", "up_jesAbsolute_2018", "up_jesBBEC1", "up_jesBBEC1_2018",
    "up_jesEC2", "up_jesEC2_2018", "up_jesFlavorQCD", "up_jesFragmentation",
    "up_jesHF", "up_jesHF_2018", "up_jesPileUpDataMC", "up_jesPileUpPtBB",
    "up_jesPileUpPtEC1", "up_jesPileUpPtEC2", "up_jesPileUpPtHF", "up_jesPileUpPtRef",
    "up_jesRelativeBal", "up_jesRelativeFSR", "up_jesRelativeJEREC1", "up_jesRelativeJEREC2",
    "up_jesRelativeJERHF", "up_jesRelativePtBB", "up_jesRelativePtEC1", "up_jesRelativePtEC2",
    "up_jesRelativePtHF", "up_jesRelativeSample", "up_jesRelativeSample_2018",
    "up_jesRelativeStatEC", "up_jesRelativeStatFSR", "up_jesRelativeStatHF",
    "up_jesSinglePionECAL", "up_jesSinglePionHCAL", "up_jesTimePtEta",
    "up_lf", "up_lfstats1", "up_lfstats2",

        ]
    featureNames = [
                # Jet 1
                    'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass',
                    'jet1_nMuons','jet1_nTightMuons','jet1_nElectrons', 'jet1_btagDeepFlavB', 'jet1_btagPNetB', 'jet1_idx',
                    'jet1_rawFactor', 'jet1_PNetRegPtRawCorr', 'jet1_PNetRegPtRawCorrNeutrino',
                    'jet1_id', 'jet1_puId',
                    'jet1_btag_central',            'jet1_btag_down',           'jet1_btag_up',
                    #'jet1_btag_down_jes',           'jet1_btag_down_pileup',    'jet1_btag_down_statistic', 'jet1_btag_down_type3',
                    #'jet1_btag_up_jes',             'jet1_btag_up_pileup',      'jet1_btag_up_statistic',   'jet1_btag_up_type3',
                    #'jet1_btag_down_correlated',    'jet1_btag_down_uncorrelated',
                    #'jet1_btag_up_correlated',      'jet1_btag_up_uncorrelated'
                # Jet 2
                    'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass',
                    'jet2_nMuons', 'jet2_nTightMuons', 'jet2_nElectrons', 'jet2_btagDeepFlavB','jet2_btagPNetB','jet2_idx',
                    'jet2_rawFactor', 'jet2_PNetRegPtRawCorr', 'jet2_PNetRegPtRawCorrNeutrino',
                    'jet2_id', 'jet2_puId',
                    'jet2_btag_central',   
    #"down_cferr1", "down_cferr2", "down_hf", "down_hfstats1", "down_hfstats2",
    #"down_jes", "down_jesAbsolute", "down_jesAbsoluteMPFBias", "down_jesAbsoluteScale",
    #"down_jesAbsoluteStat", "down_jesAbsolute_2018", "down_jesBBEC1", "down_jesBBEC1_2018",
    #"down_jesEC2", "down_jesEC2_2018", "down_jesFlavorQCD", "down_jesFragmentation",
    #"down_jesHF", "down_jesHF_2018", "down_jesPileUpDataMC", "down_jesPileUpPtBB",
    #"down_jesPileUpPtEC1", "down_jesPileUpPtEC2", "down_jesPileUpPtHF", "down_jesPileUpPtRef",
    #"down_jesRelativeBal", "down_jesRelativeFSR", "down_jesRelativeJEREC1", "down_jesRelativeJEREC2",
    #"down_jesRelativeJERHF", "down_jesRelativePtBB", "down_jesRelativePtEC1", "down_jesRelativePtEC2",
    #"down_jesRelativePtHF", "down_jesRelativeSample", "down_jesRelativeSample_2018",
    #"down_jesRelativeStatEC", "down_jesRelativeStatFSR", "down_jesRelativeStatHF",
    #"down_jesSinglePionECAL", "down_jesSinglePionHCAL", "down_jesTimePtEta",
    #"down_lf", "down_lfstats1", "down_lfstats2",
    #"up_cferr1", "up_cferr2", "up_hf", "up_hfstats1", "up_hfstats2",
    #"up_jes", "up_jesAbsolute", "up_jesAbsoluteMPFBias", "up_jesAbsoluteScale",
    #"up_jesAbsoluteStat", "up_jesAbsolute_2018", "up_jesBBEC1", "up_jesBBEC1_2018",
    #"up_jesEC2", "up_jesEC2_2018", "up_jesFlavorQCD", "up_jesFragmentation",
    #"up_jesHF", "up_jesHF_2018", "up_jesPileUpDataMC", "up_jesPileUpPtBB",
    #"up_jesPileUpPtEC1", "up_jesPileUpPtEC2", "up_jesPileUpPtHF", "up_jesPileUpPtRef",
    #"up_jesRelativeBal", "up_jesRelativeFSR", "up_jesRelativeJEREC1", "up_jesRelativeJEREC2",
    #"up_jesRelativeJERHF", "up_jesRelativePtBB", "up_jesRelativePtEC1", "up_jesRelativePtEC2",
    #"up_jesRelativePtHF", "up_jesRelativeSample", "up_jesRelativeSample_2018",
    #"up_jesRelativeStatEC", "up_jesRelativeStatFSR", "up_jesRelativeStatHF",
    #"up_jesSinglePionECAL", "up_jesSinglePionHCAL", "up_jesTimePtEta",
    #"up_lf", "up_lfstats1", "up_lfstats2",

                # Jet 3
                    'jet3_pt', 'jet3_eta', 'jet3_phi', 'jet3_mass', 'jet3_nTightMuons',
                    'jet3_btagPNetB', 'jet3_btagDeepFlavB', 'dR_jet3_dijet',
                # Dijet
                    'dijet_pt', 'dijet_eta', 'dijet_phi', 'dijet_mass', 'dijet_dR', 'dijet_dEta', 'dijet_dPhi', 
                    'dijet_twist', 'dijet_cs', 'normalized_dijet_pt', 
                # PNet variables
                    'jet1_pt_pnet', 'jet2_pt_pnet', 
                    'dijet_pt_pnet', 'dijet_eta_pnet', 'dijet_phi_pnet', 'dijet_mass_pnet',
                # Event Variables
                    'nJets', 'nJets_20GeV', 'ht', 'nJets_30_btagMedWP', 'ttbar_tag', 'nSV',  # Error
                # Trig Muon
                    'muon_pt', 'muon_eta',  'muon_ptRel', 'muon_dxySig', 'muon_dzSig', 'muon_IP3d', 'muon_sIP3d', 'muon_tightId',
                    'muon_pfRelIso03_all', 'muon_pfRelIso04_all', 'muon_tkIsoId', 'muon_charge',
                # Trig Muon 2
                'leptonClass',
                    'muon2_pt', 'muon2_eta',  'muon2_dxySig', 'muon2_dzSig', 'muon2_IP3d', 'muon2_sIP3d', 'muon2_tightId',
                    'muon2_pfRelIso03_all', 'muon2_pfRelIso04_all', 'muon2_tkIsoId', 'muon2_charge',
                    'dimuon_mass',
                # Trigger Paths
                    'Muon_fired_HLT_Mu12_IP6',  'Muon_fired_HLT_Mu7_IP4',   'Muon_fired_HLT_Mu8_IP3',  'Muon_fired_HLT_Mu8_IP5',    'Muon_fired_HLT_Mu8_IP6',
                    'Muon_fired_HLT_Mu9_IP4',   'Muon_fired_HLT_Mu9_IP5',   'Muon_fired_HLT_Mu9_IP6',
                    'PV_npvs', 'Pileup_nTrueInt', 'sf']
    
    if mc == False:
        # We are in data remove scale factors
        filteredFeatureNames = [feature for feature in featureNames if feature not in featureToRemove]
        return filteredFeatureNames
    else:
        return featureNames



