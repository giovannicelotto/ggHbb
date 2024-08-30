def getFeatures(leptonClass=0):
    featuresForTraining=[
       #'jet1_pt',
        'jet1_eta',
        'jet1_phi',
       #'jet1_mass',
        #'jet1_nMuons',
        'jet1_nTightMuons',
       #'jet1_nElectrons',
        'jet1_btagDeepFlavB', #'jet1_area',
        'jet1_qgl',
       #'jet2_pt',
        'jet2_eta',
        'jet2_phi',
       # 'jet2_mass',
       #'jet2_nMuons',
        'jet2_nTightMuons',
       #'jet2_nElectrons',
        'jet2_btagDeepFlavB', #'jet2_area',
        'jet2_qgl',
        'jet3_pt',
        'jet3_eta', 'jet3_phi',
        'jet3_mass',
        'normalized_dijet_pt',
       #'dijet_pt',
    #'dijet_eta', 'dijet_phi',
    #   'dijet_mass',
    #   'dijet_dR',
    #   'dijet_dEta', 'dijet_dPhi',
    #   'dijet_twist',# 'nJets',
       'nJets_20GeV',
    #   'ht',
    #'muon_pt',
    #   'muon_eta',
       'muon_dxySig',  
       'muon_dzSig',
       'muon_IP3d',
       'muon_sIP3d',
       'leptonClass',
       'dijet_cs',
       'nSV',
       'muon_pfRelIso03_all',
       'muon_tkIsoId',
       'muon_tightId',
       ]
    
    columnsToRead = [   
    'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass', 'jet1_nMuons', 'jet1_nTightMuons',
    'jet1_nElectrons', 'jet1_btagDeepFlavB', 'jet1_area', 'jet1_qgl',
    'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass', 'jet2_nMuons','jet2_nTightMuons',
    'jet2_nElectrons', 'jet2_btagDeepFlavB', 'jet2_area', 'jet2_qgl',
    'jet3_pt', 'jet3_eta', 'jet3_phi', 'jet3_mass',
    'dijet_pt', 'dijet_eta', 'dijet_phi', 'dijet_mass', 'dijet_dR',
    'dijet_dEta', 'dijet_dPhi', 'dijet_angVariable', 'dijet_twist', 'nJets',
    'nJets_20GeV', 'nSV',
    'ht', 'muon_pt', 'muon_eta', 'muon_dxySig', 'muon_dzSig', 'muon_IP3d',
    'muon_sIP3d', 'muon_tightId', 'muon_pfRelIso03_all', 'muon_tkIsoId',
    'leptonClass',
    'dijet_cs', 'normalized_dijet_pt',

    'sf']

    if leptonClass==0:
        pass
    elif leptonClass == 1:
        features_2 = ['muon2_pt', 'muon2_eta',  'muon2_dxySig', 'muon2_dzSig', 'muon2_IP3d', 'muon2_sIP3d', 'muon2_tightId',
        'muon2_pfRelIso03_all', 'muon2_pfRelIso04_all', 'muon2_tkIsoId', 'muon2_charge', 'dimuon_mass']
        #columnsToRead = columnsToRead + features_2
        #featuresForTraining = featuresForTraining + features_2

    elif leptonClass == 2:
        features_2 = ['muon2_pt', 'muon2_eta',  'muon2_dxySig', 'muon2_dzSig', 'muon2_IP3d', 'muon2_sIP3d', 'muon2_tightId',
        'muon2_pfRelIso03_all', 'muon2_pfRelIso04_all', 'muon2_tkIsoId', 'muon2_charge', 'dimuon_mass']
        #columnsToRead = columnsToRead + features_2
        #featuresForTraining = featuresForTraining + features_2
    elif leptonClass == 3:
        pass
    
    return featuresForTraining, columnsToRead