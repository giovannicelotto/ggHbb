import numpy as np
def getFeatures(inFolder):
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
    'dijet_cs', 'normalized_dijet_pt',

    'sf']
    np.save(inFolder+"/featuresForTraining.npy", featuresForTraining)
    return featuresForTraining, columnsToRead