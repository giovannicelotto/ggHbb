import numpy as np
def getFeatures(outFolder=None, massHypo=False):
   featuresForTraining=[   
    'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass',
    #'jet1_nMuons',
    'jet1_nTightMuons',#'jet1_nElectrons',
    'jet1_btagDeepFlavB', 'jet1_idx', 'jet1_puId',
    
    'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass',
    #'jet2_nMuons',
    'jet2_nTightMuons', #'jet2_nElectrons',
    'jet2_btagDeepFlavB', 'jet2_idx', 'jet2_puId',
    
    'jet3_pt', 'jet3_eta', 'jet3_phi', 'jet3_mass', 'jet3_nTightMuons',
    'jet3_btagDeepFlavB', 'dR_jet3_dijet',

    'dijet_pt', 'dijet_eta', 'dijet_phi', 'dijet_mass', 'dijet_dR',
    #'dijet_dEta', 'dijet_dPhi', 
    'dijet_twist', 'dijet_cs', 'normalized_dijet_pt', 

    'nJets', 'nJets_20GeV', 'nSV', 'ht',

    'muon_pt', 'muon_eta',  'muon_ptRel', 'muon_dxySig', 'muon_dzSig', 'muon_IP3d', 'muon_sIP3d', 'muon_tightId',
   'muon_pfRelIso03_all', 'muon_pfRelIso04_all', 'muon_tkIsoId', # 'muon_charge',
   #'PU_SF',
   # 'sf'
   ]
   
    
   columnsToRead = [   
    'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass',
    'jet1_nMuons', 'jet1_nTightMuons', 'jet1_nElectrons', 'jet1_btagDeepFlavB', 'jet1_idx','jet1_puId',
    
    'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass',
    'jet2_nMuons','jet2_nTightMuons', 'jet2_nElectrons', 'jet2_btagDeepFlavB', 'jet2_idx','jet2_puId',
    
    'jet3_pt', 'jet3_eta', 'jet3_phi', 'jet3_mass', 'jet3_nTightMuons',
    'jet3_btagDeepFlavB', 'dR_jet3_dijet',

    'dijet_pt', 'dijet_eta', 'dijet_phi', 'dijet_mass', 'dijet_dR', 'dijet_dEta', 'dijet_dPhi', 
    'dijet_twist', 'dijet_cs', 'normalized_dijet_pt', 

    'nJets', 'nJets_20GeV', 'nSV', 'ht',

    'muon_pt', 'muon_eta',  'muon_ptRel', 'muon_dxySig', 'muon_dzSig', 'muon_IP3d', 'muon_sIP3d', 'muon_tightId',
   'muon_pfRelIso03_all', 'muon_pfRelIso04_all', 'muon_tkIsoId', # 'muon_charge',
   'PU_SF',
    'sf']
   
   if outFolder is not None:
      np.save(outFolder+"/model/featuresForTraining.npy", featuresForTraining)
   if massHypo==True:
      featuresForTraining = featuresForTraining + ['massHypo']
   return featuresForTraining, columnsToRead