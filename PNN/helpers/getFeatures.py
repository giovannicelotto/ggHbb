import numpy as np
def getFeatures(outFolder=None, massHypo=False, bin_center=False, simple=False):
   featuresForTraining=[   
   'jet1_pt',
   #'jet1_eta', 'jet1_phi', 
   'jet1_mass',
    #'jet1_nMuons',
    'jet1_nTightMuons',#'jet1_nElectrons',
    #'jet1_btagDeepFlavB',
    #'jet1_idx', #'jet1_puId',
    'jet1_sv_pt',
   'jet1_sv_mass',
   'jet1_sv_Ntrk',
   'jet1_sv_3dSig',
   'jet1_nConstituents',
    
    'jet2_pt', #'jet2_eta', 'jet2_phi',
    'jet2_mass',
    #'jet2_nMuons',
    'jet2_nTightMuons', #'jet2_nElectrons',
    'jet2_btagDeepFlavB', #'jet2_idx', 'jet2_puId',

    'jet2_sv_pt',
   'jet2_sv_mass',
   'jet2_sv_Ntrk',
   'jet2_sv_3dSig',
   'jet2_nConstituents',
    
    #'jet3_pt', 'jet3_eta', 'jet3_phi', 'jet3_mass', 'jet3_nTightMuons',
    #'jet3_btagDeepFlavB',
    #'dR_jet3_dijet',

    'dijet_pt',
    #'dijet_eta', 'dijet_phi', #
    #'dijet_mass',
    #'dijet_dR',
    'dijet_dEta', 'dijet_dPhi', 
    'dijet_twist', 'dijet_cs', 'normalized_dijet_pt', 
    'cos_theta_star',
   'dijet_pTAsymmetry',
   'centrality',
   'sphericity',
   'lambda1',
   'lambda2',
   'lambda3',
   'T_max',
   'phiT_max',

    'nJets', 'nSV', 'ht',
      'nMuons',
      'nIsoMuons',
      'nElectrons',
      #'nJet20_L',
      #'nJet20_M',
      #'nJet20_T',
      'nJets_20',
      'nJets_30',
      'nJets_50',

    #'nJets',

    'muon_pt',#'muon_eta',  #'muon_ptRel',
    'muon_dxySig', 'muon_dzSig', #'muon_IP3d', 'muon_sIP3d',
    #'muon_tightId',
    'muon_pfRelIso03_all', #'muon_pfRelIso04_all', 'muon_tkIsoId', # 'muon_charge',
    'leptonClass',
   #'PU_SF',
   # 'sf'
   ]
   
    
   columnsToRead = [   
    'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass',
    'jet1_nMuons', 'jet1_nTightMuons', 'jet1_nElectrons', 'jet1_btagDeepFlavB', 'jet1_idx','jet1_puId',

   'jet1_sv_pt',
   'jet1_sv_mass',
   'jet1_sv_Ntrk',
   'jet1_sv_3dSig',
   'jet1_nConstituents',
    
    'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass',
    'jet2_nMuons','jet2_nTightMuons', 'jet2_nElectrons', 'jet2_btagDeepFlavB', 'jet2_idx','jet2_puId',

   'jet2_sv_pt',
   'jet2_sv_mass',
   'jet2_sv_Ntrk',
   'jet2_sv_3dSig',
   'jet2_nConstituents',
    
    'jet3_pt', 'jet3_eta', 'jet3_phi', 'jet3_mass', 'jet3_nTightMuons',
    #'jet3_btagDeepFlavB',
    'dR_jet3_dijet',

    'dijet_pt', 'dijet_eta', 'dijet_phi', 'dijet_mass', 'dijet_dR', 'dijet_dEta', 'dijet_dPhi', 
    'dijet_twist', 'dijet_cs', 'normalized_dijet_pt', 

   'cos_theta_star',
   'dijet_pTAsymmetry',
   'centrality',
   'sphericity',
   'lambda1',
   'lambda2',
   'lambda3',
   'T_max',
   'phiT_max',

    'nJets', 'nSV', 'ht',
      'nMuons',
      'nIsoMuons',
      'nElectrons',
      #'nJet20_L',
      #'nJet20_M',
      #'nJet20_T',
      'nJets_20',
      'nJets_30',
      'nJets_50',

    'muon_pt', 'muon_eta',  'muon_ptRel', 'muon_dxySig', 'muon_dzSig', 'muon_IP3d', 'muon_sIP3d', 'muon_tightId',
   'muon_pfRelIso03_all', 'muon_pfRelIso04_all', 'muon_tkIsoId', # 'muon_charge',
   'leptonClass',
   'PU_SF',
    'sf']
   if simple==True:
      # For non boosted topology
      featuresForTraining=[   
   'jet1_pt',
   'jet1_eta',
   'jet1_btagTight',
   'jet1_sv_pt',
   'jet1_sv_Ntrk',
   'jet1_sv_3dSig',
   'jet1_nConstituents',

   'jet2_pt',
   'jet2_eta',
   'jet2_btagTight',
   'jet2_sv_pt',
   'jet2_sv_mass',
   'jet2_sv_Ntrk',
   'jet2_nConstituents',

##
   'dijet_pt',
   'dijet_dEta', 'dijet_dPhi', 
   'dijet_twist', 'dijet_cs', 
   'dijet_mass',
   'cos_theta_star',
   'dijet_pTAsymmetry',
   'centrality',

   'lambda1',
   'lambda2',
   'lambda3',

   'phiT_max',

   'nSV',
   'ht',
   'nMuons',
   'nIsoMuons',

   'nJets_20',
   'nJets_30',
   'nJets_50',
   'muon_pt',
   'muon_pfRelIso03_all',

   ]
      print("Simpe Training is True")
      print("Following features have been used", featuresForTraining)
   if massHypo==True:
      featuresForTraining = featuresForTraining + ['massHypo']
      print("massHypo added to the features")
   if bin_center:
      featuresForTraining = featuresForTraining + ['bin_center']
      print("bin_center added to the features")
   if outFolder is not None:
      np.save(outFolder+"/featuresForTraining.npy", featuresForTraining)

   print("-"*50)
   return featuresForTraining, columnsToRead


def getFeaturesHighPt(outFolder=None, massHypo=False):
   featuresForTraining=[   
   'jet1_pt',       'jet1_mass',    'jet1_nTightMuons',     'jet1_sv_pt',
   'jet1_sv_mass',  'jet1_sv_Ntrk', 'jet1_sv_3dSig',        'jet1_nConstituents',
    'jet2_pt',      'jet2_mass',    'jet2_nTightMuons',     'jet2_sv_pt',
   'jet2_sv_mass',  'jet2_sv_Ntrk', 'jet2_sv_3dSig',        'jet2_nConstituents',
    'dijet_pt',     'dijet_dEta',   'dijet_dPhi',           'dijet_twist',
    'dijet_cs',     'normalized_dijet_pt',                  'dijet_pTAsymmetry',
    'cos_theta_star', 
   'centrality',    'sphericity', 'lambda1',
   'lambda2',      'lambda3',       'T_max',
   'phiT_max',
    'nJets', 'nSV', 'ht',
      'nMuons', 'nIsoMuons', 'nElectrons',
      'nJets_20', 'nJets_30','nJets_50',
    'muon_pt','muon_ptRel', 'muon_dxySig', 'muon_dzSig', 'muon_pfRelIso03_all', 
    'leptonClass',
   ]
   columnsToRead = [   
    'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass',
    'jet1_nMuons', 'jet1_nTightMuons', 'jet1_nElectrons', 'jet1_btagDeepFlavB', 'jet1_idx','jet1_puId',

   'jet1_sv_pt',
   'jet1_sv_mass',
   'jet1_sv_Ntrk',
   'jet1_sv_3dSig',
   'jet1_nConstituents',
    
    'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass',
    'jet2_nMuons','jet2_nTightMuons', 'jet2_nElectrons', 'jet2_btagDeepFlavB', 'jet2_idx','jet2_puId',

   'jet2_sv_pt',
   'jet2_sv_mass',
   'jet2_sv_Ntrk',
   'jet2_sv_3dSig',
   'jet2_nConstituents',
    
   'jet3_nTightMuons',  'dR_jet3_dijet',
    'dijet_pt', 'dijet_eta', 'dijet_phi', 'dijet_mass', 'dijet_dR', 'dijet_dEta', 'dijet_dPhi', 
    'dijet_twist', 'dijet_cs', 'normalized_dijet_pt',    'cos_theta_star',   'dijet_pTAsymmetry',
   'centrality',   'sphericity',   'lambda1',   'lambda2',   'lambda3',   'T_max',   'phiT_max',    'nJets', 'nSV', 'ht',      'nMuons',      'nIsoMuons',      'nElectrons',
      'nJets_20',      'nJets_30',      'nJets_50',
    'muon_pt', 'muon_eta',  'muon_ptRel', 'muon_dxySig', 'muon_dzSig', 'muon_IP3d', 'muon_sIP3d', 'muon_tightId',   'muon_pfRelIso03_all', 'muon_pfRelIso04_all', 'muon_tkIsoId', 'leptonClass',
   'PU_SF',    'sf']
   if massHypo==True:
      featuresForTraining = featuresForTraining + ['massHypo']
      print("massHypo added to the features")
   if outFolder is not None:
      np.save(outFolder+"/featuresForTraining.npy", featuresForTraining)

   print("-"*50)
   return featuresForTraining, columnsToRead