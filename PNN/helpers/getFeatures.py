import numpy as np
def getFeatures(outFolder=None,  ):
   featuresForTraining=[   
   'jet1_pt',       'jet1_eta', 'jet1_mass', 'jet1_phi',       
   'jet1_btagTight',
   'jet1_btagDeepFlavB',
   'jet1_sv_Ntrk', 'jet1_sv_3dSig',        'jet1_nConstituents', 'jet1_nTightMuons', 'jet1_nElectrons',
   #'jet1_leadTrackPt',

   'jet2_pt',       'jet2_eta', 'jet2_mass', 'jet2_phi',       
   'jet2_btagTight',
   'jet2_btagDeepFlavB',
   'jet2_sv_Ntrk', 'jet2_sv_3dSig',        'jet2_nConstituents', 'jet2_nTightMuons', 'jet2_nElectrons',
   #'jet2_leadTrackPt',

   'jet3_pt',  'jet3_eta', 'jet3_phi', 'jet3_mass',
   'jet3_btagWP',
   'jet3_nTightMuons', 
   #'jet3_leadTrackPt',

   #'jet1_rest_pt',   'jet1_rest_eta',    'jet1_rest_phi',
   #'jet2_rest_pt',   'jet2_rest_eta',  'jet2_rest_phi',
   
   'dR_jet3_dijet', 'dPhi_jet3_dijet',
   
   'dijet_pt',    'dijet_eta',   'dijet_phi', 'dijet_mass', 
   'dijet_dR',    'dijet_dEta',   'dijet_dPhi',  'dijet_twist',
   'dijet_cs',     'normalized_dijet_pt',              
   'cos_theta_star', 'dijet_pTAsymmetry',
   'centrality',    'sphericity', 
   'lambda1', 'lambda2',      'lambda3',  #'T_max',
   #'phiT_max',
    #'nJets',
    #'nSV',
   #'ht',
   #'nMuons', 'nIsoMuons', 'nElectrons',
   #'nJets_20', 'nJets_30','nJets_50',
   'muon_pt','muon_eta','muon_phi',#'muon_mass',# 'muon_ptRel', 'muon_dxySig', 'muon_dzSig', 'muon_pfRelIso03_all', 
   #'leptonClass', 'dimuon_mass'
   ]
   columnsToRead = featuresForTraining + ['PU_SF',    'sf', 'genWeight','btag_central', 
                                           'b_gen_pt', 'b_gen_eta', 'b_gen_phi', 'b_gen_mass', 'antib_gen_pt', 'antib_gen_eta', 'antib_gen_phi', 'antib_gen_mass', 'higgs_gen_pt', 'higgs_gen_eta',
                                          'higgs_gen_phi', 'higgs_gen_mass',
                                          'dR_jet1_genQuark', 'dR_jet2_genQuark', 'dpT_jet1_genQuark', 'dpT_jet2_genQuark']
   if outFolder is not None:
      np.save(outFolder+"/featuresForTraining.npy", featuresForTraining)

   print("-"*50)
   return featuresForTraining, columnsToRead


def getFeaturesHighPt(outFolder=None):
   featuresForTraining=[   
   'jet1_pt',       'jet1_eta', 'jet1_mass', 'jet1_phi',       
   'jet1_btagTight',
   'jet1_sv_Ntrk', 'jet1_sv_3dSig',        'jet1_nConstituents', 'jet1_nTightMuons', 'jet1_nElectrons',
   'jet1_leadTrackPt',

   'jet2_pt',       'jet2_eta', 'jet2_mass', 'jet2_phi',       
   'jet2_btagTight',
   'jet2_sv_Ntrk', 'jet2_sv_3dSig',        'jet2_nConstituents', 'jet2_nTightMuons', 'jet2_nElectrons',
   'jet2_leadTrackPt',

   'jet3_pt',  'jet3_eta', 'jet3_phi', 'jet3_mass',
   'jet3_btagWP',
   'jet3_nTightMuons', 
   'jet3_leadTrackPt',

   'jet1_rest_pt',   'jet1_rest_eta',    'jet1_rest_phi',
   'jet2_rest_pt',   'jet2_rest_eta',  'jet2_rest_phi',
   
   'dR_jet3_dijet', 'dPhi_jet3_dijet',
   
   'dijet_pt',    'dijet_eta',   'dijet_phi', 'dijet_mass', 
   'dijet_dR',    'dijet_dEta',   'dijet_dPhi',  'dijet_twist',
   'dijet_cs',     'normalized_dijet_pt',              
   'cos_theta_star', 'dijet_pTAsymmetry',
   'centrality',    'sphericity', 
   'lambda1', 'lambda2',      'lambda3',  #'T_max',
   #'phiT_max',
    #'nJets',
    #'nSV',
   #'ht',
   #'nMuons', 'nIsoMuons', 'nElectrons',
   'nJets_20', 'nJets_30','nJets_50',
   'muon_pt','muon_eta','muon_phi',#'muon_mass',# 'muon_ptRel', 'muon_dxySig', 'muon_dzSig', 'muon_pfRelIso03_all', 
   'leptonClass', 'dimuon_mass'
   ]
   columnsToRead = featuresForTraining + ['PU_SF',    'sf', 'genWeight','btag_central', 
                                           'b_gen_pt', 'b_gen_eta', 'b_gen_phi', 'b_gen_mass', 'antib_gen_pt', 'antib_gen_eta', 'antib_gen_phi', 'antib_gen_mass', 'higgs_gen_pt', 'higgs_gen_eta',
                                          'higgs_gen_phi', 'higgs_gen_mass',
                                          'dR_jet1_genQuark', 'dR_jet2_genQuark', 'dpT_jet1_genQuark', 'dpT_jet2_genQuark']
   if outFolder is not None:
      np.save(outFolder+"/featuresForTraining.npy", featuresForTraining)

   print("-"*50)
   return featuresForTraining, columnsToRead