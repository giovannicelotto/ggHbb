import numpy as np

plot_vars = {
    "jet1_pt": {
        "bins": np.linspace(0, 200, 41),
        "xlabel": "jet1_pt [GeV]",
    },
    "jet1_eta": {
        "bins": np.linspace(-2, 2, 41),
        "xlabel": "jet1_eta",
    },
    "jet1_phi": {
        "bins": np.linspace(-3.14, 3.14, 21),
        "xlabel": "jet1_phi",
    },
    "jet1_btagWP": {
        "bins": np.linspace(0, 4, 21),
        "xlabel": "jet1_btagWP",
    },
    "jet1_btagDeepFlavB": {
        "bins": np.linspace(0, 1, 21),
        "xlabel": "jet1_btagDeepFlavB",
    },
    "jet1_nMuons": {
        "bins": np.linspace(0, 3, 21),
        "xlabel": "jet1_nMuons",
    },
    "jet1_nConstituents": {
        "bins": np.linspace(0, 50, 51),
        "xlabel": "jet1_nConstituents",
    },
    "jet1_leadTrackPt": {
        "bins": np.linspace(0, 40, 21),
        "xlabel": "jet1_leadTrackPt",
    },
    "jet1_nElectrons": {
        "bins": np.linspace(0, 4, 21),
        "xlabel": "jet1_nElectrons",
    },
    "jet1_btagTight": {
        "bins": np.linspace(0, 1, 21),
        "xlabel": "jet1_btagTight",
    },
    "jet1_sv_pt": {
        "bins": np.linspace(0, 50, 51),
        "xlabel": "jet1_sv_pt",
    },
    "jet1_sv_mass": {
        "bins": np.linspace(0, 20, 21),
        "xlabel": "jet1_sv_mass",
    },
    "jet1_sv_Ntrk": {
        "bins": np.linspace(0, 21, 21),
        "xlabel": "jet1_sv_Ntrk",
    },
    "dR_jet1_dijet": {
        "bins": np.linspace(0, 4, 21),
        "xlabel": "dR_jet1_dijet",
    },
    "dPhi_jet1_dijet": {
        "bins": np.linspace(0, 3, 21),
        "xlabel": "dPhi_jet1_dijet",
    },
    "dEta_jet1_dijet": {
        "bins": np.linspace(0, 3, 21),
        "xlabel": "dEta_jet1_dijet",
    },


    "jet2_pt": {
        "bins": np.linspace(0, 200, 41),
        "xlabel": "jet2_pt [GeV]",
    },
    "jet2_eta": {
        "bins": np.linspace(-2, 2, 41),
        "xlabel": "jet2_eta [GeV]",
    },
    "jet2_phi": {
        "bins": np.linspace(-3.14, 3.14, 21),
        "xlabel": "jet2_phi",
    },
    "jet2_btagWP": {
        "bins": np.linspace(0, 4, 21),
        "xlabel": "jet2_btagWP",
    },
    "jet2_btagDeepFlavB": {
        "bins": np.linspace(0, 1, 21),
        "xlabel": "jet2_btagDeepFlavB",
    },
    "jet2_nMuons": {
        "bins": np.linspace(0, 3, 21),
        "xlabel": "jet2_nMuons",
    },
    "jet2_nConstituents": {
        "bins": np.linspace(0, 50, 51),
        "xlabel": "jet2_nConstituents",
    },
    "jet2_leadTrackPt": {
        "bins": np.linspace(0, 40, 21),
        "xlabel": "jet2_leadTrackPt",
    },
    "jet2_nElectrons": {
        "bins": np.linspace(0, 4, 21),
        "xlabel": "jet2_nElectrons",
    },
    "jet2_btagTight": {
        "bins": np.linspace(0, 1, 21),
        "xlabel": "jet2_btagTight",
    },
    "jet2_sv_pt": {
        "bins": np.linspace(0, 50, 51),
        "xlabel": "jet2_sv_pt",
    },
    "jet2_sv_mass": {
        "bins": np.linspace(0, 20, 21),
        "xlabel": "jet2_sv_mass",
    },
    "jet2_sv_Ntrk": {
        "bins": np.linspace(0, 21, 21),
        "xlabel": "jet2_sv_Ntrk",
    },
    "dR_jet2_dijet": {
        "bins": np.linspace(0, 4, 21),
        "xlabel": "dR_jet2_dijet",
    },
    "dPhi_jet2_dijet": {
        "bins": np.linspace(0, 3, 21),
        "xlabel": "dPhi_jet2_dijet",
    },
    "dEta_jet2_dijet": {
        "bins": np.linspace(0, 3, 21),
        "xlabel": "dEta_jet2_dijet",
    },

    "jet3_pt": {
        "bins": np.linspace(0, 200, 41),
        "xlabel": "jet3_pt [GeV]",
    },
    "jet3_eta": {
        "bins": np.linspace(-2, 2, 41),
        "xlabel": "jet3_eta [GeV]",
    },
    "jet3_phi": {
        "bins": np.linspace(-3.14, 3.14, 21),
        "xlabel": "jet3_phi",
    },
    "jet3_btagWP": {
        "bins": np.linspace(0, 4, 21),
        "xlabel": "jet3_btagWP",
    },
    "jet3_btagDeepFlavB": {
        "bins": np.linspace(0, 1, 21),
        "xlabel": "jet3_btagDeepFlavB",
    },
    "jet3_nMuons": {
        "bins": np.linspace(0, 3, 21),
        "xlabel": "jet3_nMuons",
    },
    "jet3_nConstituents": {
        "bins": np.linspace(0, 50, 51),
        "xlabel": "jet3_nConstituents",
    },
    "jet3_leadTrackPt": {
        "bins": np.linspace(0, 40, 21),
        "xlabel": "jet3_leadTrackPt",
    },
    "jet3_nElectrons": {
        "bins": np.linspace(0, 4, 21),
        "xlabel": "jet3_nElectrons",
    },
    "jet3_btagTight": {
        "bins": np.linspace(0, 1, 21),
        "xlabel": "jet3_btagTight",
    },
    "jet3_sv_pt": {
        "bins": np.linspace(0, 50, 51),
        "xlabel": "jet3_sv_pt",
    },
    "jet3_sv_mass": {
        "bins": np.linspace(0, 20, 21),
        "xlabel": "jet3_sv_mass",
    },
    "jet3_sv_Ntrk": {
        "bins": np.linspace(0, 21, 21),
        "xlabel": "jet3_sv_Ntrk",
    },
    "dR_jet3_dijet": {
        "bins": np.linspace(0, 4, 21),
        "xlabel": "dR_jet3_dijet",
    },
    "dPhi_jet3_dijet": {
        "bins": np.linspace(0, 3, 21),
        "xlabel": "dPhi_jet3_dijet",
    },
    "dEta_jet3_dijet": {
        "bins": np.linspace(0, 5, 21),
        "xlabel": "dEta_jet3_dijet",
    },



    "jet4_pt": {
        "bins": np.linspace(0, 100, 41),
        "xlabel": "jet4_pt [GeV]",
    },
    "jet4_eta": {
        "bins": np.linspace(-2, 2, 41),
        "xlabel": "jet4_eta [GeV]",
    },
    "jet4_phi": {
        "bins": np.linspace(-3.14, 3.14, 21),
        "xlabel": "jet4_phi",
    },
    "jet4_btagWP": {
        "bins": np.linspace(0, 4, 21),
        "xlabel": "jet4_btagWP",
    },
    "jet4_btagDeepFlavB": {
        "bins": np.linspace(0, 1, 21),
        "xlabel": "jet4_btagDeepFlavB",
    },
    "jet4_nMuons": {
        "bins": np.linspace(0, 3, 21),
        "xlabel": "jet4_nMuons",
    },
    "jet4_nConstituents": {
        "bins": np.linspace(0, 50, 51),
        "xlabel": "jet4_nConstituents",
    },
    "jet4_leadTrackPt": {
        "bins": np.linspace(0, 40, 21),
        "xlabel": "jet4_leadTrackPt",
    },
    "jet4_nElectrons": {
        "bins": np.linspace(0, 4, 21),
        "xlabel": "jet4_nElectrons",
    },
    "jet4_btagTight": {
        "bins": np.linspace(0, 1, 21),
        "xlabel": "jet4_btagTight",
    },
    "jet4_sv_pt": {
        "bins": np.linspace(0, 10, 51),
        "xlabel": "jet4_sv_pt",
    },
    "jet4_sv_mass": {
        "bins": np.linspace(0, 20, 21),
        "xlabel": "jet4_sv_mass",
    },
    "jet4_sv_Ntrk": {
        "bins": np.linspace(0, 21, 21),
        "xlabel": "jet4_sv_Ntrk",
    },
    "dR_jet4_dijet": {
        "bins": np.linspace(0, 4, 21),
        "xlabel": "dR_jet4_dijet",
    },
    "dPhi_jet4_dijet": {
        "bins": np.linspace(0, 3, 21),
        "xlabel": "dPhi_jet4_dijet",
    },
    "dEta_jet4_dijet": {
        "bins": np.linspace(0, 3, 21),
        "xlabel": "dEta_jet4_dijet",
    },


    "jet1_muon_pt": {
        "bins": np.linspace(-0,45,31),
        "xlabel": "jet1_muon_pt"
    },
    "jet1_muon_eta": {
        "bins": np.linspace(-1.5,1.5,31),
        "xlabel": "jet1_muon_eta"
    },
    "jet1_muon_phi": {
        "bins": np.linspace(-3.14, 3.14, 21),
        "xlabel": "jet1_muon_phi"
    },
    "jet1_muon_ptRel": {
        "bins": np.linspace(0, 10, 21),
        "xlabel": "jet1_muon_ptRel"
    },
    "jet1_muon_dxySig": {
        "bins": np.linspace(-20, 20, 21),
        "xlabel": "jet1_muon_dxySig"
    },
    "jet1_muon_dxy": {
        "bins": np.linspace(-0.5, 0.5, 21),
        "xlabel": "jet1_muon_dxy"
    },
    "jet1_muon_dzSig": {
        "bins": np.linspace(-10, 10, 21),
        "xlabel": "jet1_muon_dzSig"
    },
    "jet1_muon_IP3d": {
        "bins": np.linspace(0, 0.1, 21),
        "xlabel": "jet1_muon_IP3d"
    },
    "jet1_muon_sIP3d": {
        "bins": np.linspace(4, 20, 21),
        "xlabel": "jet1_muon_sIP3d"
    },
    "jet1_muon_tightId": {
        "bins": np.linspace(0, 10, 21),
        "xlabel": "jet1_muon_tightId"
    },
    "jet1_muon_pfRelIso03_all": {
        "bins": np.linspace(0, 10, 21),
        "xlabel": "jet1_muon_pfRelIso03_all"
    },
    "jet1_muon_pfRelIso04_all": {
        "bins": np.linspace(0, 10, 21),
        "xlabel": "jet1_muon_pfRelIso04_all"
    },
    "jet1_muon_charge": {
        "bins": np.linspace(-1.5, 1.5, 21),
        "xlabel": "jet1_muon_charge"
    },


    "dimuon_mass": {
        "bins": np.linspace(0.107,51,31),
        "xlabel": "dimuon_mass"
    },
    "sphericity": {
        "bins": np.linspace(0, 1, 21),
        "xlabel": "sphericity"
    },
    "lambda1": {
        "bins": np.linspace(0, 1, 21),
        "xlabel": "lambda1"
    },
    "lambda2": {
        "bins": np.linspace(0, 1, 21),
        "xlabel": "lambda2"
    },
    "lambda3": {
        "bins": np.linspace(0, 1, 21),
        "xlabel": "lambda3"
    },
    "nJets": {
        "bins": np.linspace(0, 5, 21),
        "xlabel": "nJets"
    },
    "nJets_20": {
        "bins": np.linspace(0, 5, 21),
        "xlabel": "nJets_20"
    },
    "nJets_30": {
        "bins": np.linspace(0, 5, 21),
        "xlabel": "nJets_30"
    },
    "nJets_50": {
        "bins": np.linspace(0, 5, 21),
        "xlabel": "nJets_50"
    },
    "ht": {
        "bins": np.linspace(50, 300, 21),
        "xlabel": "ht"
    },
    "nMuons": {
        "bins": np.linspace(0, 5, 21),
        "xlabel": "nMuons"
    },
    "nIsoMuons": {
        "bins": np.linspace(0, 5, 21),
        "xlabel": "nIsoMuons"
    },
    "nElectrons": {
        "bins": np.linspace(0, 5, 21),
        "xlabel": "nElectrons"
    },
    "nSV": {
        "bins": np.linspace(0, 21, 21),
        "xlabel": "nSV"
    },
    "dijet_eta": {
        "bins": np.linspace(-5, 5, 21),
        "xlabel": "dijet_eta"
    },
    "dijet_phi": {
        "bins": np.linspace(-3.14, 3.14, 21),
        "xlabel": "dijet_phi"
    },
    "dijet_mass": {
        "bins": np.linspace(30, 500, 41),
        "xlabel": "dijet_mass"
    },
    "dijet_dR": {
        "bins": np.linspace(0, 4, 21),
        "xlabel": "dijet_dR"
    },
    #"dijet_dEta": {
    #    "bins": np.linspace(0, 4, 21),
    #    "xlabel": "dijet_dEta"
    #},
    "dijet_dPhi": {
        "bins": np.linspace(0, 4, 21),
        "xlabel": "dijet_dPhi"
    },
    "dijet_twist": {
        "bins": np.linspace(0, 3.14/2, 21),
        "xlabel": "dijet_twist"
    },
    "dijet_cs": {
        "bins": np.linspace(0, 1, 21),
        "xlabel": "dijet_cs"
    },
    "normalized_dijet_pt": {
        "bins": np.linspace(0, 1, 21),
        "xlabel": "normalized_dijet_pt"
    },
    "dijet_pt": {
        "bins": np.linspace(100, 300, 21),
        "xlabel": "Dijet pt [GeV]"
    },
    "dijet_mass": {
        "bins": np.linspace(50, 300, 21),
        "xlabel": "Dijet mass [GeV]"
    },
    "dijet_pT_asymmetry": {
        "bins": np.linspace(0, 1, 21),
        "xlabel": "dijet_pT_asymmetry"
    },
    "PNN": {
        "bins": np.linspace(0, 1, 51),
        "xlabel": "NN score"
    },
    "PNN_pca": {
        "bins": np.linspace(0, 1, 21),
        "xlabel": "NN score rotated"
    },
    "PNN_qm": {
        "bins": np.linspace(0, 1, 51),
        "xlabel": "NN score Morphed"
    },
}
