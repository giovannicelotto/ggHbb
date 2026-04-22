from processing import apply_cuts_single
#JEC_Variations = [
#            "JECAbsoluteMPFBias", "JECAbsoluteScale","JECAbsoluteStat", "JECFlavorQCD","JECFragmentation", "JECPileUpDataMC","JECPileUpPtBB", "JECPileUpPtEC1",
#            "JECPileUpPtEC2", "JECPileUpPtHF","JECPileUpPtRef", "JECRelativeBal","JECRelativeFSR", "JECRelativeJEREC1","JECRelativeJEREC2", "JECRelativeJERHF",
#            "JECRelativePtBB", "JECRelativePtEC1","JECRelativePtEC2", "JECRelativePtHF","JECRelativeSample", "JECRelativeStatEC","JECRelativeStatFSR", "JECRelativeStatHF",
#            "JECSinglePionECAL", "JECSinglePionHCAL","JECTimePtEta"
#        ]
def apply_top_pt(dfMC, up):
    df_ = dfMC.copy()
    if up:
        df_["weight"] = dfMC["weight"]*dfMC['top_pt_reweighting']
    if not up:
        df_["weight"] = dfMC["weight"]/dfMC['top_pt_reweighting']
    return df_


def apply_btag(dfMC, hf, up):
    df_ = dfMC.copy()
    if hf:
        if up:
            df_["weight"] *= dfMC['btag_sf_hf_up']/dfMC['btag_sf_hf_central']
        else:
            df_["weight"] *= dfMC['btag_sf_hf_down']/dfMC['btag_sf_hf_central']
    else:
        if up:
            df_["weight"] *= dfMC['btag_sf_light_up']/dfMC['btag_sf_lightf_central']
        else:
            df_["weight"] *= dfMC['btag_sf_light_down']/dfMC['btag_sf_lightf_central']
    return df_

def apply_jet_puid(dfMC, up):
    df_ = dfMC.copy()
    if up:
        df_["weight"] *= dfMC['jet_pileupId_SF_up']/dfMC['jet_pileupId_SF_nom']
    else:
        df_["weight"] *= dfMC['jet_pileupId_SF_down']/dfMC['jet_pileupId_SF_nom']
    return df_

def apply_JECUnc(dfMC, variation, up):
    df_ = dfMC.copy()
    if up:
        df_["PNN_qm"] = dfMC["NN_qm_Jet_sys_"+variation+"_up"]
        df_["dijet_pt"] = dfMC["dijet_pt_Jet_sys_"+variation+"_up"]
    else:
        df_["PNN_qm"] = dfMC["NN_qm_Jet_sys_"+variation+"_down"]
        df_["dijet_pt"] = dfMC["dijet_pt_Jet_sys_"+variation+"_down"]
    return df_


def apply_ElectronRecoUnc(dfMC, up):
    df_ = dfMC.copy()
    if up:
        df_["weight"] *= dfMC['Electron_tt_RECO_SF_up']/dfMC['Electron_tt_RECO_SF']
    else:
        df_["weight"] *= dfMC['Electron_tt_RECO_SF_down']/dfMC['Electron_tt_RECO_SF']
    return df_
def apply_ElectronIdUnc(dfMC, up):
    df_ = dfMC.copy()
    
    if up:
        df_["weight"] *= dfMC['Electron_tt_ID_SF_up']/dfMC['Electron_tt_ID_SF']
    else:
        df_["weight"] *= dfMC['Electron_tt_ID_SF_down']/dfMC['Electron_tt_ID_SF']
    return df_

def apply_MuonRecoStatUnc(dfMC, up):
    df_ = dfMC.copy()
    
    if up:
        df_["weight"] *= (dfMC['Muon_tt_RECO_SF']+dfMC['Muon_tt_RECO_stat'])/dfMC['Muon_tt_RECO_SF']
    else:
        df_["weight"] *= (dfMC['Muon_tt_RECO_SF']-dfMC['Muon_tt_RECO_stat'])/dfMC['Muon_tt_RECO_SF']
    return df_

def apply_MuonRecoSystUnc(dfMC, up):
    df_ = dfMC.copy()
    if up:
        df_["weight"] *= (dfMC['Muon_tt_RECO_SF']+dfMC['Muon_tt_RECO_syst'])/dfMC['Muon_tt_RECO_SF']
    else:
        df_["weight"] *= (dfMC['Muon_tt_RECO_SF']-dfMC['Muon_tt_RECO_syst'])/dfMC['Muon_tt_RECO_SF']
    return df_
def apply_MuonIdSystUnc(dfMC, up):
    df_ = dfMC.copy()
    if up:
        df_["weight"] *= (dfMC['Muon_tt_ID_SF']+dfMC['Muon_tt_ID_syst'])/dfMC['Muon_tt_ID_SF']
    else:
        df_["weight"] *= (dfMC['Muon_tt_ID_SF']-dfMC['Muon_tt_ID_syst'])/dfMC['Muon_tt_ID_SF']
    return df_
def apply_MuonISOSystUnc(dfMC, up):
    df_ = dfMC.copy()
    if up:
        df_["weight"] *= (dfMC['Muon_tt_ISO_SF']+dfMC['Muon_tt_ISO_syst'])/dfMC['Muon_tt_ISO_SF']
    else:
        df_["weight"] *= (dfMC['Muon_tt_ISO_SF']-dfMC['Muon_tt_ISO_syst'])/dfMC['Muon_tt_ISO_SF']
    return df_
def apply_trig_sf_unc(dfMC, up):
    df_ = dfMC.copy()
    if up:
        df_["weight"] *= (dfMC['trig_sf']+dfMC['err_trig_sf'])/dfMC['trig_sf']
    else:
        df_["weight"] *= (dfMC['trig_sf']-dfMC['err_trig_sf'])/dfMC['trig_sf']
    return df_



def build_systematics(dfMC, config):
    variations = {}

    # nominal
    variations["nominal"] = apply_cuts_single(dfMC.copy(), config)

    # example systematics
    for syst in config.get("systematics", []):
        print(f"Applying systematic: {syst}")
        if syst == "top_pT":
            df_up = apply_top_pt(dfMC, up=True)
            df_down = apply_top_pt(dfMC, up=False)
            variations["top_pTUp"] = apply_cuts_single(df_up, config)
            variations["top_pTDown"] = apply_cuts_single(df_down, config)

        elif syst == "btag":
            df_hf_up  = apply_btag(dfMC, hf=1, up=True)
            df_hf_down  = apply_btag(dfMC, hf=1, up=False)
            df_lightf_up  = apply_btag(dfMC, hf=0, up=True)
            df_lightf_down  = apply_btag(dfMC, hf=0, up=False)

            variations["btag_hfUp"] = apply_cuts_single(df_hf_up, config)
            variations["btag_hfDown"] = apply_cuts_single(df_hf_down, config)
            variations["btag_lightfUp"] = apply_cuts_single(df_lightf_up, config)
            variations["btag_lightfDown"] = apply_cuts_single(df_lightf_down, config)

        # Include all JEC variations
        elif syst.startswith("JEC"):
            df_up = apply_JECUnc(dfMC, variation=syst, up=True)
            df_down = apply_JECUnc(dfMC, variation=syst, up=False)
            variations[syst+"Up"] = apply_cuts_single(df_up, config)
            variations[syst+"Down"] = apply_cuts_single(df_down, config)

        elif syst=="jet_puid":
            variations["jet_puidUp"] = apply_cuts_single(apply_jet_puid(dfMC, up=True), config)
            variations["jet_puidDown"] = apply_cuts_single(apply_jet_puid(dfMC, up=False), config)
        elif syst=="ElectronReco":
            variations["ElectronRecoUp"] = apply_cuts_single(apply_ElectronRecoUnc(dfMC, up=True), config)
            variations["ElectronRecoDown"] = apply_cuts_single(apply_ElectronRecoUnc(dfMC, up=False), config)
        elif syst=="ElectronId":
            variations["ElectronIdUp"] = apply_cuts_single(apply_ElectronIdUnc(dfMC, up=True), config)
            variations["ElectronIdDown"] = apply_cuts_single(apply_ElectronIdUnc(dfMC, up=False), config)
        elif syst=="MuonRecoStat":
            variations["MuonRecoStatUp"] = apply_cuts_single(apply_MuonRecoStatUnc(dfMC, up=True), config)
            variations["MuonRecoStatDown"] = apply_cuts_single(apply_MuonRecoStatUnc(dfMC, up=False), config)
        elif syst=="MuonRecoSyst":
            variations["MuonRecoSystUp"] = apply_cuts_single(apply_MuonRecoSystUnc(dfMC, up=True), config)
            variations["MuonRecoSystDown"] = apply_cuts_single(apply_MuonRecoSystUnc(dfMC, up=False), config)
        elif syst=="MuonIdSyst":
            variations["MuonIdSystUp"] = apply_cuts_single(apply_MuonIdSystUnc(dfMC, up=True), config)
            variations["MuonIdSystDown"] = apply_cuts_single(apply_MuonIdSystUnc(dfMC, up=False), config)
        elif syst=="MuonISOSyst":
            variations["MuonISOSystUp"] = apply_cuts_single(apply_MuonISOSystUnc(dfMC, up=True), config)
            variations["MuonISOSystDown"] = apply_cuts_single(apply_MuonISOSystUnc(dfMC, up=False), config)
        elif syst=="trig_sf":
            variations["trig_sfUp"] = apply_cuts_single(apply_trig_sf_unc(dfMC, up=True), config)
            variations["trig_sfDown"] = apply_cuts_single(apply_trig_sf_unc(dfMC, up=False), config)
        

    return variations