import numpy as np
def getMuon_SF(json_data, wp_name, eta, pt):
    """
    Extract scale factor (and uncertainties) for given eta, pt.
    Handles underflow/overflow by taking the first/last bin.
    """
    wp_dict = json_data[wp_name]["abseta_pt"]
    binning = wp_dict["binning"]

    abseta_bins = binning[0]["binning"]
    pt_bins = binning[1]["binning"]

    abseta = abs(eta)

    # --- find abseta bin (with underflow/overflow handling) ---
    if abseta < abseta_bins[0]:
        lo, hi = abseta_bins[0], abseta_bins[1]
    elif abseta >= abseta_bins[-1]:
        lo, hi = abseta_bins[-2], abseta_bins[-1]
    else:
        for i in range(len(abseta_bins) - 1):
            if abseta_bins[i] <= abseta < abseta_bins[i + 1]:
                lo, hi = abseta_bins[i], abseta_bins[i + 1]
                break
    ab_key = f"abseta:[{lo},{hi}]"

    # --- find pt bin (with underflow/overflow handling) ---
    if pt < pt_bins[0]:
        plo, phi = pt_bins[0], pt_bins[1]
    elif pt >= pt_bins[-1]:
        plo, phi = pt_bins[-2], pt_bins[-1]
    else:
        for i in range(len(pt_bins) - 1):
            if pt_bins[i] <= pt < pt_bins[i + 1]:
                plo, phi = pt_bins[i], pt_bins[i + 1]
                break
    pt_key = f"pt:[{plo},{phi}]"

    # --- extract the info ---
    sf_info = wp_dict[ab_key][pt_key]

    return sf_info


def get_muon_recoSF(data, eta):
    """
    Returns (value, stat, syst) for a muon of given eta, using the medium-pT bin for all pT.
    If extrap_frac is set (e.g. 0.005 for 0.5%), it is added in quadrature to 'syst'.
    """
    sf_dict = data["NUM_TrackerMuons_DEN_genTracks"]["abseta_pt"]
    abseta_bins = sf_dict["binning"][0]["binning"]
    abseta = abs(eta)

    # find abseta bin (inclusive on lower edge, exclusive on upper)
    ab_key = None
    for lo, hi in zip(abseta_bins[:-1], abseta_bins[1:]):
        if lo <= abseta < hi:
            ab_key = f"abseta:[{lo},{hi}]"
            break
    if ab_key is None:
        # Out of range: clamp to nearest bin and (optionally) inflate syst
        if abseta < abseta_bins[0]:
            lo, hi = abseta_bins[0], abseta_bins[1]
        else:
            lo, hi = abseta_bins[-2], abseta_bins[-1]
        ab_key = f"abseta:[{lo},{hi}]"

    # always use the medium-pT bin key
    pt_key = "pt:[40,60]"

    info = sf_dict[ab_key][pt_key]
    val, stat, syst = info["value"], info["stat"], info["syst"]

    return {"value": val, "stat": stat, "syst": syst}

def btag_wp(btag, pt):
    if pt < 20:
        return 0
    if btag < 0.049:
        return 0
    if btag < 0.2783:
        return 1
    if btag < 0.71:
        return 2
    return 3

def get_btag_map_efficiency(jet_pt, jet_eta, flav, eff_map_data):
    '''
    evt["jet_pt"] = scalar value of pt of the jet considered
    evt["jet_eta"] = scalar value of eta of the jet considered
    flav = scalar value of flav of the jet considered (0, 4, 5)
    eff_map_data = scalar value of wp of the jet considered (L, M, T)
    '''
    pt_bins = np.array(eff_map_data['pt_bins'])
    eta_bins = np.array(eff_map_data['eta_bins'])
    eff_map = {
        'b': np.array(eff_map_data['eff_map']['b']),
        'c': np.array(eff_map_data['eff_map']['c']),
        'light': np.array(eff_map_data['eff_map']['light'])
    }

    # From the map extract the efficiency

    # Absolute eta
    abs_eta = np.abs(jet_eta)
    # Digitize bins: returns bin indices
    pt_bin_idx = np.digitize(jet_pt, pt_bins) - 1
    eta_bin_idx = np.digitize(abs_eta, eta_bins) - 1
    
    # Clip indices to valid range (handle overflow by clipping to last bin)
    pt_bin_idx = np.clip(pt_bin_idx, 0, len(pt_bins) - 2)
    eta_bin_idx = np.clip(eta_bin_idx, 0, len(eta_bins) - 2)

    # Select flavour key for efficiency
    if flav == 5:
        flav_key = 'b'
    elif flav == 4:
        flav_key = 'c'
    else:
        flav_key = 'light'

    eff_array = eff_map[flav_key]
    eff = eff_array[pt_bin_idx, eta_bin_idx]

    return eff, flav_key

def log_bad_btag_sf(
    logfile,
    process,
    event,
    weight_factor,
):
    with open(logfile, "a+") as f:
        f.write(
            f"{process} ev={event} "
            f"weight_factor={weight_factor}\n"
        )


def get_trig_SF(LeadingMuons_inJets, Muon_isTriggering, Muon_pt, Muon_dxy, Muon_dxyErr, muon_2_isTriggering, effData_rootfile, effMC_rootfile,):
    def remove_underflow_overflow(xbin, ybin, hist):
        if xbin == hist.GetNbinsX()+1:
            xbin=xbin-1
        if ybin == hist.GetNbinsY()+1:
            ybin=ybin-1
        # if underflow gets the same triggerSF as the first bin
        if xbin == 0:
            xbin=1
        if ybin == 0:
            ybin=1
        return xbin, ybin

    if len(LeadingMuons_inJets)==1:
        hist_MC = effMC_rootfile.Get("hMap")
        hist_Data = effData_rootfile.Get("hMap")
        xbin1_MC = hist_MC.GetXaxis().FindBin(Muon_pt[LeadingMuons_inJets[0]])
        ybin1_MC = hist_MC.GetYaxis().FindBin(Muon_dxy[LeadingMuons_inJets[0]]/Muon_dxyErr[LeadingMuons_inJets[0]])
        xbin1_Data = hist_Data.GetXaxis().FindBin(Muon_pt[LeadingMuons_inJets[0]])
        ybin1_Data = hist_Data.GetYaxis().FindBin(Muon_dxy[LeadingMuons_inJets[0]]/Muon_dxyErr[LeadingMuons_inJets[0]])
        xbin1_MC, ybin1_MC = remove_underflow_overflow(xbin1_MC, ybin1_MC, hist_MC)
        xbin1_Data, ybin1_Data = remove_underflow_overflow(xbin1_Data, ybin1_Data, hist_Data)
        efficiency_data = hist_Data.GetBinContent(xbin1_Data, ybin1_Data)
        efficiency_MC = hist_MC.GetBinContent(xbin1_MC, ybin1_MC)
        sf = efficiency_data / efficiency_MC if efficiency_MC > 0 else 1.0
    if len(LeadingMuons_inJets)>1:
        #Two muons need to be checked
        hist_MC = effMC_rootfile.Get("hMap")
        hist_Data = effData_rootfile.Get("hMap")
        xbin1_MC = hist_MC.GetXaxis().FindBin(Muon_pt[LeadingMuons_inJets[0]])
        ybin1_MC = hist_MC.GetYaxis().FindBin(Muon_dxy[LeadingMuons_inJets[0]]/Muon_dxyErr[LeadingMuons_inJets[0]])
        xbin2_MC = hist_MC.GetXaxis().FindBin(Muon_pt[LeadingMuons_inJets[1]])
        ybin2_MC = hist_MC.GetYaxis().FindBin(Muon_dxy[LeadingMuons_inJets[1]]/Muon_dxyErr[LeadingMuons_inJets[1]])
        xbin1_Data = hist_Data.GetXaxis().FindBin(Muon_pt[LeadingMuons_inJets[0]])
        ybin1_Data = hist_Data.GetYaxis().FindBin(Muon_dxy[LeadingMuons_inJets[0]]/Muon_dxyErr[LeadingMuons_inJets[0]])
        xbin2_Data = hist_Data.GetXaxis().FindBin(Muon_pt[LeadingMuons_inJets[1]])
        ybin2_Data = hist_Data.GetYaxis().FindBin(Muon_dxy[LeadingMuons_inJets[1]]/Muon_dxyErr[LeadingMuons_inJets[1]])

        xbin1_MC, ybin1_MC = remove_underflow_overflow(xbin1_MC, ybin1_MC, hist_MC)
        xbin2_MC, ybin2_MC = remove_underflow_overflow(xbin2_MC, ybin2_MC, hist_MC)
        xbin1_Data, ybin1_Data = remove_underflow_overflow(xbin1_Data, ybin1_Data, hist_Data)
        xbin2_Data, ybin2_Data = remove_underflow_overflow(xbin2_Data, ybin2_Data, hist_Data)

        if ((Muon_isTriggering[LeadingMuons_inJets[0]]) & (Muon_isTriggering[LeadingMuons_inJets[1]])):
            efficiency_data = hist_Data.GetBinContent(xbin1_Data, ybin1_Data) * hist_Data.GetBinContent(xbin2_Data, ybin2_Data)
            efficiency_MC = hist_MC.GetBinContent(xbin1_MC, ybin1_MC) * hist_MC.GetBinContent(xbin2_MC, ybin2_MC)
            sf = efficiency_data / efficiency_MC if efficiency_MC > 0 else 1.0
        elif ((Muon_isTriggering[LeadingMuons_inJets[0]]) & (not Muon_isTriggering[LeadingMuons_inJets[1]])):
            efficiency_data = hist_Data.GetBinContent(xbin1_Data, ybin1_Data) * (1-hist_Data.GetBinContent(xbin2_Data, ybin2_Data))
            efficiency_MC = hist_MC.GetBinContent(xbin1_MC, ybin1_MC) * (1-hist_MC.GetBinContent(xbin2_MC, ybin2_MC))
            sf = efficiency_data / efficiency_MC if efficiency_MC > 0 else 1.0
        elif ((not Muon_isTriggering[LeadingMuons_inJets[0]]) & (Muon_isTriggering[LeadingMuons_inJets[1]])):
            efficiency_data = (1-hist_Data.GetBinContent(xbin1_Data, ybin1_Data)) * (hist_Data.GetBinContent(xbin2_Data, ybin2_Data))
            efficiency_MC = (1-hist_MC.GetBinContent(xbin1_MC, ybin1_MC)) * (hist_MC.GetBinContent(xbin2_MC, ybin2_MC))
            sf = efficiency_data / efficiency_MC if efficiency_MC > 0 else 1.0
        else:
            assert False, "This assert was called"


    return np.float32(sf)


def get_btag_SF(btagMapsExist, evt, maskJets, corrDeepJet_FixedWP_comb, corrDeepJet_FixedWP_light, eff_maps_cache_btag, wp_converter, processName):
    
    btag_sf_hf_central, btag_sf_hf_up,  btag_sf_hf_down, btag_sf_lightf_central, btag_sf_light_up, btag_sf_light_down = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    if not btagMapsExist:
        pass
    else:
        for j in np.arange(evt["nJet"])[maskJets]:

            # Extract the tagged WP of the current jet
            if wp_converter.evaluate("L") <= evt["Jet_btagDeepFlavB"][j] < wp_converter.evaluate("M"):
                wp = "L"
            elif wp_converter.evaluate("M") <= evt["Jet_btagDeepFlavB"][j] < wp_converter.evaluate("T"):
                wp = "M"
            elif wp_converter.evaluate("T") <= evt["Jet_btagDeepFlavB"][j]:
                wp = "T"
            else:
                wp = None  
            # Determine btag SF for different systematics
            flav = abs(evt["Jet_hadronFlavour"][j])
            is_hf = flav in (4, 5)
            is_lf = flav == 0
            
            # Determine efficiency
            eff_T, flav_key = get_btag_map_efficiency(evt["Jet_pt"][j], evt["Jet_eta"][j], evt["Jet_hadronFlavour"][j], eff_maps_cache_btag["T"])
            eff_M, _        = get_btag_map_efficiency(evt["Jet_pt"][j], evt["Jet_eta"][j], evt["Jet_hadronFlavour"][j], eff_maps_cache_btag["M"])
            eff_L, _        = get_btag_map_efficiency(evt["Jet_pt"][j], evt["Jet_eta"][j], evt["Jet_hadronFlavour"][j], eff_maps_cache_btag["L"])

            eps = 1e-8
            if wp=="T":
                if is_hf:
                    currentJet_btagSF_T_central = corrDeepJet_FixedWP_comb.evaluate("central", "T", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    currentJet_btagSF_T_up = corrDeepJet_FixedWP_comb.evaluate("up", "T", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    currentJet_btagSF_T_down = corrDeepJet_FixedWP_comb.evaluate("down", "T", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    btag_sf_hf_central *= (currentJet_btagSF_T_central) 
                    btag_sf_hf_up *= (currentJet_btagSF_T_up)
                    btag_sf_hf_down *= (currentJet_btagSF_T_down)

                elif is_lf:
                    currentJet_btagSF_T_central = corrDeepJet_FixedWP_light.evaluate("central", "T", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    currentJet_btagSF_T_up = corrDeepJet_FixedWP_light.evaluate("up", "T", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    currentJet_btagSF_T_down = corrDeepJet_FixedWP_light.evaluate("down", "T", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    btag_sf_lightf_central *= (currentJet_btagSF_T_central) 
                    btag_sf_light_up *= (currentJet_btagSF_T_up) 
                    btag_sf_light_down *= (currentJet_btagSF_T_down) 

            elif wp == "M":
                denom = eff_M - eff_T
                if is_hf:
                    currentJet_btagSF_M_central = corrDeepJet_FixedWP_comb.evaluate("central", "M", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    currentJet_btagSF_M_up = corrDeepJet_FixedWP_comb.evaluate("up", "M", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    currentJet_btagSF_M_down = corrDeepJet_FixedWP_comb.evaluate("down", "M", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    currentJet_btagSF_T_central = corrDeepJet_FixedWP_comb.evaluate("central", "T", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    currentJet_btagSF_T_up = corrDeepJet_FixedWP_comb.evaluate("up", "T", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    currentJet_btagSF_T_down = corrDeepJet_FixedWP_comb.evaluate("down", "T", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))

                    btag_sf_hf_central *= (currentJet_btagSF_M_central*eff_M-currentJet_btagSF_T_central*eff_T) / denom
                    btag_sf_hf_up *= (currentJet_btagSF_M_up*eff_M-currentJet_btagSF_T_up*eff_T) / denom
                    btag_sf_hf_down *= (currentJet_btagSF_M_down*eff_M-currentJet_btagSF_T_down*eff_T) / denom
                elif is_lf:
                    currentJet_btagSF_M_central = corrDeepJet_FixedWP_light.evaluate("central", "M", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    currentJet_btagSF_M_up = corrDeepJet_FixedWP_light.evaluate("up", "M", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    currentJet_btagSF_M_down = corrDeepJet_FixedWP_light.evaluate("down", "M", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    currentJet_btagSF_T_central = corrDeepJet_FixedWP_light.evaluate("central", "T", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    currentJet_btagSF_T_up = corrDeepJet_FixedWP_light.evaluate("up", "T", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    currentJet_btagSF_T_down = corrDeepJet_FixedWP_light.evaluate("down", "T", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))

                    btag_sf_lightf_central *= (currentJet_btagSF_M_central*eff_M-currentJet_btagSF_T_central*eff_T) / denom
                    btag_sf_light_up *= (currentJet_btagSF_M_up*eff_M-currentJet_btagSF_T_up*eff_T) / denom
                    btag_sf_light_down *= (currentJet_btagSF_M_down*eff_M-currentJet_btagSF_T_down*eff_T) / denom

            elif wp == "L":
                denom = eff_L - eff_M
                if is_hf:
                    currentJet_btagSF_L_central = corrDeepJet_FixedWP_comb.evaluate("central", "L", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    currentJet_btagSF_L_up = corrDeepJet_FixedWP_comb.evaluate("up", "L", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    currentJet_btagSF_L_down = corrDeepJet_FixedWP_comb.evaluate("down", "L", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    currentJet_btagSF_M_central = corrDeepJet_FixedWP_comb.evaluate("central", "M", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    currentJet_btagSF_M_up = corrDeepJet_FixedWP_comb.evaluate("up", "M", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    currentJet_btagSF_M_down = corrDeepJet_FixedWP_comb.evaluate("down", "M", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))

                    btag_sf_hf_central *= (currentJet_btagSF_L_central*eff_L-currentJet_btagSF_M_central*eff_M) / denom
                    btag_sf_hf_up *= (currentJet_btagSF_L_up*eff_L-currentJet_btagSF_M_up*eff_M) / denom
                    btag_sf_hf_down *= (currentJet_btagSF_L_down*eff_L-currentJet_btagSF_M_down*eff_M) / denom
                elif is_lf:
                    currentJet_btagSF_L_central = corrDeepJet_FixedWP_light.evaluate("central", "L", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    currentJet_btagSF_L_up = corrDeepJet_FixedWP_light.evaluate("up", "L", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    currentJet_btagSF_L_down = corrDeepJet_FixedWP_light.evaluate("down", "L", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    currentJet_btagSF_M_central = corrDeepJet_FixedWP_light.evaluate("central", "M", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    currentJet_btagSF_M_up = corrDeepJet_FixedWP_light.evaluate("up", "M", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    currentJet_btagSF_M_down = corrDeepJet_FixedWP_light.evaluate("down", "M", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))

                    btag_sf_lightf_central *= (currentJet_btagSF_L_central*eff_L-currentJet_btagSF_M_central*eff_M) / denom
                    btag_sf_light_up *= (currentJet_btagSF_L_up*eff_L-currentJet_btagSF_M_up*eff_M) / denom
                    btag_sf_light_down *= (currentJet_btagSF_L_down*eff_L-currentJet_btagSF_M_down*eff_M) / denom
            
            elif wp is None:
                denom = 1.0 - eff_L
                if is_hf:
                    currentJet_btagSF_L_central = corrDeepJet_FixedWP_comb.evaluate("central", "L", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    currentJet_btagSF_L_up = corrDeepJet_FixedWP_comb.evaluate("up", "L", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    currentJet_btagSF_L_down = corrDeepJet_FixedWP_comb.evaluate("down", "L", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    btag_sf_hf_central *= (1.0 - currentJet_btagSF_L_central * eff_L) / denom
                    btag_sf_hf_up *= (1.0 - currentJet_btagSF_L_up * eff_L) / denom
                    btag_sf_hf_down *= (1.0 - currentJet_btagSF_L_down * eff_L) / denom
                elif is_lf:
                    currentJet_btagSF_L_central = corrDeepJet_FixedWP_light.evaluate("central", "L", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    currentJet_btagSF_L_up = corrDeepJet_FixedWP_light.evaluate("up", "L", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))
                    currentJet_btagSF_L_down = corrDeepJet_FixedWP_light.evaluate("down", "L", abs(evt["Jet_hadronFlavour"][j]), float(abs(evt["Jet_eta"][j])), float(evt["Jet_pt"][j]))

                    btag_sf_lightf_central *= (1.0 - currentJet_btagSF_L_central * eff_L) / denom
                    btag_sf_light_up *= (1.0 - currentJet_btagSF_L_up * eff_L) / denom
                    btag_sf_light_down *= (1.0 - currentJet_btagSF_L_down * eff_L) / denom



        # Jet is M not T
        # Check for nan or infinty
        for sf in [btag_sf_hf_central, btag_sf_hf_up, btag_sf_hf_down,btag_sf_lightf_central,  btag_sf_light_up, btag_sf_light_down]:
            if not np.isfinite(sf):
                log_bad_btag_sf(
                    logfile="/t3home/gcelotto/ggHbb/flatter/bad_btagSF.log",
                    process=processName,
                    event=evt["event"],
                    weight_factor=sf,
                )
        

    return {
        "btag_sf_hf_central" : btag_sf_hf_central,
        "btag_sf_hf_up" : btag_sf_hf_up,
        "btag_sf_hf_down" : btag_sf_hf_down,
        "btag_sf_lightf_central" : btag_sf_lightf_central,
        "btag_sf_light_up" : btag_sf_light_up,
        "btag_sf_light_down" : btag_sf_light_down,
            }

#def get_muon_isoSF(data, pt, eta):
#    """
#    Returns (value, stat, syst) for a muon of given pt, eta.
#    """
#    sf_dict = data["NUM_TrackerMuons_DEN_genTracks"]["abseta_pt"]
#    abseta_bins = sf_dict["binning"][0]["binning"]
#    abseta = abs(eta)
#
#    # find abseta bin (inclusive on lower edge, exclusive on upper)
#    ab_key = None
#    for lo, hi in zip(abseta_bins[:-1], abseta_bins[1:]):
#        if lo <= abseta < hi:
#            ab_key = f"abseta:[{lo},{hi}]"
#            break
#    if ab_key is None:
#        # Out of range: clamp to nearest bin and (optionally) inflate syst
#        if abseta < abseta_bins[0]:
#            lo, hi = abseta_bins[0], abseta_bins[1]
#        else:
#            lo, hi = abseta_bins[-2], abseta_bins[-1]
#        ab_key = f"abseta:[{lo},{hi}]"
#
#    # always use the medium-pT bin key
#    pt_key = "pt:[40,60]"
#
#    info = sf_dict[ab_key][pt_key]
#    val, stat, syst = info["value"], info["stat"], info["syst"]
#
#    return {"value": val, "stat": stat, "syst": syst}