import ROOT
from array import array
import numpy as np
import yaml
from pathlib import Path
def apply_syst(df, syst, cat, particle):
    if syst=="puid_up":
        print("Before", df['weight'].sum())
        df['weight'] = df['weight'] * df['jet_pileupId_SF_up'] / df['jet_pileupId_SF_nom']
        print("After", df['weight'].sum())
    elif syst=="puid_down":
        df['weight'] = df['weight'] * df['jet_pileupId_SF_down'] / df['jet_pileupId_SF_nom']
    elif syst=="btag_hf_up":
        df['weight'] = df['weight'] * df['btag_sf_hf_up'] / df['btag_sf_hf_central']
    elif syst=="btag_hf_down":
        df['weight'] = df['weight'] * df['btag_sf_hf_down'] / df['btag_sf_hf_central']
    elif syst=="btag_lightf_up":
        df['weight'] = df['weight'] * df['btag_sf_light_up'] / df['btag_sf_lightf_central']
    elif syst=="btag_lightf_down":
        df['weight'] = df['weight'] * df['btag_sf_light_down'] / df['btag_sf_lightf_central']
    elif syst=="scale":
        LHEScaleWeight_MuFdown_MuRdown = df['weight'] * df['LHEScaleWeight_MuFdown_MuRdown']
        LHEScaleWeight_MuFdown_MuRnom = df['weight'] * df['LHEScaleWeight_MuFdown_MuRnom']
        LHEScaleWeight_MuFdown_MuRup = df['weight'] * df['LHEScaleWeight_MuFdown_MuRup']
        LHEScaleWeight_MuFnom_MuRdown = df['weight'] * df['LHEScaleWeight_MuFnom_MuRdown']
        LHEScaleWeight_MuFnom_MuRup = df['weight'] * df['LHEScaleWeight_MuFnom_MuRup']
        LHEScaleWeight_MuFup_MuRdown = df['weight'] * df['LHEScaleWeight_MuFup_MuRdown']
        LHEScaleWeight_MuFup_MuRnom = df['weight'] * df['LHEScaleWeight_MuFup_MuRnom']
        LHEScaleWeight_MuFup_MuRup = df['weight'] * df['LHEScaleWeight_MuFup_MuRup']
        scale_weights = np.stack([
                        LHEScaleWeight_MuFdown_MuRdown,
                        LHEScaleWeight_MuFdown_MuRnom,
                        LHEScaleWeight_MuFdown_MuRup,
                        LHEScaleWeight_MuFnom_MuRdown,
                        LHEScaleWeight_MuFnom_MuRup,
                        LHEScaleWeight_MuFup_MuRdown,
                        LHEScaleWeight_MuFup_MuRnom,
                        LHEScaleWeight_MuFup_MuRup,
                    ], axis=0)
        N0 = df['weight'].sum()
        Ni = scale_weights.sum(axis=1)  # sum over events per variation
        delta_up   = np.max(Ni - N0)
        delta_down = np.max(N0 - Ni)
        lnN_scale = 1 + max(delta_up, delta_down) / N0

        yaml_file = Path(f"/t3home/gcelotto/ggHbb/WSFit/Configs/systematics/cat{cat}_{particle}syst.yaml")
        lnN_scale_value = lnN_scale  # from your calculation
        if yaml_file.exists():
            with open(yaml_file, "r") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}
        data['Scale'] = float(lnN_scale_value)
        with open(yaml_file, "w") as f:
            yaml.safe_dump(data, f)

        return None
    elif syst=="PS_ISR":
        PSWeight_ISRdown_FSRnom = df['weight'] * df['PSWeight_ISRdown_FSRnom']
        PSWeight_ISRup_FSRdown = df['weight'] * df['PSWeight_ISRup_FSRnom']
        PS_weights = np.stack([
            PSWeight_ISRdown_FSRnom,
            PSWeight_ISRup_FSRdown,
                    ], axis=0)
        N0 = df['weight'].sum()
        Ni = PS_weights.sum(axis=1)  # sum over events per variation
        delta_up   = np.max(Ni - N0)
        delta_down = np.max(N0 - Ni)
        lnN_PS = 1 + max(delta_up, delta_down) / N0
        print("PS DELTA UP", delta_up)
        print("PS DELTA DOWN", delta_down)
        print("PS N0", N0)

        yaml_file = Path(f"/t3home/gcelotto/ggHbb/WSFit/Configs/systematics/cat{cat}_{particle}syst.yaml")
        lnN_PS_value = lnN_PS  # from your calculation
        print("PS VALUE", lnN_PS_value)
        if yaml_file.exists():
            with open(yaml_file, "r") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}
        data['PS_ISR'] = float(lnN_PS_value)
        with open(yaml_file, "w") as f:
            yaml.safe_dump(data, f)

        return None
        
    elif syst=="PS_FSR":
        #PSWeight_ISRdown_FSRnom = df['weight'] * df['PSWeight_ISRdown_FSRnom']
        #PSWeight_ISRup_FSRdown = df['weight'] * df['PSWeight_ISRup_FSRnom']
        PSWeight_ISRnom_FSRdown = df['weight'] * df['PSWeight_ISRnom_FSRdown']
        PSWeight_ISRnom_FSRup = df['weight'] * df['PSWeight_ISRnom_FSRup']
        PS_weights = np.stack([
            #PSWeight_ISRdown_FSRnom,
            #PSWeight_ISRup_FSRdown,
            PSWeight_ISRnom_FSRdown,
            PSWeight_ISRnom_FSRup
                    ], axis=0)
        print("Is there nan in PSWeight_ISRnom_FSRdown", df.PSWeight_ISRnom_FSRdown.isna().sum())
        N0 = df['weight'].sum()
        Ni = PS_weights.sum(axis=1)  # sum over events per variation
        delta_up   = np.max(abs(Ni - N0))
        delta_down = np.max(abs(N0 - Ni))
        lnN_PS = 1 + max(delta_up, delta_down) / N0
        yaml_file = Path(f"/t3home/gcelotto/ggHbb/WSFit/Configs/systematics/cat{cat}_{particle}syst.yaml")
        lnN_PS_value = lnN_PS  # from your calculation
        print("PS VALUE", lnN_PS_value)
        if yaml_file.exists():
            with open(yaml_file, "r") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}
        data['PS_FSR'] = float(lnN_PS_value)
        with open(yaml_file, "w") as f:
            yaml.safe_dump(data, f)

        return None
    elif syst=="alphaS":
        alphaS_down = df['weight'] * df['LHEAlphasWeight_down']
        alphaS_up = df['weight'] * df['LHEAlphasWeight_up']
        alphaS_weights = np.stack([
            alphaS_down,
            alphaS_up
                    ], axis=0)

        N0 = df['weight'].sum()
        Ni = alphaS_weights.sum(axis=1)  # sum over events per variation
        delta_up   = np.max(abs(Ni - N0))
        delta_down = np.max(abs(N0 - Ni))
        lnN_PS = 1 + max(delta_up, delta_down) / N0
        yaml_file = Path(f"/t3home/gcelotto/ggHbb/WSFit/Configs/systematics/cat{cat}_{particle}syst.yaml")
        lnN_PS_value = lnN_PS  # from your calculation
        if yaml_file.exists():
            with open(yaml_file, "r") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}
        data['alphaS'] = float(lnN_PS_value)
        with open(yaml_file, "w") as f:
            yaml.safe_dump(data, f)

        return None
        
        


    return df

# --- Helper functions ---
def make_hist(name, values, weights, nbins, xmin, xmax):
    hist = ROOT.TH1F(name, name, nbins - 1, xmin, xmax)
    arr_values = array('d', values)
    arr_weights = array('d', weights)
    hist.FillN(len(values), arr_values, arr_weights)
    return hist

def make_roodatahist(name, hist, x_var):
    return ROOT.RooDataHist(name, name, ROOT.RooArgList(x_var), hist)

def build_dscb_gaus_model(x_var, config, label):
    mean = ROOT.RooRealVar(f"mu_{label}", f"mu", config['parameters']['nominal']['mean']['value'], 85, 150)
    sigma_cb = ROOT.RooRealVar(f"sigma_CB_{label}", f"sigma_CB_{label}", config['parameters']['nominal']['sigma']['value'], 4, 20)
    sigma_gaus = ROOT.RooRealVar(f"sigma_gauss_{label}", f"sigma_gauss_{label}", config['parameters']['nominal']['sigmaG']['value'], 4, 20)
    
    alpha1 = ROOT.RooRealVar(f"alpha1_{label}", f"alpha1_{label}", config['parameters']['nominal']['alphaL']['value'], 0.1, 5.0)
    alpha2 = ROOT.RooRealVar(f"alpha2_{label}", f"alpha2_{label}", config['parameters']['nominal']['alphaR']['value'], 0.1, 5.0)
    nL = ROOT.RooRealVar(f"nL_{label}", f"nL_{label}", config['parameters']['nominal']['nL']['value'], 1, 100.0)
    nR = ROOT.RooRealVar(f"nR_{label}", f"nR_{label}", config['parameters']['nominal']['nR']['value'], 1, 100.0)
    
    gaus = ROOT.RooGaussian(f"gauss_{label}", f"gauss_{label}", x_var, mean, sigma_gaus)
    dscb = ROOT.RooDoubleCB(f"dscb_{label}", "Double-Sided Crystal Ball", x_var, mean, sigma_cb, alpha1, nL, alpha2, nR)
    
    frac = ROOT.RooRealVar(f"f_CB_Gaus{label}", f"f_CB_Gaus{label}", config['parameters']['nominal']['fraction_dscb']['value'], 0, 1)

    
    print("Name is ", f"model_{label}")
    model = ROOT.RooAddPdf(f"model_{label}", f"model_{label}", ROOT.RooArgList(dscb, gaus), ROOT.RooArgList(frac))
    model.fixCoefNormalization(ROOT.RooArgSet(x_var))
    
    return model, dscb, gaus, [mean, sigma_cb, sigma_gaus, alpha1, alpha2, nL, nR, frac]
import re

def prettify_param_name(name):
    # 1. Replace sigma with Greek sigma
    name = name.replace("sigma", "#sigma")

    # 2. Remove _H_c<number>
    name = re.sub(r"_H_c\d+", "", name)

    # 3. Replace remaining underscores with spaces
    name = name.replace("_", " ")

    return name

def plot_model(x_var, data_hist, model, components=[], filename="plot.png", title="Fit"):
    x_var.SetTitle("m_{bb} [GeV]")
    frame = x_var.frame(ROOT.RooFit.Title(title))
    data_hist.plotOn(frame, ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2), ROOT.RooFit.Name("data"))
    
    model.plotOn(frame, ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.Name("model"))
    for comp in components:
        model.plotOn(frame, ROOT.RooFit.Components(comp['name']), ROOT.RooFit.LineColor(comp['color']), ROOT.RooFit.LineStyle(comp['style']), ROOT.RooFit.Name(comp['name']))
    leg = ROOT.TLegend(0.7, 0.3, 0.9, 0.6)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(frame.findObject("data"),  "Data",  "PE")
    leg.AddEntry(frame.findObject("model"), "Total model", "L")

    for comp in components:
        leg.AddEntry(
            frame.findObject(comp["name"]),
            comp["label"],
            "L"
        )

    #frame.addPlotable(model, "L")  # ensures model is drawn
    #frame.getAttText().SetTextSize(0.03)  # smaller text
    # Draw parameter values

    #model.paramOn(frame,ROOT.RooFit.Layout(0.55, 0.95, 0.9))  # x1, x2, y
    params = model.getParameters(data_hist)

    left = ROOT.RooArgSet()
    right = ROOT.RooArgSet()

    i = 0
    it = params.createIterator()
    p = it.Next()
    while p:
        if i % 2 == 0:
            left.add(p)
        else:
            right.add(p)
        i += 1
        p = it.Next()
#    model.paramOn(frame,ROOT.RooFit.Parameters(left),ROOT.RooFit.Layout(0.1, 0.5, 0.9))

#    model.paramOn(    frame,    ROOT.RooFit.Parameters(right),    ROOT.RooFit.Layout(0.55, 0.9, 0.9))

    
    c = ROOT.TCanvas("c", "c", 600, 600)
    ymax = frame.GetMaximum()
    #frame.SetMaximum(1.35 * ymax)
    frame.Draw()
    leg.Draw()
    print("SAVING ", filename)
    c.SaveAs(filename)
