import ROOT
from array import array
import numpy as np
import yaml
from pathlib import Path

def make_hist(name, values, weights, nbins, xmin, xmax):
    hist = ROOT.TH1F(name, name, nbins - 1, xmin, xmax)
    arr_values = array('d', values)
    arr_weights = array('d', weights)
    hist.FillN(len(values), arr_values, arr_weights)
    return hist

def make_roodatahist(name, hist, x_var):
    return ROOT.RooDataHist(name, name, ROOT.RooArgList(x_var), hist)


def fit_sum_of_gaussians(x, datahist, max_gaussians=5, mean=(90.,50.,300.), sigma=(10.,1.,300.), particle="Z", verbose=True):
    mean, mean_min, mean_max = mean
    sigma_, sigma_min, sigma_max = sigma
    results = []

    for n in range(1, max_gaussians + 1):
        gaussians = ROOT.RooArgList()
        coeffs = ROOT.RooArgList()


        components = []
        parameters = []

        for i in range(n):
            mu = ROOT.RooRealVar(f"mu{particle}_{n}_{i}", f"mu{particle}_{n}_{i}", mean, mean_min, mean_max)
            sigma = ROOT.RooRealVar(f"sigma{particle}_{n}_{i}", f"sigma{particle}_{n}_{i}", sigma_, sigma_min, sigma_max)

            gaus = ROOT.RooGaussian(f"g{particle}_{n}_{i}", f"g{particle}_{n}_{i}", x, mu, sigma)

            gaussians.add(gaus)

            components.append(gaus)
            parameters.extend([mu, sigma])

            if i < n - 1:
                frac = ROOT.RooRealVar(f"frac{particle}_{n}_{i}", f"frac{particle}_{n}_{i}", 0.5, 0.0, 1.0)
                coeffs.add(frac)
                parameters.append(frac)

        model = ROOT.RooAddPdf(f"model_{particle}{n}", f"model_{particle}{n}", gaussians, coeffs, True)

        # also keep model alive
        components.append(model)

        fit_result = model.fitTo(
            datahist,
            ROOT.RooFit.Save(),
            ROOT.RooFit.PrintLevel(-1)
        )

        frame = x.frame()
        datahist.plotOn(frame)
        model.plotOn(frame)

        chi2_ndof = frame.chiSquare()

        results.append({
            "n": n,
            "model": model,
            "fit_result": fit_result,
            "chi2_ndof": chi2_ndof,
            "components": components,
            "parameters": parameters
        })

        if verbose:
            print(f"[n = {n}] chi2/ndof = {chi2_ndof:.4f}")

    best = min(results, key=lambda r: r["chi2_ndof"])

    print("\nBest model:")
    print(f"n = {best['n']} with chi2/ndof = {best['chi2_ndof']:.4f}")

    return best, results


def save_syst_variation(cat, particle, syst_name, value):
    yaml_file = Path(
        f"/t3home/gcelotto/ggHbb/WSFit/Configs/systematics/"
        f"cat{cat}_{particle}syst.yaml"
    )

    if yaml_file.exists():
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    data[syst_name] = float(value)

    with open(yaml_file, "w") as f:
        yaml.safe_dump(data, f)

def compute_lnn(Ni, Nf):
    """
    Ni : nominal normalization (scalar)
    Nf : array of varied normalizations
    """
    delta = np.max(np.abs(Nf - Ni))
    return 1 + delta / Ni

def apply_syst(df, syst, cat, particle):

    Ni = df["weight"].sum()

    weight_variations = {
        "puid_up": (
            df["jet_pileupId_SF_up"] /
            df["jet_pileupId_SF_nom"]
        ),
        "puid_down": (
            df["jet_pileupId_SF_down"] /
            df["jet_pileupId_SF_nom"]
        ),
        "btag_hf_up": (
            df["btag_sf_hf_up"] /
            df["btag_sf_hf_central"]
        ),
        "btag_hf_down": (
            df["btag_sf_hf_down"] /
            df["btag_sf_hf_central"]
        ),
        "btag_lightf_up": (
            df["btag_sf_light_up"] /
            df["btag_sf_lightf_central"]
        ),
        "btag_lightf_down": (
            df["btag_sf_light_down"] /
            df["btag_sf_lightf_central"]
        ),
    }

    # -----------------------------------------
    # Simple weight systematics
    # -----------------------------------------
    if syst in weight_variations:

        df["weight"] *= weight_variations[syst]

        Nf = df["weight"].sum()
        lnN = Nf / Ni

        save_syst_variation(
            cat,
            particle,
            syst,
            lnN
        )

        return df

    # -----------------------------------------
    # Scale variations
    # -----------------------------------------
    elif syst == "scale":

        scale_cols = [
            "LHEScaleWeight_MuFdown_MuRdown",
            "LHEScaleWeight_MuFdown_MuRnom",
            "LHEScaleWeight_MuFdown_MuRup",
            "LHEScaleWeight_MuFnom_MuRdown",
            "LHEScaleWeight_MuFnom_MuRup",
            "LHEScaleWeight_MuFup_MuRdown",
            "LHEScaleWeight_MuFup_MuRnom",
            "LHEScaleWeight_MuFup_MuRup",
        ]

        scale_weights = np.stack(
            [df["weight"] * df[col] for col in scale_cols],
            axis=0
        )

        Nf = scale_weights.sum(axis=1)

        lnN = compute_lnn(Ni, Nf)

        save_syst_variation(
            cat,
            particle,
            "Scale",
            lnN
        )

        return None

    # -----------------------------------------
    # PS ISR
    # -----------------------------------------
    elif syst == "PS_ISR":

        ps_cols = [
            "PSWeight_ISRdown_FSRnom",
            "PSWeight_ISRup_FSRnom",
        ]

        ps_weights = np.stack(
            [df["weight"] * df[col] for col in ps_cols],
            axis=0
        )

        Nf = ps_weights.sum(axis=1)

        lnN = compute_lnn(Ni, Nf)

        save_syst_variation(
            cat,
            particle,
            "PS_ISR",
            lnN
        )

        return None

    # -----------------------------------------
    # PS FSR
    # -----------------------------------------
    elif syst == "PS_FSR":

        ps_cols = [
            "PSWeight_ISRnom_FSRdown",
            "PSWeight_ISRnom_FSRup",
        ]

        ps_weights = np.stack(
            [df["weight"] * df[col] for col in ps_cols],
            axis=0
        )

        Nf = ps_weights.sum(axis=1)

        lnN = compute_lnn(Ni, Nf)

        save_syst_variation(
            cat,
            particle,
            "PS_FSR",
            lnN
        )

        return None

    # -----------------------------------------
    # alphaS
    # -----------------------------------------
    elif syst == "alphaS":

        alpha_cols = [
            "LHEAlphasWeight_down",
            "LHEAlphasWeight_up",
        ]

        alpha_weights = np.stack(
            [df["weight"] * df[col] for col in alpha_cols],
            axis=0
        )

        Nf = alpha_weights.sum(axis=1)

        lnN = compute_lnn(Ni, Nf)

        save_syst_variation(
            cat,
            particle,
            "alphaS",
            lnN
        )

        return None

    return df

def plot_model(x_var, data_hist, model, components=[], filename="plot.png", title="Fit"):
    x_var.SetTitle("m_{bb} [GeV]")
    frame = x_var.frame(ROOT.RooFit.Title(title))
    data_hist.plotOn(frame, ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2), ROOT.RooFit.Name("data"), ROOT.RooFit.MarkerStyle(20), ROOT.RooFit.MarkerSize(0.6))
    
    model.plotOn(frame, ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.Name("model"))
    chi2 = frame.chiSquare()

    print("chi2/ndof =", chi2)
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
    c.Divide(1, 2)

    # Pad superiore: plot principale
    pad1 = c.cd(1)
    pad1.SetPad(0, 0.3, 1, 1.0)
    pad1.SetBottomMargin(0.02)
    pad1.SetGridx()

    frame.Draw()
    leg.Draw()

    # Pad inferiore: pull histogram
    pad2 = c.cd(2)
    pad2.SetPad(0, 0.0, 1, 0.3)
    pad2.SetTopMargin(0.04)
    pad2.SetBottomMargin(0.35)
    pad2.SetGridx()

    # Crea il frame dei pull dal RooFit frame
    pull_hist = frame.pullHist("data", "model")

    # Crea un nuovo frame per i pull (stessi limiti x del frame principale)
    x_var = frame.getPlotVar()
    pull_frame = x_var.frame(
        ROOT.RooFit.Title(""),
        ROOT.RooFit.Range(frame.GetXaxis().GetXmin(), frame.GetXaxis().GetXmax())
    )

    pull_hist.SetMarkerSize(0.6)
    pull_frame.addPlotable(pull_hist, "P")
    pull_frame.SetTitle("")
    # Formattazione pull frame
    pull_frame.GetYaxis().SetTitle("Pull")
    pull_frame.GetYaxis().SetNdivisions(505)
    pull_frame.GetYaxis().SetTitleSize(0.15)
    pull_frame.GetYaxis().SetTitleOffset(0.3)
    pull_frame.GetYaxis().SetLabelSize(0.12)
    pull_frame.GetXaxis().SetTitleSize(0.15)
    pull_frame.GetXaxis().SetLabelSize(0.12)
    pull_frame.GetYaxis().SetRangeUser(-3, 3)

    # Linea di riferimento a zero
    line = ROOT.TLine(
        frame.GetXaxis().GetXmin(), 0,
        frame.GetXaxis().GetXmax(), 0
    )
    line.SetLineColor(ROOT.kRed)
    line.SetLineStyle(2)

    pull_frame.Draw()
    line.Draw("same")

    print("SAVING ", filename)
    c.SaveAs(filename)
