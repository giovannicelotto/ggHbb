# %%
import ROOT
import numpy as np
import pandas as pd
import yaml
import sys
import argparse
import os
import json
import mplhep as hep

from helpers.load_dfs import getDfsFromConfig
from signal_modeling_helpers import apply_syst, make_hist, make_roodatahist,    plot_model, fit_sum_of_gaussians
# %%
# -----------------------------
# Config & arguments
# -----------------------------

def parse_args(cfg):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="7")
    parser.add_argument('-s', '--syst', type=str, default=None)

    args = parser.parse_args([]) if hasattr(sys, 'ps1') or not sys.argv[1:] else parser.parse_args()

    yaml_systematics = cfg.get("systematics", [])
    if args.syst and args.syst not in yaml_systematics:
        parser.error(f"Systematic '{args.syst}' not in YAML: {yaml_systematics}")

    return args

# -----------------------------
# Setup
# -----------------------------

def setup_root():
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.FATAL)
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    ROOT.gROOT.SetBatch(True)
    ROOT.gSystem.CompileMacro(
        "/t3home/gcelotto/ggHbb/newFit/rooFit/helpersFunctions/RooDoubleCB.cc",
        "kf"
    )
    hep.style.use("CMS")





def load_main_config():
    with open("/t3home/gcelotto/ggHbb/WSFit/allSteps/step1_cfg.yaml") as f:
        return yaml.safe_load(f)


def resolve_config_path(template, config_id):
    path = template.replace("CONFIG", config_id)
    if not os.path.exists(path):
        print(f"[WARNING] Missing {path}, fallback to config 0")
        path = template.replace("CONFIG", "0")
    return path


# -----------------------------
# Data preparation
# -----------------------------

def load_dataframes(config_id):
    dfMC_Z, dfMC_H, df, nbins, nbins_MC, x1, x2 = getDfsFromConfig(cat_idx=int(config_id))
    print(f"[INFO] Limits: {x1}, {x2}, nbins={nbins}")
    return dfMC_Z, dfMC_H, df, nbins, nbins_MC, x1, x2


def apply_systematics_if_needed(dfMC_Z, dfMC_H, syst, config_id):
    if not syst:
        return dfMC_Z, dfMC_H

    print(f"[INFO] Applying systematic: {syst}")
    dfMC_Z = apply_syst(dfMC_Z, syst, config_id, particle="Z")
    dfMC_H = apply_syst(dfMC_H, syst, config_id, particle="H")

    if dfMC_H is None:
        print("[INFO] Scale systematic applied (lnN only)")
        sys.exit()

    return dfMC_Z, dfMC_H


# -----------------------------
# Histograms
# -----------------------------

def build_roohists(df, dfMC_Z, dfMC_H, nbins, nbins_MC, x1, x2, config_id):
    x = ROOT.RooRealVar(f"dijet_mass_c{config_id}", "", x1, x2)

    hist_data = make_hist(f"hist_data_{config_id}", df.dijet_mass, np.ones(len(df)), nbins, x1, x2)
    hist_H = make_hist(f"hist_H_{config_id}", dfMC_H.dijet_mass, dfMC_H.weight, nbins_MC, x1, x2)
    hist_Z = make_hist(f"hist_Z_{config_id}", dfMC_Z.dijet_mass, dfMC_Z.weight, nbins_MC, x1, x2)

    return (
        x,
        make_roodatahist(f"rooHist_data_cat{config_id}", hist_data, x),
        make_roodatahist(f"rooHist_Z_cat{config_id}", hist_Z, x),
        make_roodatahist(f"rooHist_H_cat{config_id}", hist_H, x),
    )


# -----------------------------
# Model building
# -----------------------------

def build_model(x, rooHist, mean_guess, xmin_fit, xmax_fit, particle="Z", category_number=7,tag="nominal"):
    x.setRange("fit_range", xmin_fit, xmax_fit)
    best, _ = fit_sum_of_gaussians(
        x, rooHist,
        max_gaussians=5,
        mean=mean_guess,
        sigma=(10., 4., 300.),
        particle=particle,
        category_number=category_number,
        tag=tag
    )

    model = best["model"]

    fit_result = model.fitTo(
        rooHist,
        ROOT.RooFit.Range("fit_range"),
        ROOT.RooFit.NormRange("fit_range"),
        ROOT.RooFit.Extended(False),
        ROOT.RooFit.Binned(True),
        ROOT.RooFit.SumW2Error(True),
        ROOT.RooFit.Save()
    )

    print("[INFO] Fit result:")
    fit_result.Print("v")

    # Freeze parameters
    for p in best["parameters"]:
        p.setConstant(True)

    return model, best


# -----------------------------
# Extended PDFs
# -----------------------------

def build_extended(dfMC_Z, dfMC_H, config_id):
    nZ = ROOT.RooRealVar(f"nZ_cat{config_id}", "", np.sum(dfMC_Z.weight), 0, 5e4)
    nH = ROOT.RooRealVar(f"nH_cat{config_id}", "", np.sum(dfMC_H.weight), 0, 5e4)
    nZ.setConstant(True)
    nH.setConstant(True)
    return nZ, nH

    mu_Z = ROOT.RooRealVar("mu_Z", "", 1.0, 0.0, 2.0)
    mu_H = ROOT.RooRealVar("mu_H", "", 1.0, 0.0, 2.0)

    ext_Z = ROOT.RooExtendPdf(
        f"ext_Z_{config_id}", "",
        model_Z,
        ROOT.RooFormulaVar("nZmu", "@0*@1", ROOT.RooArgList(nZ, mu_Z))
    )

    ext_H = ROOT.RooExtendPdf(
        f"ext_H_{config_id}", "",
        model_H,
        ROOT.RooFormulaVar("nHmu", "@0*@1", ROOT.RooArgList(nH, mu_H))
    )

    return ext_Z, ext_H


# -----------------------------
# Workspace
# -----------------------------

def build_workspace(objects, output_path):
    w = ROOT.RooWorkspace("WS", "WS")

    for obj in objects:
        name = obj.GetName()
        if isinstance(obj, ROOT.RooAbsData):
            if not w.data(name):
                getattr(w, "import")(obj)
        else:
            if not w.arg(name):
                getattr(w, "import")(obj, ROOT.RooFit.RecycleConflictNodes())

    f_out = ROOT.TFile(output_path, "RECREATE")
    print(f"[INFO] Saving workspace to {output_path}...")
    w.Write()
    f_out.Close()


# -----------------------------
# Main
# -----------------------------
# %%

#if __name__ == "__main__":
setup_root()

cfg = load_main_config()
args = parse_args(cfg)
# %%

base_dfMC_Z, base_dfMC_H, df, nbins, nbins_MC, x1, x2 = load_dataframes(args.config)
# %%
systematics = [None] + cfg.get("systematics", [])

for syst in systematics:
    tag = syst if syst else "nominal"

    print(f"\n[INFO] Running systematic: {syst or 'nominal'}")


    dfMC_Z = base_dfMC_Z.copy()
    dfMC_H = base_dfMC_H.copy()

    # Apply systematic
    dfMC_Z, dfMC_H = apply_systematics_if_needed(dfMC_Z, dfMC_H, syst, args.config)

    # Build RooFit objects
    x, roo_data, roo_Z, roo_H = build_roohists(
        df, dfMC_Z, dfMC_H, nbins, nbins_MC, x1, x2, args.config
    )

    model_H, best_H = build_model(x, roo_H, (125., 50., 300.), 50, 300, category_number=(args.config), particle="H", tag=tag)
    model_Z, best_Z = build_model(x, roo_Z, (90., 50., 300.), 50, 300, category_number=(args.config), particle="Z", tag=tag)

    nZ, nH = build_extended(dfMC_Z, dfMC_H, args.config)


    plot_model(
        x, roo_Z, model_Z,
        components=[{'name': f'g_{best_Z["n"]}_{i}', 'color': ROOT.kBlue+i, 'style': ROOT.kDashed, 'label': f"G{i}"} for i in range(best_Z["n"])],
        filename=cfg["output_Z"].replace("CONFIG", args.config).replace("SYST", tag),
        title=""
    )

    plot_model(
        x, roo_H, model_H,
        components=[{'name': f'g_{best_H["n"]}_{i}', 'color': ROOT.kBlue+i, 'style': ROOT.kDashed, 'label': f"G{i}"} for i in range(best_H["n"])],
        filename=cfg["output_H"].replace("CONFIG", args.config).replace("SYST", tag),
        title=""
    )

    output_ws = cfg["output_ws"].replace("CONFIG", args.config).replace("SYST", tag)

    build_workspace(
        [x, roo_data, roo_Z, roo_H, model_Z, model_H, nZ, nH],
        output_ws
    )

    print(f"[INFO] Z yield: {np.sum(dfMC_Z.weight)}")
    print(f"[INFO] H yield: {np.sum(dfMC_H.weight)}")