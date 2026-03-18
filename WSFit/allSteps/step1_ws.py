# %%
# Opens the step1_cfg.yaml
# Checks if the systematic called is present in the yaml
# Takes the path of Z and H config files for the initial parameters of the fit
# Opens the dataframes
# apply systematic variation
# Build roodatahist
# Build models
import ROOT
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.FATAL)
ROOT.gErrorIgnoreLevel = ROOT.kError
ROOT.gErrorIgnoreLevel = ROOT.kWarning
import numpy as np
import pandas as pd
import yaml
import sys
import argparse
from array import array
import mplhep as hep
import os
import json
# Custom modules
from helpers.getDfsFromConfig import getDfsFromConfig
from functions import cut, getDfProcesses_v2
from step1_ws_helpers import apply_syst, make_hist, make_roodatahist, build_dscb_gaus_model, plot_model
# Style
hep.style.use("CMS")
ROOT.gROOT.SetBatch(True)
ROOT.gSystem.CompileMacro("/t3home/gcelotto/ggHbb/newFit/rooFit/helpersFunctions/RooDoubleCB.cc", "kf")


# %%
# --- Arguments ---
with open("/t3home/gcelotto/ggHbb/WSFit/allSteps/step1_cfg.yaml", 'r') as stream:
    cfg = yaml.safe_load(stream)
yaml_systematics = cfg.get("systematics", [])
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default="0", help='Config File')
parser.add_argument("-s", "--syst",    type=str,    default=None,    help="Which systematic to apply")
args = parser.parse_args([]) if hasattr(sys, 'ps1') or not sys.argv[1:] else parser.parse_args()

if args.syst is not None and args.syst not in yaml_systematics:
    parser.error(
        f"systematics not present in yaml file: '{args.syst}'. "
        f"Available systematics: {yaml_systematics}")
# ---

dfMC_Z, dfMC_H, df, nbins, x1, x2 = getDfsFromConfig(idx=int(args.config))
print(f"[INFO] Limits on RooRealVar: {x1}, {x2}, nbins={nbins}")


config_path_Z = cfg["config_path_Z"].replace("CONFIG", args.config)
config_path_H = cfg["config_path_H"].replace("CONFIG", args.config)
if not os.path.exists(config_path_Z):
    print(f"[WARNING] Config file for Z not found at {config_path_Z}. Trying with config 0.")
    config_path_Z = cfg["config_path_Z"].replace("CONFIG", "0")
if not os.path.exists(config_path_H):
    print(f"[WARNING] Config file for Z not found at {config_path_Z}. Trying with config 0.")
    config_path_H = cfg["config_path_H"].replace("CONFIG", "0")


if args.syst:
    print(f"[INFO] Applying systematic variation: {args.syst}")
    dfMC_Z = apply_syst(dfMC_Z, args.syst, args.config, particle="Z")
    dfMC_H = apply_syst(dfMC_H, args.syst, args.config, particle="H")
    if dfMC_H is None:
        print(f"[INFO] Scale systematic applied. Check the corresponding YAML for lnN values.")
        sys.exit()



# %%
# --- Histograms & RooDataHists ---
print(f"dijet_mass_c{args.config}")
x_cat = ROOT.RooRealVar(f"dijet_mass_c{args.config}", f"dijet_mass_c{args.config}", x1, x2)

hist_data_cat = make_hist(f"hist_data_cat{args.config}", df.dijet_mass, np.ones(len(df)), nbins, x1, x2)
hist_Z_cat = make_hist(f"hist_Z_cat{args.config}", dfMC_Z.dijet_mass, dfMC_Z.weight, nbins, x1, x2)
hist_H_cat = make_hist(f"hist_H_cat{args.config}", dfMC_H.dijet_mass, dfMC_H.weight, nbins, x1, x2)

rooHist_data_cat = make_roodatahist(f"rooHist_data_cat{args.config}", hist_data_cat, x_cat)
rooHist_Z_cat = make_roodatahist(f"rooHist_Z_cat{args.config}", hist_Z_cat, x_cat)
rooHist_H_cat = make_roodatahist(f"rooHist_H_cat{args.config}", hist_H_cat, x_cat)

# %%
# --- Z model ---
with open(config_path_Z, 'r') as f:
    config_Z = yaml.safe_load(f)
ext = f"_{(args.syst).replace('_up', 'Up').replace('_down','Down')}" if args.syst else ""
model_Z_c, dscb_Z, gaus_Z, params_Z = build_dscb_gaus_model(x_cat, config_Z, f"Z_c{args.config}{ext}")
model_Z_c.Print()
rooHist_Z_cat.Print()
# Fit
model_Z_c.fitTo(rooHist_Z_cat, ROOT.RooFit.Extended(False), ROOT.RooFit.Binned(True), ROOT.RooFit.SumW2Error(True))


# Plot
plot_model(
    x_cat, rooHist_Z_cat, model_Z_c,
    components=[{'name': f'dscb_Z_c{args.config}{ext}', 'color': ROOT.kBlue, 'style': ROOT.kDashed, 'label': "DSCB"},
                {'name': f'gauss_Z_c{args.config}{ext}', 'color': ROOT.kGreen+2, 'style': ROOT.kDotted, 'label': "Gauss"}],
    filename=cfg["output_Z"].replace("CONFIG", args.config).replace("SYST", args.syst if args.syst else "nominal"),
    title=" "
)

# Freeze Z parameters
# Default category

outpath = "/t3home/gcelotto/ggHbb/WSFit/output/cat0/fit_params_Z.json"
category = args.syst if args.syst is not None else "nominal"

# Load existing JSON if it exists
if os.path.exists(outpath):
    with open(outpath, "r") as f:
        all_params = json.load(f)
else:
    all_params = {}

# Overwrite / add only the chosen category
all_params[category] = {
    p.GetName(): {
        "value": float(p.getVal()),
        "error": float(p.getError())
    }
    for p in params_Z
}

# Freeze parameters
for p in params_Z:
    p.setConstant(True)

# Write back the full JSON
with open(outpath, "w") as f:
    json.dump(all_params, f, indent=2)

# %%
# --- Higgs model ---
with open(config_path_H, 'r') as f:
    config_H = yaml.safe_load(f)

model_H_c, dscb_H, gaus_H, params_H = build_dscb_gaus_model(x_cat, config_H, f"H_c{args.config}{ext}")
model_H_c.fitTo(rooHist_H_cat, ROOT.RooFit.Extended(False), ROOT.RooFit.Binned(True), ROOT.RooFit.SumW2Error(True))

plot_model(
    x_cat, rooHist_H_cat, model_H_c,
    components=[{'name': f'dscb_H_c{args.config}{ext}', 'color': ROOT.kBlue, 'style': ROOT.kDashed, 'label': "DSCB"},
                {'name': f'gauss_H_c{args.config}{ext}', 'color': ROOT.kGreen+2, 'style': ROOT.kDotted, 'label': "Gauss"}],
    filename=cfg["output_H"].replace("CONFIG", args.config).replace("SYST", args.syst if args.syst else "nominal"),
    title=" ")

# Freeze H parameters
for p in params_H:
    p.setConstant(True)

# %%
# --- Extended yields ---
nZ_cat = ROOT.RooRealVar(f"nZ_cat{args.config}", f"nZ_cat{args.config}", np.sum(dfMC_Z.weight), 0, 5e4)
nZ_cat.setConstant(True)

nH_cat = ROOT.RooRealVar(f"nH_cat{args.config}", f"nH_cat{args.config}", np.sum(dfMC_H.weight), 0, 5e4)
nH_cat.setConstant(True)

mu_Z = ROOT.RooRealVar("mu_Z", "signal strength", 1.0, 0.0, 2.0)
mu_H = ROOT.RooRealVar("mu_H", "signal strength", 1.0, 0.0, 2.0)

nZ_times_mu = ROOT.RooFormulaVar(f"nZ_times_mu{args.config}", "@0*@1", ROOT.RooArgList(nZ_cat, mu_Z))
extended_Z_cat = ROOT.RooExtendPdf(f"extended_Z_cat{args.config}", f"extended_Z_cat{args.config}", model_Z_c, nZ_times_mu)

nH_times_mu = ROOT.RooFormulaVar(f"nH_times_mu{args.config}", "@0*@1", ROOT.RooArgList(nH_cat, mu_H))
extended_H_cat = ROOT.RooExtendPdf(f"extended_H_cat{args.config}", f"extended_H_cat{args.config}", model_H_c, nH_times_mu)
# %%
# --- Workspace ---
w_sig = ROOT.RooWorkspace("WS", "WS")
to_import = [x_cat, rooHist_data_cat, rooHist_Z_cat, model_H_c, model_Z_c, rooHist_H_cat, extended_Z_cat, extended_H_cat]

for obj in to_import:
    name = obj.GetName()
    if isinstance(obj, ROOT.RooAbsData):
        if not w_sig.data(name):
            getattr(w_sig, "import")(obj)
    else:
        if not w_sig.arg(name):
            getattr(w_sig, "import")(obj, ROOT.RooFit.RecycleConflictNodes())

# Save workspace
f_out = ROOT.TFile(cfg["output_ws"].replace("CONFIG", args.config).replace("SYST", args.syst if args.syst else "nominal"), "RECREATE")
w_sig.Write()
f_out.Close()

print(f"[INFO] Z events: {np.sum(dfMC_Z.weight)}, H events: {np.sum(dfMC_H.weight)}")
