import ROOT
import argparse
import yaml
import os
parser = argparse.ArgumentParser(description="Enrich multipdf workspace with extra PDF.")
parser.add_argument("-c", "--category", type=int, help="Index of the workspace")
args = parser.parse_args()
source_file = f"/t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdfEnriched/ws{args.category}.root"
target_file = f"/t3home/gcelotto/ggHbb/WSFit/ws/step5/ws{args.category}.root"

f_source = ROOT.TFile.Open(source_file, "READ")
ws_source = f_source.Get("ws3")
if not ws_source:
    raise RuntimeError("Workspace 'ws3' not found in source file")

# Clone the workspace
ws_clone = ws_source.Clone("ws3")  # clone with same name
f_target = ROOT.TFile.Open(target_file, "RECREATE")

with open("/t3home/gcelotto/ggHbb/WSFit/allSteps/step1_cfg.yaml", 'r') as stream:
    cfg = yaml.safe_load(stream)
variations = cfg["systematics"]
if variations is not None:
    for var in variations:
        print(f"Variation {var}")
        in_path = f"/t3home/gcelotto/ggHbb/WSFit/ws/step1/ws{args.category}_{var}.root"
        if not os.path.exists(in_path):
            print("Scale uncertainty")
            continue
        f_var = ROOT.TFile.Open(in_path, "READ")
        ws_var = f_var.Get("WS")

        H_pdf_name = f"model_H_c{args.category}_{var.replace('_up', 'Up').replace('_down', 'Down')}"
        pdf = ws_var.pdf(H_pdf_name)
        if not pdf:
            raise RuntimeError(f"PDF {H_pdf_name} not found in {in_path}")
        ws_clone._import(pdf)

        Z_pdf_name = f"model_Z_c{args.category}_{var.replace('_up', 'Up').replace('_down', 'Down')}"
        pdf = ws_var.pdf(Z_pdf_name)

        if not pdf:
            raise RuntimeError(f"PDF {Z_pdf_name} not found in {in_path}")
        ws_clone._import(pdf)

        f_var.Close()
f_target.cd()
ws_clone.Write()
f_target.Close()
f_source.Close()
