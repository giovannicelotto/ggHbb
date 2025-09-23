import ROOT
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Enrich multipdf workspace with extra PDF.")
    parser.add_argument("idx", type=int, help="Index of the workspace")
    args = parser.parse_args()

    idx = args.idx

    # Input files
    ws1_path = f"/t3home/gcelotto/ggHbb/WSFit/ws/step1/ws{idx}.root"
    multipdf_path = f"/t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdf/multipdf_{idx}.root"
    out_path = f"/t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdfEnriched/ws{idx}.root"

    # Open files
    f1 = ROOT.TFile.Open(ws1_path)
    f2 = ROOT.TFile.Open(multipdf_path, "READ")

    if not f1 or f1.IsZombie():
        raise RuntimeError(f"Cannot open {ws1_path}")
    if not f2 or f2.IsZombie():
        raise RuntimeError(f"Cannot open {multipdf_path}")

    # Get workspaces
    ws1 = f1.Get("workspace_sig")
    ws2 = f2.Get("multipdf")

    if not ws1:
        raise RuntimeError("Workspace 'workspace_sig' not found in first file")
    if not ws2:
        raise RuntimeError("Workspace 'multipdf' not found in multipdf file")

    # Get PDF
    pdf_name = f"model_H_c{idx}"
    pdf = ws1.pdf(pdf_name)
    if not pdf:
        raise RuntimeError(f"PDF {pdf_name} not found in first workspace")
    nH = ws1.var("nH_cat%d"%idx)
    model_H_c_norm = ROOT.RooRealVar("model_H_c%d_norm"%idx, "model_H_c%d_norm"%idx, nH.getVal()*1, 0, nH.getVal()*5)

    higgs_hist_name = f"rooHist_H_cat{idx}"
    higgs_hist = ws1.data(higgs_hist_name)
    if not higgs_hist:
        raise RuntimeError(f"RooDataHist '{higgs_hist_name}' not found in ws1")



    # Import PDF into second workspace
    getattr(ws2, "import")(pdf, ROOT.RooFit.RecycleConflictNodes())
    getattr(ws2, "import")(higgs_hist, ROOT.RooFit.RecycleConflictNodes())
    getattr(ws2, "import")(model_H_c_norm)

    # Save enriched workspace
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fout = ROOT.TFile(out_path, "RECREATE")
    ws2.Write("ws3")
    fout.Close()

    f1.Close()
    f2.Close()

    print(f"Enriched workspace saved to {out_path}")

if __name__ == "__main__":
    main()