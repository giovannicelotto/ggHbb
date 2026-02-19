# Take the ftest results and add the Higgs PDF and related objects to the multipdf workspace
import ROOT
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Enrich multipdf workspace with extra PDF.")
    parser.add_argument("-c", "--category", type=int, help="Index of the workspace")
    args = parser.parse_args()



    # Input files
    ws1_path = f"/t3home/gcelotto/ggHbb/WSFit/ws/step1/ws{args.category}_nominal.root"
    multipdf_path = f"/t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdf/multipdf_{args.category}.root"
    out_path = f"/t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdfEnriched/ws{args.category}.root"

    # Open files
    f1 = ROOT.TFile.Open(ws1_path)
    f2 = ROOT.TFile.Open(multipdf_path, "READ")

    if not f1 or f1.IsZombie():
        raise RuntimeError(f"Cannot open {ws1_path}")
    if not f2 or f2.IsZombie():
        raise RuntimeError(f"Cannot open {multipdf_path}")

    # Get workspaces
    ws1 = f1.Get("WS")
    ws2 = f2.Get("multipdf")

    if not ws1:
        raise RuntimeError("Workspace 'WS' not found in first file")
    if not ws2:
        raise RuntimeError("Workspace 'multipdf' not found in multipdf file")

    # Get PDF
    Higgs_pdf_name = f"model_H_c{args.category}"
    Hpdf = ws1.pdf(Higgs_pdf_name)
    if not Hpdf:
        raise RuntimeError(f"PDF {Higgs_pdf_name} not found in first workspace")
    



    Z_pdf_name = f"model_Z_c{args.category}"
    Zpdf = ws1.pdf(Z_pdf_name)
    


    nH = ws1.var("nH_cat%d"%args.category)
    nZ = ws1.var("nZ_cat%d"%args.category)
    nBkg = ws2.var("CMS_hgg_0_2016_13TeV_bkg_norm")

    # Create new variables
    model_H_c_norm = ROOT.RooRealVar("model_H_c%d_norm"%args.category, "model_H_c%d_norm"%args.category, nH.getVal()*1, 0, nH.getVal()*5)
    model_H_c_norm.setConstant(True)
    model_Z_c_norm = ROOT.RooRealVar("model_Z_c%d_norm"%args.category, "model_Z_c%d_norm"%args.category, nZ.getVal()*1, 0, nZ.getVal()*5)
    model_Z_c_norm.setConstant(True)
    pdfname = f"CMS_hgg_0_2016_13TeV_bkg"
    bkg_norm = ROOT.RooRealVar(pdfname+"_noZ_norm", pdfname+"_noZ_norm", nBkg.getVal()-nZ.getVal(), nBkg.getVal()*0.5, nBkg.getVal()*1.5)

    higgs_hist_name = f"rooHist_H_cat{args.category}"
    higgs_hist = ws1.data(higgs_hist_name)

    Z_hist_name = f"rooHist_Z_cat{args.category}"
    Z_hist = ws1.data(Z_hist_name)

    if not higgs_hist:
        raise RuntimeError(f"RooDataHist '{higgs_hist_name}' not found in WS")
    

# SAve MultiPDF
    
    multipdf = ws2.pdf(pdfname)
    if not multipdf:
        raise RuntimeError(f"MultiPdf {pdfname} not found in workspace")

    pdfindex = ws2.cat(f"pdfindex_{args.category}_2016_13TeV")
    if not pdfindex:
        raise RuntimeError(f"Index variable not found: pdfindex_{args.category}_2016_13TeV")

    # Extract the PDFs inside the multipdf
    n_pdfs = multipdf.getNumPdfs()
    print(f"Found {n_pdfs} PDFs in multipdf — extracting background components (non-Z).")

    pdf_list_noz = ROOT.RooArgList()

    for i in range(n_pdfs):
        pdf = multipdf.getPdf(i)
        if not pdf:
            raise RuntimeError(f"Could not get PDF {i} from multipdf")

        if not isinstance(pdf, ROOT.RooAddPdf):
            print(f"Warning: PDF {pdf.GetName()} is not a RooAddPdf — adding as is")
            pdf_list_noz.add(pdf)
            continue

        # RooAddPdf components
        comp_list = pdf.pdfList()
        if comp_list.getSize() < 2:
            raise RuntimeError(f"Expected at least 2 components in {pdf.GetName()}")

        # Take the second component (the non-Z one)
        non_z_pdf = comp_list.at(1)
        if not non_z_pdf:
            raise RuntimeError(f"Cannot get non-Z component from {pdf.GetName()}")

        new_name = pdf.GetName().replace("_with_z", "")
        new_pdf = non_z_pdf.clone(new_name)
        new_pdf.Print("t")
        pdf_list_noz.add(new_pdf)
        print(f"Extracted {new_name}")
    import yaml  # add at the top with other imports

    # YAML creation
    pdf_names = [pdf_list_noz[i].GetName() for i in range(pdf_list_noz.getSize())]
    yaml_path = os.path.splitext(out_path)[0] + "_pdfnames.yaml"
    with open(yaml_path, "w") as f_yaml:
        yaml.dump({"pdf_names": pdf_names}, f_yaml)
    print(f"PDF names written to {yaml_path}")

    
    # Build the new multipdf
    multipdf_noz = ROOT.RooMultiPdf(f"{pdfname}_noZ",
                                    f"Background-only multipdf (no Z)",
                                    pdfindex,
                                    pdf_list_noz)

    # Save to a new wor



    # Import PDF into second workspace
    getattr(ws2, "import")(Hpdf, ROOT.RooFit.RecycleConflictNodes())
    getattr(ws2, "import")(Zpdf, ROOT.RooFit.RecycleConflictNodes())
    getattr(ws2, "import")(higgs_hist)
    getattr(ws2, "import")(Z_hist)
    getattr(ws2, "import")(model_H_c_norm)
    getattr(ws2, "import")(model_Z_c_norm)
    getattr(ws2, "import")(multipdf_noz)
    getattr(ws2, "import")(pdfindex)
    getattr(ws2, "import")(bkg_norm)

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