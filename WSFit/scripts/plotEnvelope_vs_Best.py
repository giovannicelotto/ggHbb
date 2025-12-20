#  Use cmsenv from CMSSW environment 
#   
import ROOT
import argparse
import mplhep as hep
hep.style.use("CMS")
import os
import matplotlib.pyplot as plt
import numpy as np
def main():
    parser = argparse.ArgumentParser(description="Enrich multipdf workspace with extra PDF.")
    parser.add_argument("--idx", type=int, help="Index of the workspace", default=0)
    args = parser.parse_args()

    idx = args.idx
    
    
    multipdf_path = f"/t3home/gcelotto/ggHbb/WSFit/ws/stepMultiPdf/multipdf_{idx}.root"
    out_path = f"/t3home/gcelotto/ggHbb/CMSSW_14_1_0_pre4/src/flashggFinalFit/Background/plots/fTest_5families_functionalities_cat{idx}"

    # Open files
    f2 = ROOT.TFile.Open(multipdf_path, "READ")
    if not f2 or f2.IsZombie():
        raise RuntimeError(f"Cannot open {multipdf_path}")
    # Get workspaces
    ws2 = f2.Get("multipdf")
    if not ws2:
        raise RuntimeError("Workspace 'multipdf' not found in multipdf file")
    # Get PDF
    pdfname = f"CMS_hgg_0_2016_13TeV_bkgshape"
    bkg_pdf = ws2.pdf(pdfname)
    if not bkg_pdf:
        raise RuntimeError(f"PDF {pdfname} not found in multipdf workspace")
    
    # Select index of Pdf
    #  OBJ: RooCategory       pdfindex_0_2016_13TeV   c : 0 at: 0x553d740 OBJ: RooCategory       pdfindex_0_2016_13TeV   c : 0 at: 0x553d740
    cat = ws2.cat(f"pdfindex_{idx}_2016_13TeV")
    if not cat:
        raise RuntimeError(f"Indexed PDF pdfindex_{idx}_2016_13TeV not found in multipdf workspace")
    
    
    # Plot comparison
    x = ws2.var(f"dijet_mass_c{idx}")
    
    y_bkg_indexed = []
    # Select category for indexed PDF
    pdfs=[]
    pdfsNames = []
    it = cat.typeIterator()
    t = it.Next()
    # Current value of Cat
    bestPdfIdx = cat.getIndex()


    while t:
        cat.setIndex(cat.lookupType(t.GetName()).getVal())
        print("Index:", cat.lookupType(t.GetName()).getVal(), "PDF : ", bkg_pdf.getCurrentPdf().GetName())
        pdfs.append(bkg_pdf.getCurrentPdf())
        pdfsNames.append(bkg_pdf.getCurrentPdf().GetName())
        t = it.Next()
    xvals = np.linspace(x.getMin(), x.getMax(), 1000)
    
    fig, ax = plt.subplots(2,1, figsize=(10, 10), sharex=True, height_ratios=[3,1])
    
    it = cat.typeIterator()
    t = it.Next()
    while t:
        catIdx = cat.lookupType(t.GetName()).getVal()
        cat.setIndex(catIdx)
        print("Index:", catIdx, "PDF : ", bkg_pdf.getCurrentPdf().GetName())
        y_bkg = []
        for xv in xvals:
            x.setVal(xv)
            y_bkg.append(pdfs[catIdx].getVal(ROOT.RooArgSet(x)))
        ax[0].plot(xvals, y_bkg, label=pdfsNames[catIdx][8:-7])
        t = it.Next()


    it = cat.typeIterator()
    t = it.Next()
    while t:
        catIdx = cat.lookupType(t.GetName()).getVal()
        cat.setIndex(catIdx)
        print("Index:", catIdx, "PDF : ", bkg_pdf.getCurrentPdf().GetName())
        y_bkg = []
        bestPdfValues = []
        for xv in xvals:
            x.setVal(xv)
            y_bkg.append(pdfs[catIdx].getVal(ROOT.RooArgSet(x)))
            bestPdfValues.append(pdfs[bestPdfIdx].getVal(ROOT.RooArgSet(x)))
        ax[1].plot(xvals, (np.array(y_bkg)-np.array(bestPdfValues))/np.array(bestPdfValues), label=pdfsNames[catIdx][8:-7])
        t = it.Next()
    ax[0].legend()
    ax[1].set_xlabel("Dijet Mass [GeV]")
    ax[0].set_ylabel("PDF Value")
    ax[1].set_ylabel("PDF Rel. Difference")
    ax[0].yaxis.label.set_position((0.5,1.))
    ax[1].yaxis.label.set_position((0.5, 1.4))
    ax[1].set_ylim(-0.015, 0.015)
    ax[1].set_xlim(x.getMin(), x.getMax())
    




    fig.savefig(os.path.join(out_path, f"bkg_pdf_indexed_cat{idx}.png"))
    print("Saved in ", os.path.join(out_path, f"bkg_pdf_indexed_cat{idx}.png"))
        
        

    #xmin = x.getMin()
    #xmax = x.getMax()
    
    
    #bin_centers = 0.5 * (bins[:-1] + bins[1:])
if __name__ == "__main__":
    main()